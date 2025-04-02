#!/usr/bin/env python3
from typing import List, cast

import numpy as np
import open3d as o3d
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
import rclpy.time

# TF2 imports
import tf2_ros
from tf2_ros import Buffer, TransformListener

# Message imports (Ensure these packages exist in your ROS 2 workspace)
from costmap_converter.msg import ObstacleArrayMsg, ObstacleMsg # Assuming ROS 2 version exists
from geometry_msgs.msg import Point32, Pose, PoseStamped, PoseWithCovariance, Twist
from leg_tracker.msg import PeopleVelocity, PersonVelocity # Assuming ROS 2 version exists
from nav_msgs.msg import Odometry, Path
# from people_msgs.msg import People, Person # Assuming ROS 2 version exists (Not used directly in original snippet)
from sensor_msgs.msg import PointCloud # Note: PointCloud2 is more common
from visualization_msgs.msg import Marker, MarkerArray

# Transformation utility (Install: pip install tf-transformations)
from tf_transformations import euler_from_quaternion

# Your custom MPC imports (Ensure these are Python 3 compatible)
from mpc.agent import EgoAgent
from mpc.dynamic_obstacle import DynamicObstacle
from mpc.environment import ROSEnvironment
from mpc.geometry import Circle, Polygon
from mpc.obstacle import StaticObstacle


class ROSInterfaceNode(Node):
    """
    ROSInterfaceNode class to interface with ROS 2 Humble
    Creates a node and subscribes to people messages for obstacles, and publishes commands on the /cmd_vel topic
    Also subscribes to waypoint pose messages for the next goal
    """

    def __init__(self):
        super().__init__("ros_mpc_interface_node")

        self.environment = ROSEnvironment(
            agent=EgoAgent(
                id=1,
                radius=0.5,
                initial_position=(0, 0),
                initial_orientation=np.deg2rad(90),
                horizon=10,
                use_warm_start=True,
                planning_time_step=0.8,
                linear_velocity_bounds=(0, 0.3),
                angular_velocity_bounds=(-0.5, 0.5),
                linear_acceleration_bounds=(-0.5, 0.5),
                angular_acceleration_bounds=(-1, 1),
                sensor_radius=3,
            ),
            static_obstacles=[],
            dynamic_obstacles=[],
            waypoints=[],
            plot=True, # Note: Plotting might behave differently in ROS 2 context
        )
        self.counter = 0 # Counter for obstacle processing logic

        # --- TF2 Setup ---
        self.tfbuffer = Buffer()
        # Pass the node instance (self) to the TransformListener
        self.listener = TransformListener(self.tfbuffer, self)

        # --- QoS Profiles ---
        # Default reliable profile for commands, paths, markers
        qos_profile_reliable = QoSProfile(
             reliability=QoSReliabilityPolicy.RELIABLE,
             history=QoSHistoryPolicy.KEEP_LAST,
             depth=10
         )
        # Sensor data profile (best effort, keep last) for high-frequency data like odom, obstacles
        qos_profile_sensor_data = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5
        )

        # --- Subscribers ---
        self.people_sub = self.create_subscription(
            PeopleVelocity,
            "/vel_pub",
            self.people_callback,
            qos_profile_reliable # Or sensor_data if high frequency
        )
        self.path_sub = self.create_subscription(
            Path,
            "/locomotor/VoronoiPlannerROS/voronoi_path",
            self.waypoint_callback,
            qos_profile_reliable
        )
        # Assuming '/point_cloud' publishes sensor_msgs/PointCloud
        self.obstacle_sub = self.create_subscription(
            PointCloud, # WARNING: Check if your topic actually publishes PointCloud or PointCloud2
            "/point_cloud",
            self.obstacle_callback,
            qos_profile_sensor_data
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            "/odom",
            self.odom_callback,
            qos_profile_sensor_data # Odometry often uses sensor data QoS
        )
        # self.goal_sub = self.create_subscription( # Example if needed
        #     PoseStamped,
        #     "/move_base_simple/goal",
        #     self.goal_update_callback,
        #     qos_profile_reliable
        # )

        # --- Publishers ---
        self.velocity_publisher = self.create_publisher(
            Twist,
            "wheelchair_diff/cmd_vel", # Consider remapping if needed
            qos_profile_reliable
        )
        self.marker_publisher = self.create_publisher(
            MarkerArray,
            "/future_states",
            qos_profile_reliable # Markers often use reliable QoS
        )

        # --- Internal State ---
        self.static_obstacle_list = []
        self.waypoints = []

        self.get_logger().info(f"{self.get_name()} node initialized successfully.")


    # --- Callbacks ---

    def odom_callback(self, message: Odometry):
        try:
            # Use rclpy.time.Time() for latest available transform
            trans = self.tfbuffer.lookup_transform("map", "base_link", rclpy.time.Time())

            # Extract orientation using geometry_msgs/Quaternion fields
            orientation_q = trans.transform.rotation
            orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
            current_yaw = euler_from_quaternion(orientation_list)[2]

            self.environment.agent.initial_state = np.array(
                [
                    trans.transform.translation.x,
                    trans.transform.translation.y,
                    current_yaw,
                ]
            )
            self.environment.agent.reset(matrices_only=True) # Reset MPC matrices with new state

        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
             self.get_logger().warn(f"TF lookup failed: {e}")
             pass # Continue without updating state if transform fails

    # WARNING: This callback assumes the input topic is sensor_msgs/PointCloud.
    # If it's sensor_msgs/PointCloud2, this needs rewriting using sensor_msgs_py.point_cloud2
    def obstacle_callback(self, message: PointCloud):
        # Logic to process obstacles only once (if counter logic is desired)
        if self.counter > 0:
             # pass # Uncomment if you only want to process the first message
             self.counter = 0 # Reset counter to always process latest cloud

        if self.counter == 0:
            point_cloud_list = [] # Using list append is slightly more Pythonic
            points = message.points if message.points else []
            for point in points:
                # Assuming PointCloud points have x, y, z fields
                point_cloud_list.append([point.x, point.y]) # Keep only x, y for processing

            if not point_cloud_list:
                self.static_obstacle_list = [] # Clear obstacles if input is empty
                self.get_logger().debug("Received empty static obstacle PointCloud.")
                return # No further processing needed

            # Process the collected points
            processed_points_2d = self._process_point_cloud(np.array(point_cloud_list))

            new_static_obstacle_list = []
            for i, point in enumerate(processed_points_2d.T): # Iterate rows if points are (2, N)
                if point.shape[0] == 2: # Ensure we have x, y
                    new_static_obstacle_list.append(
                        StaticObstacle(
                            id=i, # Simple sequential ID
                            geometry=Circle(
                                center=(point[0], point[1]),
                                radius=0.1, # Fixed radius for point obstacles
                            ),
                        )
                    )
            self.static_obstacle_list = new_static_obstacle_list
            # self.get_logger().debug(f"Processed {len(self.static_obstacle_list)} static obstacles.")

            self.counter += 1
        # else: # Logic if counter > 0
        #     pass

    def people_callback(self, message: PeopleVelocity):
        dynamic_obstacle_list: List[DynamicObstacle] = []

        for person in message.people: # Assuming message.people is the list
            person = cast(PersonVelocity, person) # Cast if needed for type hints
            try:
                linear_velocity = (person.velocity_x**2 + person.velocity_y**2)**0.5
                # Avoid division by zero if velocity is zero
                orientation_rad = 0.0
                if linear_velocity > 1e-6: # Small threshold
                    orientation_rad = np.arctan2(person.velocity_y, person.velocity_x)

                dynamic_obstacle_list.append(
                    DynamicObstacle(
                        id=person.id, # Assuming person has an ID field
                        position=(person.pose.position.x, person.pose.position.y),
                        orientation=orientation_rad, # Store orientation in radians
                        linear_velocity=linear_velocity,
                        angular_velocity=0, # Assuming zero angular velocity for people
                        horizon=10, # Or get from environment/agent params
                    )
                )
            except AttributeError as e:
                self.get_logger().warn(f"Missing attribute in PersonVelocity message: {e}")
                continue # Skip this person if message format is unexpected

        self.environment.dynamic_obstacles = dynamic_obstacle_list
        self.get_logger().debug(f"Updated with {len(dynamic_obstacle_list)} dynamic obstacles.")
        # Optional: Print obstacle states for debugging
        # self.get_logger().debug("--- Dynamic Obstacles ---")
        # for obstacle in dynamic_obstacle_list:
        #     self.get_logger().debug(f"ID {obstacle.id}: {obstacle.state}")
        # self.get_logger().debug("-------------------------")


    def waypoint_callback(self, message: Path):
        if not message.poses:
            self.get_logger().warn("Received empty path message.")
            # Decide behavior: clear waypoints or keep old ones?
            # self.waypoints = []
            # self.environment.waypoints = np.array([])
            return

        # Extract final pose from the new path message
        final_pose_msg = message.poses[-1].pose
        final_orientation_q = final_pose_msg.orientation
        final_orientation_list = [final_orientation_q.x, final_orientation_q.y, final_orientation_q.z, final_orientation_q.w]
        final_waypoint = (
             final_pose_msg.position.x,
             final_pose_msg.position.y,
             euler_from_quaternion(final_orientation_list)[2],
        )

        # Check if the new path's final waypoint is significantly different from the current last one
        should_update = False
        if not self.waypoints: # If no waypoints exist yet
             should_update = True
        else:
             try:
                 current_last_waypoint = self.waypoints[-1]
                 diff = np.array(current_last_waypoint) - np.array(final_waypoint)
                 diff_magnitude = np.linalg.norm(diff[:2]) # Check position difference primarily
                 self.get_logger().debug(f"Waypoint difference check: diff={diff}, mag={diff_magnitude}")
                 if diff_magnitude > 0.1: # Threshold for position difference
                     should_update = True
             except IndexError: # If self.waypoints somehow became empty between checks
                 should_update = True
             except Exception as e:
                 self.get_logger().error(f"Error comparing waypoints: {e}")
                 should_update = True # Update on error as a safe default

        if should_update:
            self.get_logger().info("Updating waypoints from new path.")
            new_waypoints = []
            # Sample waypoints from the path (e.g., every 30th point)
            sampling_step = 30
            sampled_poses = message.poses[::sampling_step]
            # Ensure the first and last poses are included if sampling
            if not message.poses[0] in sampled_poses and message.poses:
                sampled_poses.insert(0, message.poses[0])
            if not message.poses[-1] in sampled_poses and message.poses:
                 sampled_poses.append(message.poses[-1]) # Ensure last pose is always included


            for pose_stamped in sampled_poses:
                pose = pose_stamped.pose
                orientation_q = pose.orientation
                orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
                new_waypoints.append(
                    (
                        pose.position.x,
                        pose.position.y,
                        euler_from_quaternion(orientation_list)[2],
                    )
                )

            # # Ensure the very last waypoint from the message is included, even if not sampled
            # if new_waypoints[-1] != final_waypoint:
            #      new_waypoints.append(final_waypoint)


            self.waypoints = new_waypoints
            self.get_logger().info(f"Waypoints updated: {self.waypoints}")
            self.environment.waypoints = np.array(self.waypoints)
            self.environment.waypoint_index = 0 # Reset waypoint index
            if self.environment.waypoints.size > 0:
                 self.environment.agent.update_goal(self.environment.current_waypoint)
            else:
                 self.get_logger().warn("Waypoint list is empty after processing path.")
            # self.environment.plotter.update_goal(self.environment.waypoints) # Update plotter if used

    # --- Helper Methods ---

    def _process_point_cloud(self, point_cloud_xy: np.ndarray):
        """
        Processes a 2D numpy array of points (N, 2) using Open3D voxel downsampling.
        Returns a numpy array (2, M) where M <= max_points.
        """
        max_points = 500
        output_point_cloud_2d = np.zeros((2, max_points)) # Default empty result

        if point_cloud_xy.shape[0] == 0: # Check if input is empty
            self.get_logger().debug("Point cloud for processing is empty.")
            return output_point_cloud_2d

        try:
             # Create a 3D PointCloud object for Open3D, setting Z=0
             pcd = o3d.geometry.PointCloud()
             # Ensure input is (N, 3) for Vector3dVector
             points_3d = np.hstack((point_cloud_xy, np.zeros((point_cloud_xy.shape[0], 1))))
             pcd.points = o3d.utility.Vector3dVector(points_3d)

             # Downsample
             downsampled_pcd = pcd.voxel_down_sample(voxel_size=0.16)
             pcd_array_3d = np.asarray(downsampled_pcd.points)

             if pcd_array_3d.shape[0] == 0:
                 self.get_logger().debug("Point cloud empty after downsampling.")
                 return output_point_cloud_2d

             # Keep only X, Y coordinates
             pcd_array_2d = pcd_array_3d[:, :2]

             # Sort by distance and take top 'max_points'
             distances = np.linalg.norm(pcd_array_2d, axis=1)
             sorted_indices = np.argsort(distances)
             num_points_to_keep = min(pcd_array_2d.shape[0], max_points)
             closest_points_2d = pcd_array_2d[sorted_indices[:num_points_to_keep]]

             # Format as (2, M)
             output_point_cloud_2d[:, :num_points_to_keep] = closest_points_2d.T
             return output_point_cloud_2d

        except Exception as e:
             self.get_logger().error(f"Error processing point cloud with Open3D: {e}")
             # Return default empty array on error
             return np.zeros((2, max_points))


    def future_states_pub(self):
        """ Publishes predicted future states as markers """
        marker_array = MarkerArray()
        # Ensure states_matrix is populated
        if self.environment.agent.states_matrix is None or self.environment.agent.states_matrix.size == 0:
            return

        future_states = self.environment.agent.states_matrix
        marker_id = 0
        # Iterate columns if states_matrix is (state_dim, horizon_steps)
        for i in range(future_states.shape[1]):
            state = future_states[:, i]
            if len(state) < 2: continue # Need at least x, y

            marker = Marker()
            marker.header.frame_id = "map" # Ensure this frame exists
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.id = marker_id
            marker.pose.position.x = float(state[0])
            marker.pose.position.y = float(state[1])
            marker.pose.position.z = 0.0 # Assuming 2D visualization
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.05
            marker.scale.y = 0.05
            marker.scale.z = 0.05
            marker.color.a = 1.0
            marker.color.r = 0.0 # Color: Cyan
            marker.color.g = 1.0
            marker.color.b = 1.0
             # Optional: Add lifetime to prevent markers from lingering indefinitely
            # marker.lifetime = rclpy.duration.Duration(seconds=1.0).to_msg()

            marker_array.markers.append(marker)
            marker_id += 1

        if marker_array.markers:
             self.marker_publisher.publish(marker_array)


    # --- Main Execution Loop ---

    def run(self):
        """ Main execution loop for the MPC node """
        # Create a rate object based on the node's clock
        rate = self.create_rate(100) # 100 Hz

        while rclpy.ok():
            self.get_logger().debug("================= MPC Step =================")

            # Update environment obstacles (already handled by callbacks)
            # Ensure static obstacles are correctly assigned before stepping
            self.environment.static_obstacles = self.static_obstacle_list
            # Dynamic obstacles updated in people_callback

            # Run MPC step
            try:
                 self.environment.step()
            except Exception as e:
                 self.get_logger().error(f"Error during environment step: {e}", exc_info=True)
                 # Decide how to handle step error (e.g., publish zero velocity)
                 control_command = Twist() # Zero velocity command
                 self.velocity_publisher.publish(control_command)
                 rate.sleep()
                 continue # Skip rest of the loop iteration


            # Publish predicted states
            self.future_states_pub()

            # Publish control command
            control_command = Twist()
            # Ensure agent velocities are valid numbers
            linear_vel = self.environment.agent.linear_velocity
            angular_vel = self.environment.agent.angular_velocity

            if np.isnan(linear_vel) or np.isinf(linear_vel):
                self.get_logger().warn("NaN or Inf detected for linear velocity, sending 0.")
                linear_vel = 0.0
            if np.isnan(angular_vel) or np.isinf(angular_vel):
                self.get_logger().warn("NaN or Inf detected for angular velocity, sending 0.")
                angular_vel = 0.0

            control_command.linear.x = float(linear_vel)
            control_command.angular.z = float(angular_vel)

            self.get_logger().debug(f"Publishing cmd_vel: Linear={control_command.linear.x:.3f}, Angular={control_command.angular.z:.3f}")
            self.velocity_publisher.publish(control_command)

            # Wait according to rate
            rate.sleep()


def main(args=None):
    rclpy.init(args=args)
    node = None # Initialize node to None for graceful shutdown in case of init error
    try:
        node = ROSInterfaceNode()
        node.run() # Start the main processing loop
    except KeyboardInterrupt:
        if node:
            node.get_logger().info('Keyboard interrupt, shutting down.')
    except Exception as e:
        if node:
            node.get_logger().fatal(f"Unhandled exception in main: {e}", exc_info=True)
        else:
            print(f"Exception during node initialization: {e}") # Use print if logger isn't available
    finally:
        # Cleanup
        if node:
            # Optional: Publish zero velocity on shutdown
            shutdown_cmd = Twist()
            node.velocity_publisher.publish(shutdown_cmd)
            node.get_logger().info('Publishing zero velocity before shutdown.')
            # Destroy the node explicitly
            node.destroy_node()
        # Shutdown RCLPY
        if rclpy.ok():
             rclpy.shutdown()
        print("ROS 2 Interface shutdown complete.")


if __name__ == "__main__":
    main()



#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import rclpy.time

# TF2 imports
import tf2_ros
from tf2_ros import Buffer, TransformListener

# Message imports
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path
from visualization_msgs.msg import Marker, MarkerArray

# Transformation utility (Install: pip install tf-transformations)
from tf_transformations import euler_from_quaternion

# Your custom MPC imports (Ensure these are Python 3 compatible and handle empty obstacle lists)
from mpc.agent import EgoAgent
from mpc.environment import ROSEnvironment
# Assuming geometry/obstacle classes are not directly needed in this simplified interface
# from mpc.geometry import Circle, Polygon
# from mpc.obstacle import StaticObstacle
# from mpc.dynamic_obstacle import DynamicObstacle


class SimpleMPCNode(Node):
    """
    Simplified ROS 2 Interface for MPC.
    - Subscribes to Odometry (for TF) and Path (for waypoints).
    - Runs MPC based on agent state and waypoints only (no obstacles).
    - Publishes Twist commands and visualization markers.
    """

    def __init__(self):
        super().__init__("simple_mpc_node")

        # --- Initialize MPC Environment (without obstacles) ---
        # NOTE: Assumes ROSEnvironment and EgoAgent handle empty obstacle lists correctly.
        self.environment = ROSEnvironment(
            agent=EgoAgent(
                id=1,
                radius=0.5,
                initial_position=(0, 0), # Will be updated by odom_callback
                initial_orientation=np.deg2rad(90), # Will be updated by odom_callback
                horizon=10,
                use_warm_start=True,
                planning_time_step=0.8,
                linear_velocity_bounds=(0, 0.3),
                angular_velocity_bounds=(-0.5, 0.5),
                linear_acceleration_bounds=(-0.5, 0.5),
                angular_acceleration_bounds=(-1, 1),
                sensor_radius=3, # Sensor radius might be irrelevant without obstacles
            ),
            static_obstacles=[],    # NO static obstacles
            dynamic_obstacles=[],   # NO dynamic obstacles
            waypoints=[],           # Will be updated by waypoint_callback
            plot=True,              # Plotting might show only agent and path
        )

        # --- TF2 Setup ---
        self.tfbuffer = Buffer()
        self.listener = TransformListener(self.tfbuffer, self)

        # --- QoS Profiles ---
        qos_profile_reliable = QoSProfile(
             reliability=QoSReliabilityPolicy.RELIABLE,
             history=QoSHistoryPolicy.KEEP_LAST,
             depth=10
         )
        qos_profile_sensor_data = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5
        )

        # --- Subscribers ---
        # Subscribes to Odometry mainly to trigger TF lookup for the precise pose
        self.odom_sub = self.create_subscription(
            Odometry,
            "/odom",
            self.odom_callback,
            qos_profile_sensor_data
        )
        # Subscribes to the path topic for waypoints
        self.path_sub = self.create_subscription(
            Path,
            "/locomotor/VoronoiPlannerROS/voronoi_path", # Make sure this topic exists
            self.waypoint_callback,
            qos_profile_reliable
        )

        # --- Publishers ---
        # Publishes velocity commands computed by MPC
        self.velocity_publisher = self.create_publisher(
            Twist,
            "wheelchair_diff/cmd_vel", # Make sure this topic is correct for your robot
            qos_profile_reliable
        )
        # Publishes the predicted future states for visualization (e.g., in RViz)
        self.marker_publisher = self.create_publisher(
            MarkerArray,
            "/future_states",
            qos_profile_reliable
        )

        # --- Internal State ---
        self.waypoints = [] # Store the current list of waypoints

        self.get_logger().info(f"{self.get_name()} initialized successfully (Obstacle Avoidance Disabled).")
        self.get_logger().info(f"Subscribing to Odometry on: /odom")
        self.get_logger().info(f"Subscribing to Path on: /locomotor/VoronoiPlannerROS/voronoi_path")
        self.get_logger().info(f"Publishing Twist commands on: /wheelchair_diff/cmd_vel")
        self.get_logger().info(f"Publishing Markers on: /future_states")


    # --- Callbacks ---

    def odom_callback(self, message: Odometry):
        """
        Callback triggered by odometry messages. Uses TF2 to get the
        robot's current pose in the 'map' frame and updates the MPC agent's state.
        """
        try:
            # Get the latest transform from the map frame to the robot's base_link frame
            trans = self.tfbuffer.lookup_transform("map", "base_link", rclpy.time.Time())

            # Extract position
            pos_x = trans.transform.translation.x
            pos_y = trans.transform.translation.y

            # Extract orientation (quaternion) and convert to yaw (radians)
            orientation_q = trans.transform.rotation
            orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
            current_yaw = euler_from_quaternion(orientation_list)[2]

            # Update the MPC agent's initial state for the next planning cycle
            self.environment.agent.initial_state = np.array(
                [
                    pos_x,
                    pos_y,
                    current_yaw,
                ]
            )
            # Reset MPC matrices to use the new initial state
            self.environment.agent.reset(matrices_only=True)
            self.get_logger().debug(f"Agent state updated: x={pos_x:.2f}, y={pos_y:.2f}, yaw={np.rad2deg(current_yaw):.1f} deg")

        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
             # Log a warning if the transform lookup fails, but don't crash
             self.get_logger().warn(f"TF lookup failed ('map' to 'base_link'): {e}", throttle_duration_sec=5.0)
             pass # Continue, the agent will use its previously set state

    def waypoint_callback(self, message: Path):
        """
        Callback triggered by new Path messages. Extracts waypoints and updates
        the MPC environment's target path if it has changed significantly.
        """
        if not message.poses:
            self.get_logger().warn("Received empty path message.", once=True)
            return

        # Extract the final waypoint from the received path message
        final_pose_msg = message.poses[-1].pose
        final_orientation_q = final_pose_msg.orientation
        final_orientation_list = [final_orientation_q.x, final_orientation_q.y, final_orientation_q.z, final_orientation_q.w]
        final_waypoint = (
             final_pose_msg.position.x,
             final_pose_msg.position.y,
             euler_from_quaternion(final_orientation_list)[2], # Yaw
        )

        # Check if the new path's final waypoint is different from the current one
        should_update = False
        if not self.waypoints: # If no waypoints exist yet
             should_update = True
        else:
             try:
                 current_last_waypoint = self.waypoints[-1]
                 # Calculate difference primarily based on position
                 diff_pos = np.linalg.norm(np.array(current_last_waypoint[:2]) - np.array(final_waypoint[:2]))
                 if diff_pos > 0.1: # Update if final position differs by > 10cm
                     should_update = True
             except IndexError:
                 should_update = True
             except Exception as e:
                 self.get_logger().error(f"Error comparing waypoints: {e}")
                 should_update = True # Update on error

        if should_update:
            self.get_logger().info("Received new path. Updating waypoints.")
            new_waypoints = []
            # Extract waypoints (e.g., sample every Nth pose or use all)
            # Using all poses here for simplicity:
            for pose_stamped in message.poses:
                pose = pose_stamped.pose
                orientation_q = pose.orientation
                orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
                new_waypoints.append(
                    (
                        pose.position.x,
                        pose.position.y,
                        euler_from_quaternion(orientation_list)[2], # Yaw
                    )
                )

            self.waypoints = new_waypoints
            self.get_logger().info(f"Waypoints updated with {len(self.waypoints)} points.")
            self.environment.waypoints = np.array(self.waypoints)
            self.environment.waypoint_index = 0 # Start tracking from the first waypoint
            if self.environment.waypoints.size > 0:
                 # Update the agent's goal to the first waypoint in the new list
                 self.environment.agent.update_goal(self.environment.current_waypoint)
            else:
                 self.get_logger().warn("Waypoint list is empty after processing path.")

    # --- Helper Methods ---

    def future_states_pub(self):
        """ Publishes the predicted future states from the MPC agent as markers. """
        marker_array = MarkerArray()
        # Ensure the agent has computed states
        if self.environment.agent.states_matrix is None or self.environment.agent.states_matrix.size == 0:
            return

        future_states = self.environment.agent.states_matrix # Shape (state_dim, horizon_steps)
        marker_id = 0
        # Iterate through each predicted state in the horizon
        for i in range(future_states.shape[1]):
            state = future_states[:, i]
            if len(state) < 2: continue # Need at least x, y

            marker = Marker()
            marker.header.frame_id = "map" # Publish markers in the map frame
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "mpc_prediction" # Namespace for the markers
            marker.id = marker_id
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            # Position the marker at the predicted state
            marker.pose.position.x = float(state[0])
            marker.pose.position.y = float(state[1])
            marker.pose.position.z = 0.1 # Slightly elevate markers for visibility
            # Default orientation (marker orientation doesn't represent robot orientation here)
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            # Appearance
            marker.scale.x = 0.08
            marker.scale.y = 0.08
            marker.scale.z = 0.08
            marker.color.a = 0.8 # Semi-transparent
            marker.color.r = 0.0 # Color: Green
            marker.color.g = 1.0
            marker.color.b = 0.0
            # Automatically delete markers after a short duration
            marker.lifetime = rclpy.duration.Duration(seconds=0.5).to_msg()

            marker_array.markers.append(marker)
            marker_id += 1

        if marker_array.markers:
             self.marker_publisher.publish(marker_array)


    # --- Main Execution Loop ---

    def run(self):
        """ Main execution loop: runs MPC step, publishes commands and markers. """
        # Set the loop rate (e.g., 10 Hz - adjust based on MPC computation time)
        loop_rate = 10.0
        rate = self.create_rate(loop_rate)
        self.get_logger().info(f"MPC loop running at {loop_rate} Hz.")

        while rclpy.ok():
            # Check if we have waypoints to follow
            if not self.waypoints or self.environment.waypoints.size == 0:
                self.get_logger().info("No waypoints available. Stopping robot.", throttle_duration_sec=5.0)
                # Publish zero velocity if there's no path
                stop_command = Twist()
                self.velocity_publisher.publish(stop_command)
                rate.sleep()
                continue # Skip MPC step if no goal

            self.get_logger().debug("--- MPC Step Start ---")

            # Run the MPC step (calculates optimal control input)
            # Assumes self.environment.agent.initial_state and self.environment.waypoints are up-to-date
            try:
                 self.environment.step() # This calls the agent's MPC solve method internally
            except Exception as e:
                 # Log errors during the MPC calculation, publish zero velocity as a safety measure
                 self.get_logger().error(f"Error during MPC environment step: {e}", exc_info=True)
                 error_command = Twist()
                 self.velocity_publisher.publish(error_command)
                 rate.sleep()
                 continue # Skip the rest of this loop iteration

            # Publish the predicted trajectory for visualization
            self.future_states_pub()

            # Extract the calculated control command (first element of the optimal control sequence)
            control_command = Twist()
            linear_vel = self.environment.agent.linear_velocity
            angular_vel = self.environment.agent.angular_velocity

            # Basic check for invalid numbers (NaN or infinity)
            if np.isnan(linear_vel) or np.isinf(linear_vel) or \
               np.isnan(angular_vel) or np.isinf(angular_vel):
                self.get_logger().warn("NaN or Inf detected in control command, sending zero velocity.")
                linear_vel = 0.0
                angular_vel = 0.0

            control_command.linear.x = float(linear_vel)
            control_command.angular.z = float(angular_vel)

            # Publish the command
            self.get_logger().debug(f"Publishing cmd_vel: Linear={control_command.linear.x:.3f}, Angular={control_command.angular.z:.3f}")
            self.velocity_publisher.publish(control_command)

            self.get_logger().debug("--- MPC Step End ---")

            # Wait for the next cycle
            rate.sleep()


def main(args=None):
    """ Main function to initialize and run the ROS 2 node. """
    rclpy.init(args=args)
    node = None
    try:
        node = SimpleMPCNode()
        node.run() # Start the main processing loop
    except KeyboardInterrupt:
        if node:
            node.get_logger().info('Keyboard interrupt, shutting down.')
    except Exception as e:
        # Log any exceptions that weren't caught elsewhere
        if node:
            node.get_logger().fatal(f"Unhandled exception in main: {e}", exc_info=True)
        else:
            print(f"Exception during node initialization: {e}")
    finally:
        # Cleanup resources
        if node:
            # Publish a zero velocity command on shutdown
            shutdown_cmd = Twist()
            try: # Prevent error if publisher already destroyed
                node.velocity_publisher.publish(shutdown_cmd)
                node.get_logger().info('Published zero velocity before shutdown.')
            except Exception as pub_e:
                 if "Publisher already destroyed" not in str(pub_e):
                      print(f"Error publishing stop command on shutdown: {pub_e}")
            node.destroy_node()
        # Shutdown rclpy context
        if rclpy.ok():
             rclpy.shutdown()
        print("Simple MPC Node shutdown complete.")


if __name__ == "__main__":
    main()