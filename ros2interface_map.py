#!/usr/bin/env python3
import message_filters
import numpy as np
import rclpy
import rclpy.duration
import rclpy.time
import tf2_ros

from tf2_geometry_msgs import do_transform_pose_stamped
from tf2_geometry_msgs import do_transform_pose
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.node import Node
from scipy.spatial.transform import (
    Rotation as R,  # Replacement for tf_transformations from ROS1
)
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from circles_from_occupancy_map import get_circle_locations_from_occupancy_map
from mpc.agent import EgoAgent
from mpc.environment import ROSEnvironment
from mpc.geometry import Circle
from mpc.obstacle import StaticObstacle
import cv2
from mpc.geometry import Polygon



def euler_from_quaternion(quat, degree = False):
    return R.from_quat(quat).as_euler('xyz')

class ROSInterface(Node):
    def __init__(self):
        super().__init__('ros_mpc_interface')

        self.environment = ROSEnvironment(
            agent=EgoAgent(
                id=1,
                radius=0.45,
                initial_position=(0, 0),
                initial_orientation=np.deg2rad(90),
                horizon=6,
                use_warm_start=True,
                planning_time_step=0.8,
                linear_velocity_bounds=(-0.1, 0.3),
                angular_velocity_bounds=(-0.30, 0.30),
                linear_acceleration_bounds=(-0.5, 0.5),
                angular_acceleration_bounds=(-0.5, 0.5),
                sensor_radius=3,
            ),
            static_obstacles=[],
            dynamic_obstacles=[],
            waypoints=[],
            plot=True,
        )
        self.counter = 0

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)


        #SUBSCRIBERS
        self.create_subscription(Path, '/plan', self.waypoint_callback, 10)

        occupancy_map_subscriber = message_filters.Subscriber(
            self, OccupancyGrid, "/local_costmap/costmap"
        )
        odometry_subscriber = message_filters.Subscriber(self, Odometry, "/odom")

        time_synchronizer = message_filters.ApproximateTimeSynchronizer(
            [occupancy_map_subscriber, odometry_subscriber], queue_size=1, slop=1
        )
        time_synchronizer.registerCallback(self.planning_callback)


        # PUBLISHERS    
        self.velocity_publisher = self.create_publisher(Twist, '/wheelchair2_base_controller/cmd_vel_unstamped', 10)
        self.marker_publisher = self.create_publisher(MarkerArray, '/future_states', 10)

        self.static_obstacle_list = []
        self.waypoints = []

        self.timer = self.create_timer(0.01, self.run)

    def planning_callback(
        self, occupancy_map_msg: OccupancyGrid, odometry_msg: Odometry
        ):
        try:
            trans = self.tf_buffer.lookup_transform("map", "base_link", rclpy.time.Time())

            self.environment.agent.initial_state = np.array([
                trans.transform.translation.x,
                trans.transform.translation.y,
                euler_from_quaternion([
                    trans.transform.rotation.x,
                    trans.transform.rotation.y,
                    trans.transform.rotation.z,
                    trans.transform.rotation.w,
                ])[2],
            ])
            self.environment.agent.reset(matrices_only=True)

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f'Failed to get transform from map to base_link: {e}')
            return

        self.obstacle_callback(occupancy_map_msg)

    def run(self):
        if not self.waypoints:
            return
        self.environment.static_obstacles = self.static_obstacle_list
        
        
        self.environment.step()
        
        self.future_states_pub()

        control_command = Twist()
        control_command.linear.x = self.environment.agent.linear_velocity
        control_command.angular.z = self.environment.agent.angular_velocity

        self.velocity_publisher.publish(control_command)

    def future_states_pub(self):
        marker_array = MarkerArray()
        future_states = self.environment.agent.states_matrix
        for i, state in enumerate(future_states.T):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.id = i
            marker.pose.position.x = float(state[0])
            marker.pose.position.y = float(state[1])
            marker.pose.position.z = 0.0
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.05
            marker.scale.y = 0.05
            marker.scale.z = 0.05
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 1.0
            marker_array.markers.append(marker)

        self.marker_publisher.publish(marker_array)


    def obstacle_callback(self, msg: OccupancyGrid):
        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution
        origin = msg.info.origin
        costmap_frame = msg.header.frame_id # e.g., 'odom'

        # We need to transform obstacle points from the costmap's frame to the 'map' frame
        try:
            # The transform is from the target frame ('map') to the source frame (costmap_frame)
            # Note: lookup_transform's order can feel counter-intuitive.
            # It gives the transform TO the target_frame FROM the source_frame.
            transform = self.tf_buffer.lookup_transform('map', costmap_frame, rclpy.time.Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f'Failed to get transform from {costmap_frame} to map: {e}')
            return

        grid = np.array(msg.data, dtype=np.int8).reshape((height, width))
        binary = np.uint8((grid > 50) * 255)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self.static_obstacle_list = []
        min_obstacle_area = 0.3
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_obstacle_area:
                continue

            epsilon = 0.03 * cv2.arcLength(contour, True)
            simplified = cv2.approxPolyDP(contour, epsilon, True)

            if len(simplified) >= 2:
                polygon_map_frame = []
                for pt in simplified:
                    # First, get point in the costmap's original frame
                    local_x = pt[0][0] * resolution + origin.position.x
                    local_y = pt[0][1] * resolution + origin.position.y

                    # Now, transform this point to the 'map' frame
                    # Create a PoseStamped message for the point.
                    pose_stamped = PoseStamped()
                    pose_stamped.header.frame_id = costmap_frame
                    pose_stamped.header.stamp = msg.header.stamp # Use the costmap's timestamp
                    pose_stamped.pose.position.x = local_x
                    pose_stamped.pose.position.y = local_y
                    pose_stamped.pose.orientation.w = 1.0 # Orientation doesn't matter for a point

                    # *** THIS IS THE CORRECTED LINE ***
                    # Use do_transform_pose_stamped, which correctly handles PoseStamped objects
                    transformed_pose_stamped = do_transform_pose_stamped(pose_stamped, transform)

                    # Append the transformed point's coordinates to the list
                    polygon_map_frame.append((transformed_pose_stamped.pose.position.x, transformed_pose_stamped.pose.position.y))

                if len(polygon_map_frame) >= 3: # Ensure we have a valid polygon
                    self.static_obstacle_list.append(
                        StaticObstacle(
                            id=len(self.static_obstacle_list),
                            geometry=Polygon(vertices=polygon_map_frame)
                        )
                    )



    def waypoint_callback(self, message: Path):
        # The global plan in 'message' is typically already in the 'map' frame.
        # We no longer need to transform it to 'odom'.

        # Check if the incoming path's frame_id is 'map'. If not, a transform would be needed.
        if message.header.frame_id != 'map':
            self.get_logger().warn(f"Received path in frame '{message.header.frame_id}', but expected 'map'. Waypoints might be incorrect.")
            # If you expect other frames, you would add transform logic here.
            # For now, we assume it's 'map'.

        if not message.poses:
            return

        poses = message.poses # No transformation needed

        # The rest of your logic can remain, as it compares the new goal to the old one.
        try:
            last_new_pose = poses[-1].pose
            diff = np.array(self.waypoints[-1]) - np.array(
                (
                    last_new_pose.position.x,
                    last_new_pose.position.y,
                    euler_from_quaternion(
                        [
                            last_new_pose.orientation.x,
                            last_new_pose.orientation.y,
                            last_new_pose.orientation.z,
                            last_new_pose.orientation.w,
                        ]
                    )[2],
                )
            )
            diff = diff.sum()
        except Exception:
            diff = 0

        if not self.waypoints or abs(diff) > 0.1:
            self.get_logger().info("Updating goal waypoints in map frame.")
            waypoints = [
                (
                    pose.pose.position.x,
                    pose.pose.position.y,
                    euler_from_quaternion(
                        [
                            pose.pose.orientation.x,
                            pose.pose.orientation.y,
                            pose.pose.orientation.z,
                            pose.pose.orientation.w,
                        ]
                    )[2],
                )
                for pose in poses[::35]
            ]
            # Ensure the final goal is included
            final_pose = poses[-1].pose
            waypoints.append(
                (
                    final_pose.position.x,
                    final_pose.position.y,
                    euler_from_quaternion(
                        [
                            final_pose.orientation.x,
                            final_pose.orientation.y,
                            final_pose.orientation.z,
                            final_pose.orientation.w,
                        ]
                    )[2],
                )
            )
            self.waypoints = waypoints
            self.environment.waypoints = np.array(waypoints)
            self.environment.waypoint_index = 0
            self.environment.agent.update_goal(self.environment.current_waypoint)

def main(args=None):
    rclpy.init(args=args)
    ros_interface = ROSInterface()
    rclpy.spin(ros_interface)
    ros_interface.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
