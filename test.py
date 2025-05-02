#!/usr/bin/env python3
# === 1. Imports ===
# Standard Python libraries
import numpy as np

# ROS 2 client library for Python
import rclpy
from rclpy.node import Node # Base class for creating ROS 2 nodes
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy # Quality of Service settings
import rclpy.time # For accessing ROS time, e.g., for TF lookups

# TF2 (Transform library) imports
import tf2_ros # ROS 2 integration for TF2
from tf2_ros import Buffer, TransformListener # Tools for receiving and buffering coordinate transforms

# ROS 2 Standard Message Types
from geometry_msgs.msg import Twist # Message type for velocity commands (linear/angular)
from nav_msgs.msg import Odometry, Path # Odometry (pose/velocity estimate), Path (sequence of poses)
from visualization_msgs.msg import Marker, MarkerArray # Messages for publishing visualization shapes in RViz

# Transformation utility (requires: pip install tf-transformations)
from tf_transformations import euler_from_quaternion # Function to convert quaternions to Euler angles (like yaw)

# Your custom MPC library imports
# Ensure these modules are accessible in your Python environment (installed via setup.py)
# NOTE: These classes are assumed to handle empty obstacle lists gracefully.
from mpc.agent import EgoAgent # Your MPC agent implementation
from mpc.environment import ROSEnvironment # Your environment class managing the agent and simulation/ROS interaction

# === 2. Node Class Definition ===
class ROS2MPCInterface(Node):
    """
    ROS 2 Interface for MPC Path Following (Obstacle Avoidance Disabled).
    - Gets robot state via TF2 (triggered by Odometry).
    - Gets target path (waypoints) from a Path topic.
    - Computes velocity commands using the MPC agent.
    - Publishes Twist commands and visualization markers.
    """

    # === 3. Initialization (`__init__`) ===
    def __init__(self):
        """
        Constructor for the ROS2MPCInterface node.
        Initializes the node, MPC components, TF listener, publishers, and subscribers.
        """
        # 3.1 Initialize the parent Node class with a unique node name
        super().__init__("ros2_mpc_interface")

        # 3.2 Initialize MPC Environment and Agent
        # Creates an instance of your custom MPC environment.
        # Crucially, static_obstacles and dynamic_obstacles are empty lists,
        # meaning this setup will NOT perform obstacle avoidance.
        self.environment = ROSEnvironment(
            agent=EgoAgent( # Initialize the robot agent for the MPC
                id=1, # Agent identifier
                radius=0.5, # Agent's physical radius (meters) - used for collision checks if obstacles were present
                initial_position=(0, 0), # Placeholder, will be updated by odom_callback
                initial_orientation=np.deg2rad(90), # Placeholder, will be updated by odom_callback
                horizon=10, # MPC prediction horizon (number of steps)
                use_warm_start=True, # Optimization flag: use previous solution as initial guess
                planning_time_step=0.8, # Time step duration used in MPC planning (seconds)
                linear_velocity_bounds=(0, 0.3), # Min/Max linear velocity (m/s)
                angular_velocity_bounds=(-0.5, 0.5), # Min/Max angular velocity (rad/s)
                linear_acceleration_bounds=(-0.5, 0.5), # Min/Max linear acceleration (m/s^2)
                angular_acceleration_bounds=(-1, 1), # Min/Max angular acceleration (rad/s^2)
                sensor_radius=3, # Agent's sensor range (meters) - likely unused without obstacles
            ),
            static_obstacles=[],    # IMPORTANT: No static obstacles provided
            dynamic_obstacles=[],   # IMPORTANT: No dynamic obstacles provided
            waypoints=[],           # Waypoints list, initially empty, filled by waypoint_callback
            plot=False,             # Disable internal plotting from the environment class if it exists
        )

        # 3.3 TF2 Setup
        # Buffer stores incoming transforms for a specified duration.
        self.tfbuffer = Buffer()
        # Listener receives transforms over the network and fills the buffer.
        # It needs a reference to the node (self) to interact with ROS 2 infrastructure.
        self.listener = TransformListener(self.tfbuffer, self)

        # 3.4 Quality of Service (QoS) Profiles
        # Define communication reliability and history settings.
        # RELIABLE: Ensures delivery (retries if needed), good for commands, paths.
        qos_profile_reliable = QoSProfile(
             reliability=QoSReliabilityPolicy.RELIABLE,
             history=QoSHistoryPolicy.KEEP_LAST, # Keep only the last N messages
             depth=10 # Buffer size for KEEP_LAST history
         )
        # BEST_EFFORT: Faster, less overhead, okay to drop messages. Good for high-frequency sensor data.
        qos_profile_sensor_data = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5 # Keep fewer messages for sensor data
        )

        # 3.5 Subscribers
        # Create subscriptions to specific topics. When a message arrives, the specified callback is executed.
        # Subscribes to Odometry: Used mainly as a trigger to query TF for the latest pose.
        self.odom_sub = self.create_subscription(
            Odometry,                   # Message type
            "/odom",                    # Topic name (standard odometry topic)
            self.odom_callback,         # Function to call when a message arrives
            qos_profile_sensor_data     # QoS profile for this subscription
        )
        # Subscribes to Path: Receives the desired path for the robot to follow.
        self.path_sub = self.create_subscription(
            Path,                                       # Message type
            "/locomotor/VoronoiPlannerROS/voronoi_path", # Topic name (source of the path) - CHANGE IF NEEDED
            self.waypoint_callback,                     # Callback function
            qos_profile_reliable                        # QoS profile
        )

        # 3.6 Publishers
        # Create publishers to send messages to specific topics.
        # Publishes Twist commands: Sends velocity commands calculated by the MPC.
        self.velocity_publisher = self.create_publisher(
            Twist,                      # Message type
            "wheelchair_diff/cmd_vel",  # Topic name (where the robot controller listens) - CHANGE IF NEEDED
            qos_profile_reliable        # QoS profile
        )
        # Publishes MarkerArray: Sends visualization markers for RViz.
        self.marker_publisher = self.create_publisher(
            MarkerArray,                # Message type
            "/future_states",           # Topic name for markers
            qos_profile_reliable        # QoS profile
        )

        # 3.7 Internal State Variables
        # Store the list of waypoints received from the path topic.
        self.waypoints = []

        # 3.8 Logging Initialization Info
        # Use the node's logger for ROS 2 standard logging.
        self.get_logger().info(f"'{self.get_name()}' node initialized (Obstacle Avoidance Disabled).")
        self.get_logger().info(f"Subscribing to Odometry on: {self.odom_sub.topic_name}")
        self.get_logger().info(f"Subscribing to Path on: {self.path_sub.topic_name}")
        self.get_logger().info(f"Publishing Twist commands on: {self.velocity_publisher.topic_name}")
        self.get_logger().info(f"Publishing Markers on: {self.marker_publisher.topic_name}")


    # === 4. Callbacks ===
    def odom_callback(self, message: Odometry):
        """
        Callback for Odometry messages. Uses TF2 to get the robot's pose
        in the 'map' frame and updates the MPC agent's current state.
        """
        try:
            # 4.1 Look up the transform from 'map' frame to 'base_link' frame.
            # 'map' is typically the fixed world frame from SLAM/localization.
            # 'base_link' is typically the robot's root frame. - CHANGE IF YOUR FRAMES DIFFER
            # rclpy.time.Time() requests the latest available transform.
            trans = self.tfbuffer.lookup_transform("map", "base_link", rclpy.time.Time())

            # 4.2 Extract position (x, y) from the transform.
            pos_x = trans.transform.translation.x
            pos_y = trans.transform.translation.y

            # 4.3 Extract orientation (quaternion) from the transform.
            orientation_q = trans.transform.rotation
            # Create a list [x, y, z, w] for the conversion function.
            orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
            # Convert quaternion to Euler angles; index [2] gets the yaw (rotation around Z).
            current_yaw = euler_from_quaternion(orientation_list)[2]

            # 4.4 Update the MPC agent's state (used as the starting point for the next MPC plan).
            # This should match the state representation used by your EgoAgent.
            self.environment.agent.initial_state = np.array([pos_x, pos_y, current_yaw])

            # 4.5 Reset agent's internal MPC matrices to reflect the new initial state.
            # `matrices_only=True` might be an optimization to avoid re-creating everything.
            self.environment.agent.reset(matrices_only=True)
            # Log the updated state for debugging (at DEBUG level to avoid flooding).
            self.get_logger().debug(f"Agent state updated: x={pos_x:.2f}, y={pos_y:.2f}, yaw={np.rad2deg(current_yaw):.1f} deg")

        # 4.6 Handle TF2 exceptions if the transform lookup fails.
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
             # Log a warning, throttled to avoid spamming the console if TF is temporarily unavailable.
             self.get_logger().warn(f"TF lookup failed ('map' to 'base_link'): {e}", throttle_duration_sec=5.0)
             # Allow the node to continue; the agent will use its previous state.
             pass

    def waypoint_callback(self, message: Path):
        """
        Callback for Path messages. Extracts waypoints and updates the MPC target path.
        """
        # 4.7 Check if the received path contains any poses.
        if not message.poses:
            # Log a warning once if an empty path is received.
            self.get_logger().warn("Received empty path message.", once=True)
            # Optionally clear existing waypoints or just ignore the empty message.
            # self.waypoints = []
            # self.environment.waypoints = np.array(self.waypoints)
            return # Exit the callback

        # 4.8 Extract the final pose from the received path. Used to check if the path goal changed.
        final_pose_msg = message.poses[-1].pose
        final_orientation_q = final_pose_msg.orientation
        final_orientation_list = [final_orientation_q.x, final_orientation_q.y, final_orientation_q.z, final_orientation_q.w]
        # Convert the final pose into the (x, y, yaw) tuple format used internally.
        final_waypoint = (
             final_pose_msg.position.x,
             final_pose_msg.position.y,
             euler_from_quaternion(final_orientation_list)[2], # Yaw
        )

        # 4.9 Check if the received path is significantly different from the current one.
        should_update = False
        # If we don't have any waypoints stored yet, definitely update.
        if not self.waypoints:
             should_update = True
        else:
             try:
                 # Get the last waypoint currently stored.
                 current_last_waypoint = self.waypoints[-1]
                 # Calculate the Euclidean distance between the current last waypoint's position
                 # and the new path's final waypoint's position.
                 diff_pos = np.linalg.norm(np.array(current_last_waypoint[:2]) - np.array(final_waypoint[:2]))
                 # If the position difference exceeds a threshold (e.g., 10 cm), consider it a new path.
                 if diff_pos > 0.1:
                     should_update = True
             except IndexError: # Should not happen if self.waypoints check passed, but safe to handle.
                 should_update = True
             except Exception as e: # Catch any other comparison errors.
                 self.get_logger().error(f"Error comparing waypoints: {e}")
                 should_update = True # Update path if comparison fails.

        # 4.10 If the path should be updated:
        if should_update:
            self.get_logger().info("Received new path. Updating waypoints.")
            # Create a new list to store waypoints extracted from the message.
            new_waypoints = []
            # Iterate through all poses in the received Path message.
            # (Alternatively, you could sample them, e.g., message.poses[::10])
            for pose_stamped in message.poses:
                pose = pose_stamped.pose # Get the Pose part
                orientation_q = pose.orientation # Get the Quaternion
                # Convert quaternion to [x,y,z,w] list for the helper function.
                orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
                # Append the waypoint as an (x, y, yaw) tuple.
                new_waypoints.append(
                    (
                        pose.position.x,
                        pose.position.y,
                        euler_from_quaternion(orientation_list)[2], # Yaw
                    )
                )

            # 4.11 Update the node's internal waypoint list.
            self.waypoints = new_waypoints
            self.get_logger().info(f"Waypoints updated with {len(self.waypoints)} points.")
            # Update the waypoints within the MPC environment (converting to NumPy array).
            self.environment.waypoints = np.array(self.waypoints)
            # Reset the environment's waypoint tracker to start from the beginning of the new path.
            self.environment.waypoint_index = 0
            # If the new path is not empty, update the agent's immediate goal
            # (usually the first or next waypoint based on environment logic).
            if self.environment.waypoints.size > 0:
                 self.environment.agent.update_goal(self.environment.current_waypoint)
            else:
                 # Log a warning if the processed path resulted in zero waypoints.
                 self.get_logger().warn("Waypoint list is empty after processing path.")

    # === 5. Helper Methods ===
    def future_states_pub(self):
        """ Publishes the MPC agent's predicted future trajectory as visualization markers. """
        # 5.1 Create a MarkerArray message to hold multiple markers.
        marker_array = MarkerArray()
        # 5.2 Check if the agent has computed a trajectory (states_matrix).
        if self.environment.agent.states_matrix is None or self.environment.agent.states_matrix.size == 0:
            return # Do nothing if no prediction is available

        # 5.3 Get the predicted states matrix (usually shape: [state_dim, horizon_length]).
        future_states = self.environment.agent.states_matrix
        marker_id = 0 # Unique ID for each marker in the array.
        # 5.4 Iterate through each predicted state (columns of the matrix).
        for i in range(future_states.shape[1]):
            state = future_states[:, i] # Get the i-th predicted state [x, y, yaw, ...]
            # Ensure the state has at least x and y components.
            if len(state) < 2: continue

            # 5.5 Create a new Marker message.
            marker = Marker()
            # Set the frame_id (must match the fixed frame in RViz, e.g., "map").
            marker.header.frame_id = "map"
            # Set the timestamp.
            marker.header.stamp = self.get_clock().now().to_msg()
            # Set a namespace to group markers.
            marker.ns = "mpc_prediction"
            # Assign the unique ID.
            marker.id = marker_id
            # Set the marker type (e.g., SPHERE).
            marker.type = Marker.SPHERE
            # Set the action (ADD modifies/adds the marker).
            marker.action = Marker.ADD
            # Set the marker's position using the predicted state's x and y.
            marker.pose.position.x = float(state[0])
            marker.pose.position.y = float(state[1])
            marker.pose.position.z = 0.1 # Slightly raise markers off the ground plane.
            # Set marker orientation (usually fixed, doesn't represent robot orientation).
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            # Set marker scale (size).
            marker.scale.x = 0.08
            marker.scale.y = 0.08
            marker.scale.z = 0.08
            # Set marker color (RGBA, values 0.0 to 1.0). Green, semi-transparent.
            marker.color.a = 0.8 # Alpha (transparency)
            marker.color.r = 0.0 # Red
            marker.color.g = 1.0 # Green
            marker.color.b = 0.0 # Blue
            # Set marker lifetime (how long it persists in RViz before auto-deleting).
            marker.lifetime = rclpy.duration.Duration(seconds=0.5).to_msg()

            # 5.6 Add the configured marker to the MarkerArray.
            marker_array.markers.append(marker)
            # Increment the ID for the next marker.
            marker_id += 1

        # 5.7 If any markers were created, publish the entire MarkerArray.
        if marker_array.markers:
             self.marker_publisher.publish(marker_array)


    # === 6. Main Execution Loop (`run`) ===
    def run(self):
        """ Contains the main loop that repeatedly runs the MPC step and publishes commands. """
        # 6.1 Define the desired loop frequency (e.g., 10 Hz).
        loop_rate = 10.0
        # Create a Rate object to control the loop speed.
        rate = self.create_rate(loop_rate)
        self.get_logger().info(f"MPC control loop started at {loop_rate} Hz.")

        # 6.2 Loop continuously as long as ROS 2 is running (`rclpy.ok()`).
        while rclpy.ok():
            # 6.3 Check if waypoints are available. If not, stop the robot.
            if not self.waypoints or self.environment.waypoints.size == 0:
                # Log this state, throttled to avoid spam.
                self.get_logger().info("No waypoints available. Stopping robot.", throttle_duration_sec=5.0)
                # Create a Twist message with zero velocities.
                stop_command = Twist()
                # Publish the zero velocity command.
                self.velocity_publisher.publish(stop_command)
                # Wait for the next loop iteration according to the rate.
                rate.sleep()
                # Skip the rest of the loop (MPC step).
                continue

            # Log the start of an MPC cycle (at DEBUG level).
            self.get_logger().debug("--- MPC Step Start ---")

            # 6.4 Execute the core MPC step.
            # This involves:
            # - The environment potentially updating the agent's goal based on current position and waypoint list.
            # - The agent solving the optimization problem based on its current state, goal, and prediction model.
            try:
                 # Call the environment's step method, which should trigger the agent's MPC solve.
                 self.environment.step()
            # 6.5 Handle potential errors during the MPC calculation.
            except Exception as e:
                 # Log the error with traceback information.
                 self.get_logger().error(f"Error during MPC environment step: {e}", exc_info=True)
                 # Publish zero velocity as a safety fallback.
                 error_command = Twist()
                 self.velocity_publisher.publish(error_command)
                 # Sleep and skip the rest of this loop iteration.
                 rate.sleep()
                 continue

            # 6.6 Publish the predicted trajectory markers after a successful step.
            self.future_states_pub()

            # 6.7 Get the calculated optimal control command (velocity) from the agent.
            # This is typically the first control input from the MPC solution sequence.
            control_command = Twist()
            linear_vel = self.environment.agent.linear_velocity
            angular_vel = self.environment.agent.angular_velocity

            # 6.8 Sanity check: Ensure velocities are valid numbers (not NaN or Inf).
            if np.isnan(linear_vel) or np.isinf(linear_vel) or \
               np.isnan(angular_vel) or np.isinf(angular_vel):
                self.get_logger().warn("NaN or Inf detected in control command, sending zero velocity.")
                linear_vel = 0.0
                angular_vel = 0.0

            # 6.9 Populate the Twist message.
            control_command.linear.x = float(linear_vel)
            control_command.angular.z = float(angular_vel) # Assuming a diff-drive robot where yaw rate is controlled.

            # 6.10 Publish the Twist command.
            self.get_logger().debug(f"Publishing cmd_vel: Linear={control_command.linear.x:.3f}, Angular={control_command.angular.z:.3f}")
            self.velocity_publisher.publish(control_command)

            # Log the end of the cycle (at DEBUG level).
            self.get_logger().debug("--- MPC Step End ---")

            # 6.11 Wait until the next cycle based on the defined loop rate.
            rate.sleep()


# === 7. Main Execution (`if __name__ == "__main__":`) ===
def main(args=None):
    """
    Main function called when the script is executed.
    Initializes rclpy, creates the node, runs it, and handles shutdown.
    """
    # 7.1 Initialize the ROS 2 Python client library.
    rclpy.init(args=args)
    # Initialize node variable to None for proper error handling/cleanup.
    node = None
    try:
        # 7.2 Create an instance of the ROS2MPCInterface node class.
        node = ROS2MPCInterface()
        # 7.3 Start the node's main loop (calls the `run` method).
        node.run()
    # 7.4 Catch keyboard interrupts (Ctrl+C) for graceful shutdown.
    except KeyboardInterrupt:
        if node: # Check if node was successfully created before logging.
            node.get_logger().info('Keyboard interrupt, shutting down.')
    # 7.5 Catch any other unexpected exceptions.
    except Exception as e:
        # Log fatal errors.
        if node:
            node.get_logger().fatal(f"Unhandled exception in main: {e}", exc_info=True)
        else:
            # Use print if the logger isn't available (e.g., error during __init__).
            print(f"Exception during node initialization: {e}")
    # 7.6 Finally block: ensures cleanup code runs regardless of exceptions.
    finally:
        if node:
            # 7.7 Optional: Publish a zero velocity command just before shutting down.
            shutdown_cmd = Twist()
            try: # Protect against errors if publisher is already destroyed.
                node.velocity_publisher.publish(shutdown_cmd)
                node.get_logger().info('Published zero velocity before shutdown.')
            except Exception as pub_e:
                 # Avoid logging errors if the publisher was already cleaned up.
                 if "Publisher already destroyed" not in str(pub_e):
                      print(f"Error publishing stop command on shutdown: {pub_e}")
            # 7.8 Properly destroy the node instance, releasing resources.
            node.destroy_node()
        # 7.9 Shut down the ROS 2 Python client library.
        if rclpy.ok(): # Check if rclpy is still running before shutting down.
             rclpy.shutdown()
        print(f"'{ROS2MPCInterface.__name__}' shutdown complete.")


# 7.10 Standard Python entry point check: calls the main function when the script is run directly.
if __name__ == "__main__":
    main()