#!/usr/bin/env python3
import message_filters
import numpy as np
import rclpy
import tf2_ros
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.node import Node
from scipy.spatial.transform import (
    Rotation as R,  # Replacement for tf_transformations from ROS1
)
from tf2_ros import Buffer
from visualization_msgs.msg import Marker, MarkerArray

from circles_from_occupancy_map import get_circle_locations_from_occupancy_map
from mpc.agent import EgoAgent
from mpc.environment import ROSEnvironment
from mpc.geometry import Circle
from mpc.obstacle import StaticObstacle


def euler_from_quaternion(quat):
    return R.from_quat(quat).as_euler("xyz")


class ROSInterface(Node):
    def __init__(self):
        super().__init__("ros_mpc_interface")

        self.environment = ROSEnvironment(
            agent=EgoAgent(
                id=1,
                radius=0.5,
                initial_position=(0, 0),
                initial_orientation=np.deg2rad(90),
                horizon=5,
                use_warm_start=True,
                planning_time_step=0.8,
                linear_velocity_bounds=(0, 0.25),
                angular_velocity_bounds=(-0.25, 0.25),
                linear_acceleration_bounds=(-0.5, 0.5),
                angular_acceleration_bounds=(-1, 1),
                sensor_radius=3,
            ),
            static_obstacles=[],
            dynamic_obstacles=[],
            waypoints=[],
            plot=True,
        )

        self.tfbuffer = Buffer()

        # SUBSCRIBERS
        self.create_subscription(Path, "/plan", self.waypoint_callback, 10)
        occupancy_map_subscriber = message_filters.Subscriber(
            "/local_costmap/costmap", OccupancyGrid
        )
        odometry_subscriber = message_filters.Subscriber("/odom", Odometry)

        time_synchronizer = message_filters.ApproximateTimeSynchronizer(
            [occupancy_map_subscriber, odometry_subscriber], queue_size=1, slop=0.1
        )
        time_synchronizer.registerCallback(self.planning_callback)

        # PUBLISHERS
        self.velocity_publisher = self.create_publisher(
            Twist, "/wheelchair2_base_controller/cmd_vel_unstamped", 10
        )
        self.marker_publisher = self.create_publisher(MarkerArray, "/future_states", 10)

        self.waypoints = []

        self.timer = self.create_timer(0.01, self.run)

    def planning_callback(
        self, occupancy_map_msg: OccupancyGrid, odometry_msg: Odometry
    ):
        self.obstacle_callback(occupancy_map_msg)
        self.odometry_callback(odometry_msg)

    def run(self):
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

    def odometry_callback(self, message: Odometry):
        try:
            trans = self.tfbuffer.lookup_transform(
                "map", "base_link", rclpy.time.Time()
            )
            self.environment.agent.initial_state = np.array(
                [
                    trans.transform.translation.x,
                    trans.transform.translation.y,
                    euler_from_quaternion(
                        [
                            trans.transform.rotation.x,
                            trans.transform.rotation.y,
                            trans.transform.rotation.z,
                            trans.transform.rotation.w,
                        ]
                    )[2],
                ]
            )
            self.environment.agent.reset(matrices_only=True)
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ):
            pass

    def obstacle_callback(self, message: OccupancyGrid):
        occupancy_map = np.array(message.data).reshape(
            message.info.height, message.info.width
        )
        circle_locations = get_circle_locations_from_occupancy_map(
            occupancy_map,
            ego_position=tuple(self.environment.agent.initial_state[:2]),
            occupancy_map_resolution=message.info.resolution,
        )

        static_obstacle_list = []

        for i, point in enumerate(circle_locations):
            static_obstacle_list.append(
                StaticObstacle(
                    id=i,
                    geometry=Circle(
                        center=(point[0], point[1]),
                        radius=0.1,
                    ),
                )
            )
        # Update the environment with the latest static obstacles
        self.environment.static_obstacles = static_obstacle_list

    def waypoint_callback(self, message: Path):
        # Check if the last waypoint is close to the current position
        try:
            diff = np.array(self.waypoints[-1]) - np.array(
                (
                    message.poses[-1].pose.position.x,
                    message.poses[-1].pose.position.y,
                    euler_from_quaternion(
                        [
                            message.poses[-1].pose.orientation.x,
                            message.poses[-1].pose.orientation.y,
                            message.poses[-1].pose.orientation.z,
                            message.poses[-1].pose.orientation.w,
                        ]
                    )[2],
                )
            )
            diff = diff.sum()
        except Exception:
            diff = 0

        # If there are no waypoints or the last waypoint is not close to the current position, update the waypoints
        if self.waypoints == [] or abs(diff) > 0.1:
            waypoints = [
                # (
                #     pose.pose.position.x,
                #     pose.pose.position.y,
                #     euler_from_quaternion(
                #         [
                #             pose.pose.orientation.x,
                #             pose.pose.orientation.y,
                #             pose.pose.orientation.z,
                #             pose.pose.orientation.w,
                #         ]
                #     )[2],
                # )
                # for pose in message.poses[::30]
            ]
            waypoints.append(
                (
                    message.poses[-1].pose.position.x,
                    message.poses[-1].pose.position.y,
                    euler_from_quaternion(
                        [
                            message.poses[-1].pose.orientation.x,
                            message.poses[-1].pose.orientation.y,
                            message.poses[-1].pose.orientation.z,
                            message.poses[-1].pose.orientation.w,
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


if __name__ == "__main__":
    main()
