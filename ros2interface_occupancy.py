#!/usr/bin/env python3
from typing import List, cast

import cv2
import numpy as np
import rclpy
import tf2_ros
from geometry_msgs.msg import Point32, Pose, Twist
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.node import Node
from scipy.spatial.transform import (
    Rotation as R,  # Replacement for tf_transformations from ROS1
)
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

# from costmap_converter.msg import ObstacleArrayMsg, ObstacleMsg
# from leg_tracker.msg import PeopleVelocity, PersonVelocity
from circles_from_occupancy_map import get_circle_locations_from_occupancy_map
from mpc.agent import EgoAgent
from mpc.dynamic_obstacle import DynamicObstacle
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
        # self.counter = 0

        self.tfbuffer = Buffer()
        self.listener = TransformListener(self.tfbuffer, self)

        # self.create_subscription(PeopleVelocity, '/vel_pub', self.people_callback, 10)
        self.create_subscription(Path, "/plan", self.waypoint_callback, 10)
        self.subscription = self.create_subscription(
            OccupancyGrid, "/global_costmap/costmap", self.obstacle_callback, 10
        )
        # self.create_subscription(ObstacleArrayMsg, '/costmap_converter/costmap_obstacles', self.obstacle_callback, 10)
        self.create_subscription(Odometry, "/odom", self.odom_callback, 10)

        self.velocity_publisher = self.create_publisher(
            Twist, "/wheelchair2_base_controller/cmd_vel_unstamped", 10
        )
        self.marker_publisher = self.create_publisher(MarkerArray, "/future_states", 10)

        self.static_obstacle_list = []
        self.waypoints = []

        self.timer = self.create_timer(0.01, self.run)

    def run(self):
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

    def odom_callback(self, message: Odometry):
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
        circle_locations = get_circle_locations_from_occupancy_map(occupancy_map)

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

        self.static_obstacle_list = static_obstacle_list

    # def obstacle_callback(self, msg: OccupancyGrid):
    #     if self.counter == 0:
    #         width = msg.info.width
    #         height = msg.info.height
    #         resolution = msg.info.resolution
    #         origin = msg.info.origin

    #         grid = np.array(msg.data, dtype=np.int8).reshape((height, width))
    #         binary = np.uint8((grid > 50) * 255)

    #         contours, _ = cv2.findContours(
    #             binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    #         )

    #         self.static_obstacle_list = []

    #         for contour in contours:
    #             if len(contour) >= 3:
    #                 polygon = []
    #                 for pt in contour:
    #                     x = pt[0][0] * resolution + origin.position.x
    #                     y = pt[0][1] * resolution + origin.position.y
    #                     polygon.append((x, y))
    #                 self.static_obstacle_list.append(
    #                     StaticObstacle(
    #                         id=len(self.static_obstacle_list),
    #                         geometry=Polygon(vertices=polygon),
    #                     )
    #                 )
    #         self.counter += 1

    # def obstacle_callback(self, message: ObstacleArrayMsg):
    #     if self.counter == 0:
    #         self.static_obstacle_list = []
    #         for obstacle in message.obstacles:
    #             if len(obstacle.polygon.points[:-1]) > 2:
    #                 points = [
    #                     (point.x, point.y)
    #                     for point in cast(List[Point32], obstacle.polygon.points[:-1])
    #                 ]
    #             else:
    #                 continue
    #             self.static_obstacle_list.append(
    #                 StaticObstacle(
    #                     id=obstacle.id,
    #                     geometry=Polygon(vertices=points),
    #                 )
    #             )
    #         self.counter += 1

    # def people_callback(self, message: PeopleVelocity):
    #     dynamic_obstacle_list: List[DynamicObstacle] = []
    #     for person in message.people:
    #         dynamic_obstacle_list.append(
    #             DynamicObstacle(
    #                 id=person.id,
    #                 position=(person.pose.position.x, person.pose.position.y),
    #                 orientation=np.rad2deg(np.arctan2(person.velocity_y, person.velocity_x)),
    #                 linear_velocity=(person.velocity_x**2 + person.velocity_y**2)**0.5,
    #                 angular_velocity=0,
    #                 horizon=10,
    #             )
    #         )
    #     self.environment.dynamic_obstacles = dynamic_obstacle_list

    def waypoint_callback(self, message: Path):
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
        except:
            diff = 0

        if self.waypoints == [] or abs(diff) > 0.1:
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
                for pose in message.poses[::30]
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
