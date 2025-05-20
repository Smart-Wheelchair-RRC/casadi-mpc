#!/usr/bin/env python3
from typing import List, cast
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point32, Twist
from nav_msgs.msg import Odometry, Path
from scipy.spatial.transform import Rotation as R  

from mpc.agent import EgoAgent
from mpc.dynamic_obstacle import DynamicObstacle
from mpc.environment import ROSEnvironment
from mpc.geometry import Polygon
from mpc.obstacle import StaticObstacle

from tf2_ros import TransformListener, Buffer
import tf2_ros
#from costmap_converter.msg import ObstacleArrayMsg, ObstacleMsg
# from leg_tracker.msg import PeopleVelocity, PersonVelocity


def euler_from_quaternion(quat):
    return R.from_quat(quat).as_euler('xyz')

class ROSInterface(Node):
    def __init__(self):
        super().__init__('ros_mpc_interface')

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
                angular_acceleration_bounds=(-1,1),
                sensor_radius=3,
            ),
            static_obstacles=[],
            dynamic_obstacles=[],
            waypoints=[],
            plot=True,
        )
        self.counter = 0

        self.tfbuffer = Buffer()
        self.listener = TransformListener(self.tfbuffer, self)

        # self.create_subscription(PeopleVelocity, '/vel_pub', self.people_callback, 10)
        self.create_subscription(Path, '/plan', self.waypoint_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(ObstacleArrayMsg, '/costmap_converter/costmap_obstacles', self.obstacle_callback, 10)
        

        self.velocity_publisher = self.create_publisher(Twist, '/wheelchair2_base_controller/cmd_vel_unstamped', 10)
        self.marker_publisher = self.create_publisher(MarkerArray, '/future_states', 10)

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
            trans = self.tfbuffer.lookup_transform("map", "base_link", rclpy.time.Time())
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
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            pass

    def obstacle_callback(self, message: ObstacleArrayMsg):
        if self.counter == 0:
            self.static_obstacle_list = []
            for obstacle in message.obstacles:
                if len(obstacle.polygon.points[:-1]) > 2:
                    points = [
                        (point.x, point.y)
                        for point in cast(List[Point32], obstacle.polygon.points[:-1])
                    ]
                else:
                    continue
                self.static_obstacle_list.append(
                    StaticObstacle(
                        id=obstacle.id,
                        geometry=Polygon(vertices=points),
                    )
                )
            self.counter += 1