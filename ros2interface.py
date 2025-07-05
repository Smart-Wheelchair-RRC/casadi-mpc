#!/usr/bin/env python3

import message_filters
import numpy as np
import rclpy
import rclpy.duration
import rclpy.time
import tf2_ros

from tf2_geometry_msgs import do_transform_pose_stamped
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.node import Node
from scipy.spatial.transform import (
    Rotation as R,  # Replacement for tf_transformations from ROS1
)
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray
from mpc.model import Model

from mpc.agent import EgoAgent
from mpc.environment import ROSEnvironment
from mpc.geometry import Circle
from mpc.obstacle import StaticObstacle
import cv2
from mpc.geometry import Polygon

def euler_from_quaternion(quat, degree = False):
    return R.from_quat(quat).as_euler('xyz')

class ROS2Interface(Node):
    def __init__(self):
        super().__init__('ros_mpc_interface')
        
        self.model = Model(
            id = 1,
            initial_position=(0, 0),
            initial_orientation=np.deg2rad(90),
            horizon=7,
            use_warm_start=True,
            planning_time_step=0.8,
            linear_velocity_bounds=(0, 0.3),
            angular_velocity_bounds=(-0.3, 0.3),
            waypoints=[],
        )
        
        self.counter = 0
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.waypoints=[]
        
        self.create_subscription(Path, '/plan', self.waypoint_callback, 10)
        self.create_subscription(Odometry, "/odom", self.odom_callback, 10)
        
        self.velocity_publisher = self.create_publisher(Twist, '/wheelchair2_base_controller/cmd_vel_unstamped', 10)
        self.marker_publisher = self.create_publisher(MarkerArray, '/future_states', 10)
        self.timer = self.create_timer(0.01, self.run)
    def run(self):
        if not self.waypoints:
            return
        print("run function is running")
        self.model.step()
        self.future_states_pub()
        
        control_command = Twist()
        control_command.linear.x = self.model.linear_velocity
        control_command.angular.z = self.model.angular_velocity
        self.velocity_publisher.publish(control_command)
        
    def future_states_pub(self):
        marker_array = MarkerArray()
        future_states = self.model.states_matrix
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
        pose = message.pose.pose
        self.model.initial_state = np.array(
            [
                pose.position.x,
                pose.position.y,
                euler_from_quaternion(
                    [
                        pose.orientation.x,
                        pose.orientation.y,
                        pose.orientation.z,
                        pose.orientation.w,
                    ]
                )[2],
            ]
        )
        self.model.reset(matrices_only=True)
        
    def waypoint_callback(self, message: Path):
        try:
            transform = self.tf_buffer.lookup_transform("odom", "map", rclpy.time.Time())
        except Exception as e:
            print(e)
            return
        
        poses = [
            do_transform_pose_stamped(pose, transform)
            for pose in message.poses
        ]

        try:
            diff = np.array(self.waypoints[-1]) - np.array(
                (
                    poses[-1].pose.position.x,
                    poses[-1].pose.position.y,
                    euler_from_quaternion(
                        [
                            poses[-1].pose.orientation.x,
                            poses[-1].pose.orientation.y,
                            poses[-1].pose.orientation.z,
                            poses[-1].pose.orientation.w,
                        ]
                    )[2],
                )
            )
            diff = diff.sum()
        except Exception:
            diff = 0
            
        if self.waypoints == [] or abs(diff) > 0.1:
            print("Updating goal")
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
            waypoints.append(
                (
                    poses[-1].pose.position.x,
                    poses[-1].pose.position.y,
                    euler_from_quaternion(
                        [
                            poses[-1].pose.orientation.x,
                            poses[-1].pose.orientation.y,
                            poses[-1].pose.orientation.z,
                            poses[-1].pose.orientation.w,
                        ]
                    )[2],
                )
            )
            self.waypoints = waypoints
            self.model.waypoints = np.array(waypoints)
            self.model.waypoint_index = 0
            self.model.update_goal(self.model.current_waypoint())

def main(args=None):
    rclpy.init(args=args)
    ros_interface = ROS2Interface()
    # ros_interface.run()
    rclpy.spin(ros_interface)
    ros_interface.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()