#!/usr/bin/env python3
from typing import List, cast

import numpy as np
import rospy
from costmap_converter.msg import ObstacleArrayMsg, ObstacleMsg
from geometry_msgs.msg import Point32, PoseWithCovariance, Twist
from nav_msgs.msg import Odometry, Path
from people_msgs.msg import People, Person
from tf.transformations import euler_from_quaternion

from mpc.agent import EgoAgent
from mpc.dynamic_obstacle import DynamicObstacle
from mpc.environment import ROSEnvironment
from mpc.geometry import Circle, Polygon
from mpc.obstacle import StaticObstacle


class ROSInterface:
    """
    ROSInterface class to interface with ROS
    Creates a node and subscribes to people messages for obstacles, and publishes commands on the /cmd_vel topic
    Also subscribes to waypoint pose messages for the next goal
    """

    def __init__(self):
        self.environment = ROSEnvironment(
            agent=EgoAgent(
                id=1,
                radius=0.4,
                initial_position=(0, 0),
                initial_orientation=np.deg2rad(90),
                horizon=30,
                # planning_time_step=0.5,
                use_warm_start=True,
                planning_time_step=0.5,
                linear_velocity_bounds=(-0.26, 0.26),
                angular_velocity_bounds=(-5, 5),
                linear_acceleration_bounds=(-0.1, 0.1),
                angular_acceleration_bounds=(-5, 5),
            ),
            static_obstacles=[],
            dynamic_obstacles=[],
            waypoints=[],
        )

        rospy.init_node("ros_mpc_interface")

        rospy.Subscriber("/people", People, self.people_callback)
        rospy.Subscriber("/move_base/GlobalPlanner/plan", Path, self.waypoint_callback)
        rospy.Subscriber("/odom", Odometry, self.odom_callback)
        rospy.Subscriber(
            "/costmap_converter/costmap_obstacles",
            ObstacleArrayMsg,
            self.obstacle_callback,
        )

        self.velocity_publisher = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

    def run(self):
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            self.environment.step()
            print(self.environment.agent.goal_state, self.environment.agent.state)
            print(
                "Velocity",
                self.environment.agent.linear_velocity,
                self.environment.agent.angular_velocity,
            )

            # Publish the control command
            control_command = Twist()
            control_command.linear.x = self.environment.agent.linear_velocity
            control_command.angular.z = self.environment.agent.angular_velocity
            print(control_command.linear.x, control_command.angular.z)

            self.velocity_publisher.publish(control_command)

            rate.sleep()

    def odom_callback(self, message: Odometry):
        # Update the agent's state with the current position and orientation
        self.environment.agent.initial_state = np.array(
            [
                message.pose.pose.position.x,
                message.pose.pose.position.y,
                euler_from_quaternion(
                    [
                        message.pose.pose.orientation.x,
                        message.pose.pose.orientation.y,
                        message.pose.pose.orientation.z,
                        message.pose.pose.orientation.w,
                    ]
                )[2],
            ]
        )
        # self.environment.agent.reset(matrices_only=True)

    def obstacle_callback(self, message: ObstacleArrayMsg):
        static_obstacle_list = []

        for obstacle in message.obstacles:
            obstacle: ObstacleMsg
            # Create a static obstacle for each polygon
            points = [
                (point.x, point.y)
                for point in cast(List[Point32], obstacle.polygon.points[:-1])
            ]
            static_obstacle_list.append(
                StaticObstacle(
                    id=obstacle.id,
                    geometry=Polygon(
                        # id=obstacle.id,
                        vertices=points,
                    ),
                )
            )

        self.environment.static_obstacles = static_obstacle_list
        self.environment.plotter.update_static_obstacles(static_obstacle_list)

    def people_callback(self, message: People):
        # Create a dynamic obstacle for each person
        dynamic_obstacle_list = []

        for person in message.people:
            person: Person
            dynamic_obstacle_list.append(
                DynamicObstacle(
                    id=person.name,
                    position=(person.position.x, person.position.y),
                    orientation=np.arctan2(person.velocity.y, person.velocity.x),
                    linear_velocity=(person.velocity.x**2 + person.velocity.y**2),
                    angular_velocity=0,
                    horizon=20,
                )
            )

        self.environment.dynamic_obstacles = dynamic_obstacle_list

    def waypoint_callback(self, message: Path):
        # Update the agent's goal with the waypoint position
        # if message.header.seq == 0:
        print("Updating waypoints")
        # waypoints = [
        #     (pose.pose.position.x, pose.pose.position.y, euler_from_quaternion([pose.pose.orientation.x,pose.pose.orientation.y,pose.pose.orientation.z,pose.pose.orientation.w])[2])
        #     for pose in message.poses[::10]
        # ]
        waypoints = []
        # orientation_euler = euler_from_quaternion((0, 0, message.poses[-1].pose.orientation, 0))
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
        print("Waypoint", waypoints)
        self.environment.waypoints = np.array(waypoints)
        self.environment.waypoint_index = 0
        self.environment.agent.update_goal(self.environment.current_waypoint)


if __name__ == "__main__":
    ros_interface = ROSInterface()
    ros_interface.run()
