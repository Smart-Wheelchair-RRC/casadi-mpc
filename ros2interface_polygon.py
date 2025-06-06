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
        # self.create_subscription(PeopleVelocity, '/vel_pub', self.people_callback, 10)
        self.create_subscription(Path, '/plan', self.waypoint_callback, 10)
        # self.subscription = self.create_subscription(
        #     OccupancyGrid,
        #     '/local_costmap/costmap',
        #     # self.obstacle_callback,
        #     10)
        # # self.create_subscription(ObstacleArrayMsg, '/costmap_converter/costmap_obstacles', self.obstacle_callback, 10)
        # self.create_subscription(Odometry, '/wheelchair2_base_controller/odom', self.odom_callback, 10)

        occupancy_map_subscriber = message_filters.Subscriber(
            self, OccupancyGrid, "/local_cstmap/costmap"
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
        self.obstacle_callback(occupancy_map_msg)
        self.odom_callback(odometry_msg)

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

    def odom_callback(self, message: Odometry):
        pose = message.pose.pose
        self.environment.agent.initial_state = np.array(
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
        self.environment.agent.reset(matrices_only=True)

        # try:
        #     trans = self.tfbuffer.lookup_transform("map", "base_link", rclpy.time.Time())
        #     self.environment.agent.initial_state = np.array([
        #         trans.transform.translation.x,
        #         trans.transform.translation.y,
        #         euler_from_quaternion([
        #             trans.transform.rotation.x,
        #             trans.transform.rotation.y,
        #             trans.transform.rotation.z,
        #             trans.transform.rotation.w,
        #         ])[2],
        #     ])
        #     self.environment.agent.reset(matrices_only=True)
        # except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        #     pass
    # def obstacle_callback(self, msg: OccupancyGrid):
    #     width = msg.info.width
    #     height = msg.info.height
    #     resolution = msg.info.resolution
    #     origin = msg.info.origin
    #     print(height, width)

    #     grid = np.array(msg.data, dtype=np.int8).reshape((height, width))
    #     binary = np.uint8((grid > 50) * 255)

    #     contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #     self.static_obstacle_list = []
        
    #     for i, contour in enumerate(contours):
    #         if len(contour) < 3:
    #             continue
                
    #         # Calculate area and filter
    #         area_pixels = cv2.contourArea(contour)
    #         area_world = area_pixels * (resolution ** 2)
            
    #         if area_world < 0.05:
    #             continue
                
    #         # Get convex hull (much simpler shape)
    #         hull = cv2.convexHull(contour)
            
    #         # Further simplify if needed
    #         epsilon = 0.05 * cv2.arcLength(hull, True)
    #         simplified = cv2.approxPolyDP(hull, epsilon, True)
            
    #         # Convert to world coordinates
    #         polygon = []
    #         for pt in simplified:
    #             x = pt[0][0] * resolution + origin.position.x
    #             y = pt[0][1] * resolution + origin.position.y
    #             polygon.append((x, y))
            
    #         if len(polygon) >= 3:
    #             self.static_obstacle_list.append(
    #                 StaticObstacle(
    #                     id=i,
    #                     geometry=Polygon(vertices=polygon)  # Use optimized polygon below
    #                 )
    #             )
                
    def obstacle_callback(self, msg: OccupancyGrid):
        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution
        origin = msg.info.origin
        grid = np.array(msg.data, dtype=np.int8).reshape((height, width))
        binary = np.uint8((grid > 50) * 255)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self.static_obstacle_list = []
        min_obstacle_area = 0.3
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_obstacle_area: 
                continue
                
            # Simplify contour to reduce vertex count
            epsilon = 0.03 * cv2.arcLength(contour, True)  
            simplified = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(simplified) >= 2:
                polygon = []
                for pt in simplified:
                    x = pt[0][0] * resolution + origin.position.x
                    y = pt[0][1] * resolution + origin.position.y
                    polygon.append((x, y))
                
                self.static_obstacle_list.append(
                    StaticObstacle(
                        id=len(self.static_obstacle_list),
                        geometry=Polygon(vertices=polygon)
                    )
                )

    # def obstacle_callback(self, msg: OccupancyGrid):
    #     width = msg.info.width
    #     height = msg.info.height
    #     resolution = msg.info.resolution
    #     origin = msg.info.origin

    #     grid = np.array(msg.data, dtype=np.int8).reshape((height, width))
    #     binary = np.uint8((grid > 50) * 255)

    #     contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #     self.static_obstacle_list = []

    #     for contour in contours:
    #         if len(contour) >= 3:
    #             polygon = []
    #             for pt in contour:
    #                 x = pt[0][0] * resolution + origin.position.x
    #                 y = pt[0][1] * resolution + origin.position.y
    #                 polygon.append((x, y))
    #             self.static_obstacle_list.append(
    #                 StaticObstacle(
    #                     id=len(self.static_obstacle_list),
    #                     geometry=Polygon(vertices=polygon)
    #                 )
    #             )


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
            transform = self.tf_buffer.lookup_transform("odom", "map", rclpy.time.Time())
        except Exception as e:
            print(e)
            return
        
        poses = [
            do_transform_pose_stamped(pose, transform)
            for pose in message.poses
        ]

        # Check if the last waypoint is close to the current position
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

        # If there are no waypoints or the last waypoint is not close to the current position, update the waypoints
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
