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
                radius=0.5,
                initial_position=(0, 0),
                initial_orientation=np.deg2rad(90),
                horizon=7,
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
            plot=True,
        )
        self.counter = 0

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)


        #SUBSCRIBERS
        self.create_subscription(Path, '/plan', self.waypoint_callback, 10)

        occupancy_map_subscriber = message_filters.Subscriber(
            self, OccupancyGrid, "/global_costmap/costmap"
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
         
    def obstacle_callback(self, msg: OccupancyGrid):
        if self.counter == 0:
            import time
            self.static_obstacle_list = []
            
            area_threshold = 3

            start_time = time.time()

            image = cv2.imread('/home/container_user/wheelchair2/src/wheelchair2_navigation/maps/sim/obstacles_world.pgm')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edged = cv2.Canny(gray, 150, 255)

            contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            print("Total contours found:", len(contours))

            convex_polygons = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > area_threshold:
                    hull = cv2.convexHull(contour)
                    if len(hull) >= 1 and len(hull) <= 11:
                        convex_polygons.append(hull)

            print("Convex polygons:", len(convex_polygons))

            output_img = image.copy()
            cv2.drawContours(output_img, convex_polygons, -1, (0, 255, 0), 2)

            end_time = time.time()
            print(f"Time taken: {(end_time - start_time)*1000:.2f} ms")
            

            # contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # self.static_obstacle_list = []
            # min_obstacle_area = 0.3
            # for contour in contours:
            #     area = cv2.contourArea(contour)
            #     if area < min_obstacle_area: 
            #         continue
                    
            #     # Simplify contour to reduce vertex count
            #     epsilon = 0.03 * cv2.arcLength(contour, True)  
            #     simplified = cv2.approxPolyDP(contour, epsilon, True)
                
                # if len(simplified) >= 2:
                #     polygon = []
                #     for pt in simplified:
                #         x = pt[0][0] * resolution + origin.position.x
                #         y = pt[0][1] * resolution + origin.position.y
                #         polygon.append((x, y))
                    
            vertices = []
            for pt in hull:
                x = pt[0][0] * msg.info.resolution + msg.info.origin.position.x
                y = pt[0][1] * msg.info.resolution + msg.info.origin.position.y
                vertices.append((x, y))

            self.static_obstacle_list.append(
                StaticObstacle(
                    id=len(self.static_obstacle_list),
                    geometry=Polygon(vertices=vertices)
                )
            )
            cv2.imshow('Convex Polygons', output_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            self.counter += 1
        else:
            pass

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