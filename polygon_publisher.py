#!/usr/bin/env python3

import rospy
from costmap_converter.msg import ObstacleArrayMsg, ObstacleMsg


def obstacle_callback(msg: ObstacleArrayMsg):
	
	for obstacle in msg.obstacles:
		points = [(point.x,point.y) for point in obstacle.polygon.points]
		print(points)
		print("=============")
	print("++++++++++++++++++++++")

if __name__ == "__main__":
	rospy.init_node("polygon_publisher")
	rospy.Subscriber("/costmap_converter/costmap_obstacles", ObstacleArrayMsg, obstacle_callback)

	rospy.spin()