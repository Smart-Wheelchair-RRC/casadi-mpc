#!/usr/bin/env python3

import rospy
from nav_msgs.msg import Path, Odometry
# import rosinterface
from geometry_msgs.msg import PoseStamped

path = Path()
odom = Odometry()
# mpc_ros = rosinterface.ROSInterface()
def path_callback(msg: Path):
	path = msg
	# for pose in path.poses:
	# 	print(pose.pose.position.x, pose.pose.position.y)
	print(path.header)
	print("========================================================")
def odom_callback(msg):
	odom = msg


if __name__ == '__main__':
	rospy.init_node('planner_node')
	rospy.Subscriber('/move_base/VoronoiPlannerROS/voronoi_path', Path, path_callback)
	rospy.Subscriber('/odom', Odometry, odom_callback)
	
	# mpc_ros.run()
	rospy.spin()