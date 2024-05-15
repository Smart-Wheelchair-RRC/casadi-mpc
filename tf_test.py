#!/usr/bin/env python3
import rospy
import tf2_ros
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped

if __name__ == "__main__":
	rospy.init_node("tf_pub")
	tfBuffer = tf2_ros.Buffer()
	listener = tf2_ros.TransformListener(tfBuffer)
	

	rate = rospy.Rate(10)
	while not rospy.is_shutdown():
		try:
			trans = tfBuffer.lookup_transform("odom", "base_footprint", rospy.Time())
		except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
				rate.sleep()
				continue
		
		print(trans.transform)