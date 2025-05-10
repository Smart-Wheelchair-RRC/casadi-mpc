#!/usr/bin/env python3
from typing import List, cast
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point32, Twist
from nav_msgs.msg import Odometry, Path
from scipy.spatial.transform import Rotation as R  # Replacement for tf_transformations

from mpc.agent import EgoAgent
from mpc.dynamic_obstacle import DynamicObstacle
from mpc.environment import ROSEnvironment
from mpc.geometry import Polygon
from mpc.obstacle import StaticObstacle

from tf2_ros import TransformListener, Buffer
import tf2_ros


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
            )
        )