from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    from mpc.dynamic_obstacle import DynamicObstacle, SimulatedDynamicObstacle
    from mpc.obstacle import StaticObstacle

from mpc.geometry import Circle
from mpc.planner import MotionPlanner


class Agent(ABC):
    def __init__(
            self,
            id: int, 
            radius: float,
            initial_position: Tuple[float, float], 
            initial_orientation: float,
            planning_time_step: float,
            initial_linear_velocity: float,
            initial_angular_velocity: float,
            horizon: int,
            sensor_radius: float,
            
    )