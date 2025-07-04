from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Tuple, Union
import numpy as np

from mpc.geometry import Circle
from mpc.optimizer import MotionPlanner

class Model(ABC):
    def __init__(
        self,
        id: int,
        radius: float,
        initial_position: Tuple[float, float],
        initial_orientation: float,
        planning_time_step: float,
        horizon: int,
        initial_linear_velocity: float,
        initial_angular_velocity: float,
        linear_velocity_bounds: Tuple[float, float],
        angular_velocity_bounds: Tuple[float, float],
        state_bounds: Tuple[float, float],
        goal_position: Tuple[float, float] = None,
        goal_orientation: float = None,
        use_warm_start: bool = False,
    ):
        assert horizon > 0
        
        self.id = id
        self.geometry = Circle(center=initial_position, radius=radius)

        self.initial_state = np.array([*initial_position, initial_orientation])
        self.goal_state = (
            np.array([*goal_position, goal_orientation])
            if goal_position
            else self.initial_state
        )

        self.horizon = horizon

        self.time_step = planning_time_step
        self.linear_velocity_bounds: Tuple[float, float] = linear_velocity_bounds
        self.angular_velocity_bounds: Tuple[float, float] = angular_velocity_bounds
        self.state_bounds: Tuple[float, float] = state_bounds
        
        self.initial_linear_velocity: float = initial_linear_velocity
        self.initial_angular_velocity: float = initial_angular_velocity

        self.linear_velocity: float = self.initial_linear_velocity
        self.angular_velocity: float = self.initial_angular_velocity
        
        self.state_matrix = np.tile(self.initial_state, (self.horizon + 1, 1)).T
        self.controls_matrix = np.zeros((2, self.horizon))
        
        self.planner = MotionPlanner(time_step=self.time_step, horizon=self.horizon)
        
        self.use_warm_start = use_warm_start
        self.goal_radius = 0.5
        
        
    