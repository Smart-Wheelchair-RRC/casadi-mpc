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
        planning_time_step: float = 0.041,
        horizon: int = 50,
        initial_linear_velocity: float = 0,
        initial_angular_velocity: float = 0,
        linear_velocity_bounds: Tuple[float, float] = (0, 0.5), #change
        angular_velocity_bounds: Tuple[float, float] = (-0.8, 0.8),
        state_bounds: Tuple[float, float] = (-10, 10),
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
        
    def update_goal(self, goal: np.ndarray):
        self.goal_state = goal if (goal is not None) else self.initial_state
        
    def state(self):
        return self.state_matrix[:, 1]
    
    def at_goal(self):
        return self.geometry.calculate_distance(self.goal_state) - self.goal_radius <= 0
    
    def reset(self, matrices_only: bool = False, to_initial_state: bool = True):
        self.states_matrix = np.tile(
            (self.initial_state if to_initial_state else self.state),
            (self.horizon + 1, 1),
        ).T
        self.controls_matrix = np.zeros((2, self.horizon))
        if not matrices_only:
            self.linear_velocity = self.initial_linear_velocity
            self.angular_velocity = self.initial_angular_velocity
    def step(
        self,
        state_override: bool = False,
    ):
        self.states_matrix, self.controls_matrix = self.planner.solve(
            current_state=self.state if not state_override else self.initial_state,
            current_linear_velocity=self.linear_velocity,
            current_angular_velocity=self.angular_velocity,
            goal_state=self.goal_state,
            states_matrix=self.state_matrix,
            controls_matrix=self.controls_matrix,
            state_bounds=self.state_bounds,
            linear_velocity_bounds=self.linear_velocity_bounds,
            angular_velocity_bounds=self.angular_velocity_bounds,
        )
        self.geometry.location = self.state[:2] if not state_override else self.initial_state[:2]
        self.linear_velocity = self.controls_matrix[0, 0]
        self.angular_velocity = self.controls_matrix[1, 0] 
    
            
    