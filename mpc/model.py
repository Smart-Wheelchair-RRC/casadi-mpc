from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Tuple
import numpy as np
import time
from pathlib import Path
from mpc.geometry import Circle
from mpc.optimizer import MotionPlanner

class Model(ABC):
    def __init__(
        self,
        id: int,
        initial_position: Tuple[float, float],
        initial_orientation: float,
        planning_time_step: float = 0.8,
        horizon: int = 7,
        initial_linear_velocity: float = 0,
        initial_angular_velocity: float = 0,
        linear_velocity_bounds: Tuple[float, float] = (0, 0.3), 
        angular_velocity_bounds: Tuple[float, float] = (-0.3, 0.3),
        state_bounds: Tuple[float, float] = (-10, 10),
        goal_position: Tuple[float, float] = None,
        goal_orientation: float = None,
        waypoints: List[Tuple[Tuple, float]] = None,
        use_warm_start: bool = False,
        
    ):
        assert horizon > 0
        self.waypoints = waypoints
        self.waypoint_index = 0
        
        self.id = id

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
       
        self.states_matrix = np.tile(self.initial_state, (self.horizon + 1, 1)).T
        self.controls_matrix = np.zeros((2, self.horizon))
        
        self.planner = MotionPlanner(time_step=self.time_step, horizon=self.horizon)
        
        self.use_warm_start = use_warm_start
        self.goal_radius = 0.5
        
        self.update_goal(self.current_waypoint())
    
    def current_waypoint(self):
        if self.waypoints is None or self.waypoint_index >= len(self.waypoints):
            return None
        return self.waypoints[self.waypoint_index]
    
    def final_goal_reached(self):
        return self.waypoint_index == len(self.waypoints) - 1 and self.at_goal
    
    def update_goal(self, goal):
        
        if goal is not None:
            if isinstance(goal, tuple):
                self.goal_state = np.array(goal)
            else:
                self.goal_state = goal
        else:
            self.goal_state = self.initial_state
        
    def state(self):
        
        return self.states_matrix[:, 0] 
    
    @property
    def at_goal(self):
        current_pos = self.state()[:2] 
        goal_pos = self.goal_state[:2]
        distance = np.linalg.norm(current_pos - goal_pos)
        return distance <= self.goal_radius
    
    def reset(self, matrices_only: bool = False, to_initial_state: bool = True):
        self.states_matrix = np.tile(
            (self.initial_state if to_initial_state else self.state()),
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
        print("step function is running")
        if self.at_goal and not self.final_goal_reached():
            print("Reached waypoint", self.waypoint_index + 1)
            self.waypoint_index += 1
            self.update_goal(self.current_waypoint())
            
        current_state = self.state() if not state_override else self.initial_state
        
        self.states_matrix, self.controls_matrix = self.planner.solve(
            current_state=current_state,
            current_linear_velocity=self.linear_velocity,
            current_angular_velocity=self.angular_velocity,
            goal_state=self.goal_state,
            states_matrix=self.states_matrix, 
            controls_matrix=self.controls_matrix,
            state_bounds=self.state_bounds,
            linear_velocity_bounds=self.linear_velocity_bounds,
            angular_velocity_bounds=self.angular_velocity_bounds,
        )
            
        self.linear_velocity = self.controls_matrix[0, 0]
        self.angular_velocity = self.controls_matrix[1, 0]