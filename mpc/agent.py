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
        avoid_obstacles: bool,
        linear_velocity_bounds: Tuple[float, float],
        angular_velocity_bounds: Tuple[float, float],
        linear_acceleration_bounds: Tuple[float, float],
        angular_acceleration_bounds: Tuple[float, float],
        left_right_lane_bounds: Tuple[float, float],
        goal_position: Tuple[float, float] = None,
        goal_orientation: float = None,
        use_warm_start: bool = False,
    ):
        assert horizon > 0, "Horizon must be greater than 0"

        self.id = id
        self.geometry = Circle(center=initial_position, radius=radius)
        self.sensor_radius = sensor_radius

        self.avoid_obstacles = avoid_obstacles

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
        self.linear_acceleration_bounds: Tuple[float, float] = (
            linear_acceleration_bounds
        )
        self.angular_acceleration_bounds: Tuple[float, float] = (
            angular_acceleration_bounds
        )

        self.left_right_lane_bounds: Tuple[float, float] = left_right_lane_bounds

        self.initial_linear_velocity: float = initial_linear_velocity
        self.initial_angular_velocity: float = initial_angular_velocity

        self.linear_velocity: float = self.initial_linear_velocity
        self.angular_velocity: float = self.initial_angular_velocity

        self.states_matrix = np.tile(self.initial_state, (self.horizon + 1, 1)).T
        self.controls_matrix = np.zeros((2, self.horizon))

        self.planner = MotionPlanner(time_step=self.time_step, horizon=self.horizon)

        self.use_warm_start = use_warm_start
        self.goal_radius = 0.5

    def update_goal(self, goal: np.ndarray):
        self.goal_state = goal if (goal is not None) else self.initial_state

    @property
    def state(self):
        return self.states_matrix[:, 1]

    @abstractmethod
    def step(self):
        raise NotImplementedError

    @property
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


class EgoAgent(Agent):
    def __init__(
        self,
        id: int,
        radius: float,
        initial_position: Tuple[float, float],
        initial_orientation: float,
        planning_time_step: float = 0.041,
        initial_linear_velocity: float = 0,
        initial_angular_velocity: float = 0,
        horizon: int = 50,
        sensor_radius: float = 5,
        linear_velocity_bounds: Tuple[float, float] = (0, 12),
        angular_velocity_bounds: Tuple[float, float] = (-0.78, 0.78),
        linear_acceleration_bounds: Tuple[float, float] = (-5, 5),
        angular_acceleration_bounds: Tuple[float, float] = (-10, 10),
        left_right_lane_bounds: Tuple[float, float] = (-10, 10),
        goal_position: Tuple[float, float] = None,
        goal_orientation: float = None,
        use_warm_start: bool = False,
    ):
        super().__init__(
            id=id,
            radius=radius,
            initial_position=initial_position,
            initial_orientation=initial_orientation,
            goal_position=goal_position,
            goal_orientation=goal_orientation,
            planning_time_step=planning_time_step,
            initial_linear_velocity=initial_linear_velocity,
            initial_angular_velocity=initial_angular_velocity,
            horizon=horizon,
            sensor_radius=sensor_radius,
            avoid_obstacles=True,
            linear_velocity_bounds=linear_velocity_bounds,
            angular_velocity_bounds=angular_velocity_bounds,
            linear_acceleration_bounds=linear_acceleration_bounds,
            angular_acceleration_bounds=angular_acceleration_bounds,
            left_right_lane_bounds=left_right_lane_bounds,
            use_warm_start=use_warm_start,
        )

    def step(
        self,
        static_obstacles: List["StaticObstacle"] = [],
        dynamic_obstacles: List[
            Union["DynamicObstacle", "SimulatedDynamicObstacle"]
        ] = [],
    ):
        if not self.use_warm_start:
            self.reset(matrices_only=True, to_initial_state=False)
        self.states_matrix, self.controls_matrix = self.planner.solve(
            current_state=self.state,
            current_linear_velocity=self.linear_velocity,
            current_angular_velocity=self.angular_velocity,
            goal_state=self.goal_state,
            states_matrix=self.states_matrix,
            controls_matrix=self.controls_matrix,
            left_right_lane_bounds=self.left_right_lane_bounds,
            linear_velocity_bounds=self.linear_velocity_bounds,
            angular_velocity_bounds=self.angular_velocity_bounds,
            linear_acceleration_bounds=self.linear_acceleration_bounds,
            angular_acceleration_bounds=self.angular_acceleration_bounds,
            inflation_radius=(self.geometry.radius + 0.4),
            static_obstacles=static_obstacles,
            dynamic_obstacles=dynamic_obstacles,
        )
        self.geometry.location = self.state[:2]
        self.linear_velocity += self.controls_matrix[0, 0] * self.time_step
        self.angular_velocity += self.controls_matrix[1, 0] * self.time_step
