"""
This module is created in a separate file to prevent circular imports. Will be removed in a future update.
"""

from typing import Tuple

from mpc.agent import Agent


class ShadowAgent(Agent):
    def __init__(
        self,
        id: int,
        radius: float,
        initial_position: Tuple[float, float],
        initial_orientation: float,
        goal_position: Tuple[float, float],
        goal_orientation: float,
        planning_time_step: float = 0.041,
        initial_linear_velocity: float = 0,
        initial_angular_velocity: float = 0,
        horizon: int = 50,
        linear_velocity_bounds: Tuple[float, float] = (0, 12),
        angular_velocity_bounds: Tuple[float, float] = (-0.78, 0.78),
        linear_acceleration_bounds: Tuple[float, float] = (-5, 5),
        angular_acceleration_bounds: Tuple[float, float] = (-10, 10),
        left_right_lane_bounds: Tuple[float, float] = (-1000, 1000),
        use_warm_start: bool = True,
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
            sensor_radius=0,
            avoid_obstacles=False,
            linear_velocity_bounds=linear_velocity_bounds,
            angular_velocity_bounds=angular_velocity_bounds,
            linear_acceleration_bounds=linear_acceleration_bounds,
            angular_acceleration_bounds=angular_acceleration_bounds,
            left_right_lane_bounds=left_right_lane_bounds,
            use_warm_start=use_warm_start,
        )

    def step(self):
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
        )
        self.geometry.location = self.state[:2]
        self.linear_velocity += self.controls_matrix[0, 0] * self.time_step
        self.angular_velocity += self.controls_matrix[1, 0] * self.time_step
