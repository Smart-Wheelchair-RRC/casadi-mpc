from typing import Tuple, cast

import casadi as ca
import numpy as np

from mpc.geometry import Circle
from mpc.obstacle import Obstacle
from mpc.shadowagent import ShadowAgent


class DynamicObstacle(Obstacle):
    def __init__(
        self,
        id: int,
        position: Tuple[float, float],
        orientation: float = np.deg2rad(90),
        linear_velocity: float = 1,
        angular_velocity: float = 0,
        horizon: int = 50,  # Make sure this is equal to the horizon of the ego agent
    ):
        super().__init__(id=id, geometry=Circle(center=position, radius=1))
        self.linear_velocity = linear_velocity
        self.angular_velocity = angular_velocity
        self.horizon = horizon
        self.states_matrix = self._get_predicted_states_matrix(horizon)

    def _predict_state(self, state: np.ndarray):
        return np.array(
            [
                state[0] + self.linear_velocity * np.cos(np.deg2rad(state[2])),
                state[1] + self.linear_velocity * np.sin(np.deg2rad(state[2])),
                state[2] + self.angular_velocity,
            ]
        )

    def _get_predicted_states_matrix(self, horizon: int):
        predicted_states = np.zeros((3, horizon))
        predicted_states[:, 0] = self.state

        for time_step in range(1, horizon):
            predicted_states[:, time_step] = self._predict_state(
                predicted_states[:, time_step - 1]
            )

        return predicted_states

    def calculate_matrix_distance(self, states_matrix: np.ndarray):
        return np.stack(
            [
                self.geometry.calculate_distance(state, self.states_matrix[:2, index])
                for index, state in enumerate(states_matrix.T)
            ]
        )

    def calculate_symbolic_matrix_distance(self, symbolic_states_matrix: ca.MX):
        return cast(
            ca.MX,
            ca.vertcat(
                *[
                    self.geometry.calculate_symbolic_distance(
                        symbolic_states_matrix[:2, time_step],
                        self.states_matrix[:2, time_step],
                    )
                    for time_step in range(symbolic_states_matrix.shape[1])
                ]
            ),
        )

    @property
    def state(self):
        return self.states_matrix[:, 0]


class SimulatedDynamicObstacle(Obstacle):
    def __init__(
        self,
        id: int,
        position: Tuple[float, float],
        orientation: float = np.deg2rad(90),
        goal_position: Tuple[float, float] = (0, 0),
        goal_orientation: float = np.deg2rad(90),
        horizon: int = 50,  # Make sure this is equal to the horizon of the ego agent
    ):
        super().__init__(id=id, geometry=Circle(center=position, radius=1))
        self.shadow_agent = ShadowAgent(
            id=id,
            radius=1,
            initial_position=position,
            initial_orientation=orientation,
            goal_position=goal_position,
            goal_orientation=goal_orientation,
            horizon=horizon,
            use_warm_start=False,
        )
        self.horizon = horizon

    def calculate_matrix_distance(self, states_matrix: np.ndarray):
        return np.stack(
            [
                self.geometry.calculate_distance(state, self.states_matrix[:2, index])
                for index, state in enumerate(states_matrix.T)
            ]
        )

    def calculate_symbolic_matrix_distance(self, symbolic_states_matrix: ca.MX):
        return cast(
            ca.MX,
            ca.vertcat(
                *[
                    self.geometry.calculate_symbolic_distance(
                        symbolic_states_matrix[:2, time_step],
                        self.states_matrix[:2, time_step],
                    )
                    for time_step in range(symbolic_states_matrix.shape[1])
                ]
            ),
        )

    def step(self):
        self.shadow_agent.step()
        self.geometry.location = self.shadow_agent.state[:2]

    @property
    def state(self):
        return self.shadow_agent.state

    def reset(self):
        self.shadow_agent.reset()

    @property
    def linear_velocity(self):
        return self.shadow_agent.linear_velocity

    @property
    def angular_velocity(self):
        return self.shadow_agent.angular_velocity

    @property
    def states_matrix(self):
        return self.shadow_agent.states_matrix
