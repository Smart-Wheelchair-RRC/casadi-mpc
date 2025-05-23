from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, cast

import casadi as ca
import numpy as np

if TYPE_CHECKING:
    from mpc.geometry import Geometry


class Obstacle(ABC):
    def __init__(
        self,
        id: int,
        geometry: "Geometry",
    ):
        self.id = id
        self.geometry = geometry

    @property
    def state(self):
        return np.array(self.geometry.location + (0,))

    @abstractmethod
    def calculate_matrix_distance(self, states_matrix: np.ndarray):
        raise NotImplementedError

    @abstractmethod
    def calculate_symbolic_matrix_distance(self, symbolic_states_matrix: ca.MX):
        raise NotImplementedError

    def calculate_distance(self, state: np.ndarray):
        return self.geometry.calculate_distance(state)

    def calculate_symbolic_distance(self, symbolic_state: ca.MX):
        return self.geometry.calculate_symbolic_distance(symbolic_state)


class StaticObstacle(Obstacle):
    def __init__(
        self,
        id: int,
        geometry: "Geometry",
    ):
        super().__init__(id=id, geometry=geometry)

    def calculate_matrix_distance(self, states_matrix: np.ndarray):
        return np.stack(
            [self.geometry.calculate_distance(state) for state in states_matrix.T]
        )

    def calculate_symbolic_matrix_distance(self, symbolic_states_matrix: ca.MX):
        return cast(
            ca.MX,
            ca.vertcat(
                *[
                    self.geometry.calculate_symbolic_distance(
                        symbolic_states_matrix[:2, time_step]
                    )
                    for time_step in range(symbolic_states_matrix.shape[1])
                ]
            ),
        )
