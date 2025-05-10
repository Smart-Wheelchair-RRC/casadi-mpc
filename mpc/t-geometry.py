from abc import ABC, abstractmethod
from typing import List, Tuple

import casadi as ca
import numpy as np
from matplotlib import patches as mpatches


class Geometry(ABC):
    @property
    @abstractmethod
    def location(self) -> Tuple:
        raise NotImplementedError
    
    @location.setter
    @abstractmethod
    def location(self, value: Tuple) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def calculate_distance(
        self, distance_to: Tuple, custom_self_location: Tuple = None
    ) -> float:
        raise NotImplementedError
    
    @abstractmethod
    def calculate_symbolic_distance(
        self, distance_to: ca.MX, custom_self_location: Tuple = None 
    ) -> ca.MX:
        raise NotImplementedError
    
    @abstractmethod
    def create_patch(self) -> mpatches.Patch:
        raise NotImplementedError
    
    @abstractmethod
    def update_patch(self, path: mpatches.Patch):
        raise NotImplementedError
    

class Polygon(Geometry):
    def __init__(self, vertices: List[Tuple]):
        super().__init__()
        self.vertices = np.array(vertices)

    @property
    def location(self) -> Tuple:
        return tuple(np.mean(self.vertices, axis=0))
    
    @location.setter
    def location(self, value: Tuple) -> None:
        self.vertices += value - self.location

    def calculate_distance(
        self, distance_to: Tuple, custom_self_location: Tuple = None
    ) -> float:
        if custom_self_location is not None:
            a = self.vertices + custom_self_location - self.location
        else:
            a = self.vertices
        b = np.roll(a, -1, axis=0)
        edge = b - a
        v = np.array(distance_to[:2]) - a
        pq = (
            v
            - edge
            * np.clip(np.sum(v * edge, axis=1) / np.sum(edge * edge, axis=1), 0, 1)[
                :, None
            ]
        )
        distance = np.min(np.sum(pq**2, axis=1))

        v2 = distance_to[:2] - b
        val3 = np.roll(edge, 1, axis=1) * v
        val3 = val3[:, 1] - val3[:, 0]
        condition = np.stack([v[:, 1] >= 0, v2[:, 1] < 0, val3 > 0])
        not_condition = np.stack([v[:, 1] < 0, v2[:, 1] >= 0, val3 < 0])
        condition = np.all(np.all(condition, axis=0))
        not_condition = np.all(np.all(not_condition, axis=0))
        s = -1 if condition or not_condition else 1
        return np.sqrt(distance) * s
    
    def calculate_symbolic_distance(
        self, distance_to: ca.MX, custom_self_location: Tuple = None
    ) -> ca.MX:
        # Calculate the distance from the point (distance_to) to the polygon
        if custom_self_location is not None:
            a = self.vertices + custom_self_location - self.location
        else:
            a = self.vertices
        b = np.roll(a, -1, axis=0)
        edge = b - a
        v = ca.repmat(distance_to[:2].T, a.shape[0], 1) - a
        pq = v - edge * ca.fmin(ca.fmax(ca.sum2(v * edge) / ca.sum2(edge * edge), 0), 1)
        distance = ca.mmin(ca.sum2(pq**2))

        v2 = ca.repmat(distance_to[:2].T, b.shape[0], 1) - b
        val3 = np.roll(edge, 1, axis=1) * v
        val3 = val3[:, 1] - val3[:, 0]
        condition = ca.horzcat(v[:, 1] >= 0, v2[:, 1] < 0, val3 > 0)
        not_condition = ca.horzcat(v[:, 1] < 0, v2[:, 1] >= 0, val3 < 0)
        condition = ca.sum1(ca.sum2(condition))
        not_condition = ca.sum1(ca.sum2(not_condition))
        return ca.if_else(
            ca.eq(ca.sum1(ca.vertcat(condition, not_condition)), 1),
            ca.sqrt(distance) * -1,
            ca.sqrt(distance) * 1,
        )

    def create_patch(self) -> mpatches.Polygon:
        return mpatches.Polygon



