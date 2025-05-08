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
        self.vertices += value