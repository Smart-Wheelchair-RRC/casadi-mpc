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
    def update_patch(self, patch: mpatches.Patch):
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
        # Calculate the distance from the point (distance_to) to the polygon
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
        return mpatches.Polygon(self.vertices, fill=False, color="black")

    def update_patch(self, patch: mpatches.Polygon):
        patch.set_xy(self.vertices)

    def from_rectangle(height: float, width: float, location: Tuple) -> "Polygon":
        return Polygon(
            [
                (location[0] - width / 2, location[1] - height / 2),
                (location[0] + width / 2, location[1] - height / 2),
                (location[0] + width / 2, location[1] + height / 2),
                (location[0] - width / 2, location[1] + height / 2),
            ]
        )


class Circle(Geometry):
    def __init__(self, center: Tuple, radius: float):
        super().__init__()
        self.radius = radius
        self.center = np.array(center, dtype=np.float64)

    @property
    def location(self) -> Tuple:
        return tuple(self.center)

    @location.setter
    def location(self, value: Tuple) -> None:
        self.center += np.array(value) - self.center

    def calculate_distance(
        self, distance_to: Tuple, custom_self_location: Tuple = None
    ) -> float:
        if custom_self_location is not None:
            center = np.array(custom_self_location)
        else:
            center = self.center
        return np.linalg.norm(np.array(distance_to[:2]) - center) - self.radius

    def calculate_symbolic_distance(
        self, distance_to: ca.MX, custom_self_location: Tuple = None
    ) -> ca.MX:
        if custom_self_location is not None:
            center = np.array(custom_self_location)
        else:
            center = self.center
        return (
            ca.sqrt(
                (distance_to[0] - center[0]) ** 2 + (distance_to[1] - center[1]) ** 2
            )
            - self.radius
        )

    def create_patch(self) -> mpatches.Circle:
        return mpatches.Circle(
            self.location, self.radius, fill=False, color="black", linestyle="--"
        )

    def update_patch(self, patch: mpatches.Circle):
        # Plot the circle
        patch.set_center(self.location)

    @staticmethod
    def create_circle_from_line(start: Tuple[float, float], end: Tuple[float, float], radius: float) -> List["Circle"]:
        start = np.array(start, dtype=np.float64)
        end = np.array(end, dtype=np.float64)

        direction = end - start
        total_length = np.linalg.norm(direction)
        
        if total_length == 0:
            return [Circle(tuple(start), radius)] 

        direction_unit = direction / total_length
        num_circles = int(total_length // (2 * radius)) + 1

        centers = [start + i * 2 * radius * direction_unit for i in range(num_circles)]
        circles = [Circle(tuple(center), radius) for center in centers]
        
        return circles


# class Ellipsoid(Geometry):
#     def __init__(self, a: float, b: float):
#         super().__init__()
#         self.a = a
#         self.b = b
#         self.patch: mpatches.Ellipse = mpatches.Ellipse(
#             (0, 0), 2 * self.a, 2 * self.b, fill=False, color="black", linestyle="--"
#         )

#     def from_circle(circle: Circle):
#         return Ellipsoid(circle.radius, circle.radius)

#     def from_rectangle(rectangle: Rectangle):
#         # Return ellipsoid approximating the rectangle
#         return Ellipsoid(
#             rectangle.width / 2,
#             rectangle.height / 2,
#         )

#     def calculate_distance(self, distance_to: np.ndarray, state: np.ndarray) -> float:
#         x_coordinate = abs(distance_to[0] - state[0])
#         y_coordinate = abs(distance_to[1] - state[1])
#         return np.sqrt((x_coordinate / self.a) ** 2 + (y_coordinate / self.b) ** 2)

#     def calculate_symbolic_distance(
#         self, distance_to: ca.MX, state: np.ndarray
#     ) -> ca.MX:
#         x_coordinate = ca.fabs(distance_to[0] - state[0])
#         y_coordinate = ca.fabs(distance_to[1] - state[1])
#         return ca.sqrt((x_coordinate / self.a) ** 2 + (y_coordinate / self.b) ** 2)

#     def update_patch(self, state: np.ndarray):
#         # Plot the ellipse
#         self.patch.set_center((state[0], state[1]))
