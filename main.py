import numpy as np

from mpc.agent import EgoAgent
from mpc.dynamic_obstacle import SimulatedDynamicObstacle
from mpc.environment import LocalEnvironment
from mpc.geometry import Circle, Polygon
from mpc.obstacle import StaticObstacle

agent = EgoAgent(
    id=1,
    radius=0.2,
    initial_position=(0.5, 0.5),
    initial_orientation=np.deg2rad(90),
    horizon=3,
    use_warm_start=True,
    planning_time_step=0.5,
    linear_velocity_bounds=(-0.26, 0.26),
    angular_velocity_bounds=(-5, 5),
    linear_acceleration_bounds=(-0.1, 0.1),
    angular_acceleration_bounds=(-5, 5),
    sensor_radius=10,
)

# static_obstacle_rectangle = StaticObstacle(
#     id=1,
#     geometry=Rectangle(height=2, width=8),
#     position=(-4, 10),
# )
# static_obstacle_rectangle_2 = StaticObstacle(
#     id=2,
#     geometry=Rectangle(height=2, width=8),
#     position=(8, 10),
# )

# static_obstacle_ellipse = StaticObstacle(
#     id=2,
#     geometry=Ellipsoid.from_rectangle(Rectangle(height=2, width=8)),
#     position=(8, 10),
# )

static_obstacle_circle = StaticObstacle(
    id=3,
    geometry=Circle(center=(6, 10), radius=1),
)
static_obstacle_circle_2 = StaticObstacle(
    id=3,
    geometry=Circle(center=(8, 10), radius=1),
)
static_obstacle_circle_3 = StaticObstacle(
    id=3,
    geometry=Circle(center=(10, 10), radius=1),
)

static_polygonal_obstacle = StaticObstacle(
    id=1,
    geometry=Polygon(
        vertices=[
            (0, 0),
            (0, 5),
            (10, 5),
            (16, 6),
            (10, 0),
        ]
    ),
)

polygons = [
    [
        (-2.875, 0.025000037625432014),
        (-2.575000047683716, -0.42499998211860657),
        (-2.7249999046325684, 0.3750000298023224),
        # (-2.875, 0.025000037625432014),
    ],
    [
        (-2.674999952316284, 0.3750000298023224),
        (-2.2750000953674316, 1.125),
        # (-2.674999952316284, 0.3750000298023224),
    ],
    [
        (-2.625, -0.3749999701976776),
        (-2.2249999046325684, -1.125),
        # (-2.625, -0.3749999701976776),
    ],
    [
        (-2.2249999046325684, 1.125),
        (-1.875, 1.7750000953674316),
        # (-2.2249999046325684, 1.125),
    ],
    [
        (-2.174999952316284, -1.225000023841858),
        (-1.774999976158142, -1.875),
        (-2.174999952316284, -1.125),
        # (-2.174999952316284, -1.225000023841858),
    ],
    [
        (-1.875, 1.8250000476837158),
        (-1.225000023841858, 2.174999952316284),
        (-1.6749999523162842, 2.125),
        # (-1.875, 1.8250000476837158),
    ],
    [
        (-1.774999976158142, -1.9249999523162842),
        (-1.1749999523162842, -2.375),
        (-1.3249999284744263, -1.9249999523162842),
        # (-1.774999976158142, -1.9249999523162842),
    ],
    [
        (-1.1749999523162842, -2.325000047683716),
        (-1.0749999284744263, -2.5250000953674316),
        (-0.5249999761581421, -2.5250000953674316),
        # (-1.1749999523162842, -2.325000047683716),
    ],
    [
        (-1.1749999523162842, -1.125),
        (-0.9749999642372131, -1.1749999523162842),
        (-0.9249999523162842, -0.9249999523162842),
        (-1.1749999523162842, -0.9249999523162842),
        # (-1.1749999523162842, -1.125),
    ],
    [
        (-1.1749999523162842, 0.025000037625432014),
        (-0.9249999523162842, -0.07499996572732925),
        (-0.9249999523162842, 0.2250000387430191),
        # (-1.1749999523162842, 0.025000037625432014),
    ],
    [
        (-1.1749999523162842, 1.125),
        (-0.9749999642372131, 1.0250000953674316),
        (-0.824999988079071, 1.1750000715255737),
        (-1.024999976158142, 1.3250000476837158),
        # (-1.1749999523162842, 1.125),
    ],
    [
        (-0.875, 0.12500004470348358),
        # (-0.875, 0.12500004470348358),
    ],
    [
        (-0.47499996423721313, -2.5250000953674316),
        (0.2250000387430191, -2.4749999046325684),
        # (-0.47499996423721313, -2.5250000953674316),
    ],
    [
        (-0.07499996572732925, -1.125),
        (0.17500004172325134, -1.1749999523162842),
        (0.17500004172325134, -0.9249999523162842),
        (0.025000037625432014, -0.875),
        # (-0.07499996572732925, -1.125),
    ],
    [
        (-0.07499996572732925, 0.025000037625432014),
        (0.17500004172325134, -0.07499996572732925),
        (0.2250000387430191, 0.12500004470348358),
        (0.025000037625432014, 0.2250000387430191),
        # (-0.07499996572732925, 0.025000037625432014),
    ],
    [
        (-0.07499996572732925, 1.0750000476837158),
        (0.17500004172325134, 0.9750000238418579),
        (0.27500003576278687, 1.1750000715255737),
        (0.025000037625432014, 1.2750000953674316),
        # (-0.07499996572732925, 1.0750000476837158),
    ],
    [
        (0.2250000387430191, 1.225000023841858),
        # (0.2250000387430191, 1.225000023841858),
    ],
    [
        (0.27500003576278687, -2.5250000953674316),
        (0.6250000596046448, -2.4749999046325684),
        # (0.27500003576278687, -2.5250000953674316),
    ],
]

polygon_obstacles = [
    StaticObstacle(id=i, geometry=Polygon(vertices=vertices))
    for i, vertices in enumerate(polygons)
]

dynamic_obstacle = SimulatedDynamicObstacle(
    id=4,
    position=(13, 12.5),
    orientation=np.deg2rad(-90),
    goal_position=(8, -12.5),
    goal_orientation=np.deg2rad(90),
    horizon=30,
)

environment = LocalEnvironment(
    agent=agent,
    static_obstacles=[
        # static_obstacle_rectangle,
        # static_obstacle_rectangle_2,
        # static_obstacle_ellipse,
        # static_obstacle_circle,
        # static_obstacle_circle_2,
        # static_obstacle_circle_3,
        # static_polygonal_obstacle
        *polygon_obstacles
    ],
    dynamic_obstacles=[
        # dynamic_obstacle,
    ],
    waypoints=[(-2.47, 2.41, np.deg2rad(90)), (-5.47, 1.41, np.deg2rad(90))],
    # save_video=True,
)
environment.loop()
