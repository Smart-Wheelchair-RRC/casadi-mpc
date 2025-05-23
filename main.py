import numpy as np

from mpc.agent import EgoAgent
from mpc.dynamic_obstacle import SimulatedDynamicObstacle
from mpc.environment import LocalEnvironment
from mpc.geometry import Circle
from mpc.obstacle import StaticObstacle

agent = EgoAgent(
    id=1,
    radius=1,
    initial_position=(-16, -16),
    initial_orientation=np.deg2rad(90),
    horizon=10,
    use_warm_start=True,
    planning_time_step=0.8,
    linear_velocity_bounds=(0, 0.3),
    angular_velocity_bounds=(-0.5, 0.5),
    linear_acceleration_bounds=(-0.5, 0.5),
    angular_acceleration_bounds=(-1, 1),
    sensor_radius=5,
)

static_obstacle_radius = 0.2
walls = [
    *Circle.create_circle_from_line(
        (-20, -20), (20, -20), radius=static_obstacle_radius
    ),
    *Circle.create_circle_from_line((-20, 20), (20, 20), radius=static_obstacle_radius),
    *Circle.create_circle_from_line((20, -20), (20, 20), radius=static_obstacle_radius),
    *Circle.create_circle_from_line(
        (-20, -20), (-20, 20), radius=static_obstacle_radius
    ),
    *Circle.create_circle_from_line((1, -20), (1, 8), radius=static_obstacle_radius),
]

obstacle_geometries = walls
static_obstacles = [
    StaticObstacle(id=i, geometry=geometry)
    for i, geometry in enumerate(obstacle_geometries)
]

dynamic_obstacle = SimulatedDynamicObstacle(
    id=1,
    position=(-4, -2),
    orientation=np.deg2rad(-90),
    goal_position=(-4, -10),
    goal_orientation=np.deg2rad(-90),
    horizon=10,
)

dynamic_obstacles = [
    dynamic_obstacle,
]

# All static obstacles must have same radius
assert all(
    isinstance(obstacle.geometry, Circle)
    and obstacle.geometry.radius == obstacle_geometries[0].radius
    for obstacle in static_obstacles
), "All static obstacles must have the same radius"

# All dynamic obstacles must have same radius
assert all(
    isinstance(obstacle.geometry, Circle)
    and obstacle.geometry.radius == dynamic_obstacles[0].geometry.radius
    for obstacle in dynamic_obstacles
), "All dynamic obstacles must have the same radius"


environment = LocalEnvironment(
    agent=agent,
    static_obstacles=static_obstacles,
    dynamic_obstacles=dynamic_obstacles,
    waypoints=[
        (-2, -2, np.deg2rad(90)),
        (-2, 10, np.deg2rad(90)),
        (10, 5, np.deg2rad(90)),
    ],
    # save_video=True,
)
environment.view_environment()
environment.loop()
