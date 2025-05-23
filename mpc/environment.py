import time
from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple

import numpy as np

if TYPE_CHECKING:
    from mpc.agent import EgoAgent
    from mpc.dynamic_obstacle import DynamicObstacle, SimulatedDynamicObstacle
    from mpc.obstacle import StaticObstacle

from mpc.plotter import Plotter


class Environment:
    def __init__(
        self,
        agent: "EgoAgent",
        static_obstacles: List["StaticObstacle"],
        dynamic_obstacles: List["SimulatedDynamicObstacle"],
        waypoints: List[Tuple[Tuple, float]],
    ):
        self.agent = agent
        self.static_obstacles = static_obstacles
        self.dynamic_obstacles = dynamic_obstacles

        for obstacle in self.dynamic_obstacles:
            assert obstacle.horizon == agent.horizon, (
                "Dynamic obstacle horizon must match agent horizon"
            )

        self.waypoints = waypoints
        self.waypoint_index = 0
        self.agent.update_goal(self.current_waypoint)

        self.rollout_times = []

    @property
    def current_waypoint(self):
        return (
            self.waypoints[self.waypoint_index]
            if self.waypoint_index < len(self.waypoints)
            else None
        )

    @property
    def final_goal_reached(self):
        return self.waypoint_index == len(self.waypoints) - 1 and self.agent.at_goal

    @property
    def obstacles(self):
        return self.static_obstacles + self.dynamic_obstacles

    def step(self):
        step_start = time.perf_counter()

        filtered_static_obstacles = [
            obstacle
            for obstacle in self.static_obstacles
            if obstacle.calculate_distance(self.agent.state) <= self.agent.sensor_radius
        ]

        filtered_dynamic_obstacles = [
            obstacle
            for obstacle in self.dynamic_obstacles
            if obstacle.calculate_distance(self.agent.state) <= self.agent.sensor_radius
        ]

        filtered_obstacles = filtered_static_obstacles + filtered_dynamic_obstacles
        print("Number of Obstacles:", len(filtered_obstacles))

        self.agent.step(
            static_obstacles=filtered_static_obstacles,
            dynamic_obstacles=filtered_dynamic_obstacles,
        )

        self.rollout_times.append(time.perf_counter() - step_start)

        for obstacle in self.dynamic_obstacles:
            obstacle.step()

        if self.agent.at_goal and not self.final_goal_reached:
            print("Reached waypoint", self.waypoint_index + 1)
            self.waypoint_index += 1
            self.agent.update_goal(self.current_waypoint)

    def reset(self):
        self.agent.reset()
        self.rollout_times = []
        self.waypoint_index = 0
        self.agent.update_goal(self.current_waypoint)

        for obstacle in self.dynamic_obstacles:
            obstacle.reset()


class LocalEnvironment(Environment):
    def __init__(
        self,
        agent: "EgoAgent",
        static_obstacles: List["StaticObstacle"],
        dynamic_obstacles: List["SimulatedDynamicObstacle"],
        waypoints: List[Tuple[Tuple, float]],
        plot: bool = True,
        results_path: str = "results",
        save_video: bool = False,
    ):
        super().__init__(agent, static_obstacles, dynamic_obstacles, waypoints)
        self.plot = plot
        self.results_path = Path(results_path)
        assert plot is True if save_video else True, (
            "Cannot save video without plotting"
        )
        self.save_video = save_video
        if self.plot:
            self.plotter = Plotter(
                agent=self.agent,
                static_obstacles=self.static_obstacles,
                dynamic_obstacles=self.dynamic_obstacles,
                video_path=self.results_path if self.save_video else None,
            )

    def loop(self, max_timesteps: int = 10000):
        self.reset()

        while (not self.final_goal_reached) and max_timesteps > 0:
            self.step()
            if self.plot:
                self.plotter.update_plot(self.waypoints)
            max_timesteps -= 1
            print(
                f"Step {len(self.rollout_times)}, Time: {self.rollout_times[-1] * 1000:.2f} ms"
            )
        time_array = np.array(self.rollout_times)
        # Print metrics excluding first rollout
        print(f"Average rollout time: {time_array[1:].mean() * 1000:.2f} ms")
        if self.plot:
            self.plotter.close()
            if self.save_video:
                self.plotter.collapse_frames_to_video()

    def view_environment(self):
        assert self.plot, "Cannot view environment without plotting"
        self.plotter.update_plot(self.waypoints)
        self.plotter.reset_plot_limits()
        self.plotter.freeze_plot()


class ROSEnvironment(Environment):
    def __init__(
        self,
        agent: "EgoAgent",
        static_obstacles: List["StaticObstacle"],
        dynamic_obstacles: List["DynamicObstacle"],
        waypoints: List[Tuple[Tuple, float]],
        plot: bool = True,
        results_path: str = "results",
        save_video: bool = False,
    ):
        super().__init__(agent, static_obstacles, dynamic_obstacles, waypoints)
        self.plot = plot
        self.results_path = Path(results_path)
        assert plot is True if save_video else True, (
            "Cannot save video without plotting"
        )
        self.save_video = save_video

        if self.plot:
            self.plotter = Plotter(
                agent=self.agent,
                static_obstacles=self.static_obstacles,
                dynamic_obstacles=self.dynamic_obstacles,
                video_path=self.results_path if self.save_video else None,
            )

    def step(self):
        if self.waypoint_index == len(self.waypoints) - 1:
            print("Heading for final goal")
            self.agent.goal_radius = 0.1
            # self.agent.planner.update_orientation_weight(100)
        else:
            self.agent.goal_radius = 0.5
            # self.agent.planner.update_orientation_weight(0)

        # if self.final_goal_reached:
        #     print("Final Goal Reached")
        #     self.agent.planner.update_orientation_weight(100)
        # else:
        #     self.agent.planner.update_orientation_weight(0)

        t1 = time.perf_counter()
        static_obstacles_dict = {
            obstacle.calculate_distance(self.agent.state): obstacle
            for obstacle in self.static_obstacles
        }
        filtered_static_obstacles = [
            static_obstacles_dict[distance]
            for distance in sorted(static_obstacles_dict.keys())
            if distance <= self.agent.sensor_radius
        ]
        dynamic_obstacles_dict = {
            obstacle.calculate_distance(self.agent.state): obstacle
            for obstacle in self.dynamic_obstacles
        }
        filtered_dynamic_obstacles = [
            dynamic_obstacles_dict[distance]
            for distance in sorted(dynamic_obstacles_dict.keys())
            if distance <= self.agent.sensor_radius
        ]
        print("Number of Dyn Obstacles:", len(filtered_dynamic_obstacles))

        self.agent.step(
            static_obstacles=filtered_static_obstacles[:4],
            dynamic_obstacles=filtered_dynamic_obstacles,
        )

        t2 = time.perf_counter()
        print("Rollout Time:", t2 - t1)

        if self.plot:
            self.plotter.update_plot(self.waypoints)
            self.plotter.update_static_obstacles(filtered_static_obstacles)
            # self.plotter.update_static_obstacles(filtered_dynamic_obstacles)

        print("Current Waypoint", self.current_waypoint)
        print("Waypoints", self.waypoints)

        if self.agent.at_goal and not self.final_goal_reached:
            print("Reached waypoint", self.waypoint_index + 1)
            self.waypoint_index += 1
            self.agent.update_goal(self.current_waypoint)

    def reset(self):
        self.agent.reset()
        self.waypoint_index = 0
        self.agent.update_goal(self.current_waypoint)
