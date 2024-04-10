import time
from pathlib import Path
from typing import List

import numpy as np

from mpc.agent import EgoAgent
from mpc.obstacle import DynamicObstacle, StaticObstacle
from mpc.plotter import Plotter


class Environment:
    def __init__(
        self,
        agent: EgoAgent,
        static_obstacles: List[StaticObstacle],
        dynamic_obstacles: List[DynamicObstacle],
    ):
        self.agent = agent
        self.static_obstacles = static_obstacles
        self.dynamic_obstacles = dynamic_obstacles
        self.rollout_times = []

    @property
    def obstacles(self):
        return self.static_obstacles + self.dynamic_obstacles

    def step(self):
        step_start = time.perf_counter()
        self.agent.step(
            obstacles=[
                obstacle
                for obstacle in self.obstacles
                if obstacle.calculate_distance(self.agent.state)
                <= self.agent.sensor_radius
            ]
        )
        self.rollout_times.append(time.perf_counter() - step_start)

        for obstacle in self.dynamic_obstacles:
            obstacle.step()

    def reset(self):
        self.agent.reset()
        self.rollout_times = []

        for obstacle in self.dynamic_obstacles:
            obstacle.reset()


class LocalEnvironment(Environment):
    def __init__(
        self,
        agent: EgoAgent,
        static_obstacles: List[StaticObstacle],
        dynamic_obstacles: List[DynamicObstacle],
        plot: bool = True,
        results_path: str = "results",
        save_video: bool = False,
    ):
        super().__init__(agent, static_obstacles, dynamic_obstacles)
        self.plot = plot
        self.results_path = Path(results_path)

        assert (
            plot is True if save_video else True
        ), "Cannot save video without plotting"

        self.save_video = save_video

    def loop(self, max_timesteps: int = 10000):
        self.reset()

        if self.plot:
            plotter = Plotter(
                agent=self.agent,
                static_obstacles=self.static_obstacles,
                dynamic_obstacles=self.dynamic_obstacles,
                video_path=self.results_path if self.save_video else None,
            )

        goal_list = [
            # (5, 17),
            # np.array([5, 20, np.pi / 2]),
            # np.array([3, 12.5, np.pi / 2]),
            # np.array([1, 14, np.deg2rad(110)]),
            # np.array([-2.5, 17, np.pi / 2]),
            # # np.array([-1, 21, np.pi / 2]),
            # np.array([5, 50, np.deg2rad(90)]),
        ]

        while (not self.agent.at_goal) and max_timesteps > 0:
            # print(self.agent.goal_state)
            self.step()

            if self.plot:
                plotter.update_plot()

            if (self.agent.at_goal) and len(goal_list) > 0:
                self.agent.goal_state = goal_list.pop(0)
                if self.plot:
                    plotter.update_goal()

            max_timesteps -= 1

            # print(
            #     "Current average rollout time:",
            #     self.total_rollout_time / self.total_rollout_steps,
            # )
            print(
                f"Step {len(self.rollout_times)}, Time: {self.rollout_times[-1] * 1000:.2f} ms"
            )

        time_array = np.array(self.rollout_times)
        # Print metrics excluding first rollout
        print(f"Average rollout time: {time_array[1:].mean() * 1000:.2f} ms")

        if self.plot:
            plotter.close()

            if self.save_video:
                plotter.collapse_frames_to_video()


class ROSEnvironment(Environment):
    def __init__(
        self,
        agent: EgoAgent,
        static_obstacles: List[StaticObstacle],
        dynamic_obstacles: List[DynamicObstacle],
    ):
        super().__init__(agent, static_obstacles, dynamic_obstacles)
