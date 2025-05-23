import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

# import numpy as np
from matplotlib import pyplot as plt

if TYPE_CHECKING:
    from mpc.agent import EgoAgent
    from mpc.dynamic_obstacle import SimulatedDynamicObstacle
    from mpc.obstacle import StaticObstacle


class Plotter:
    def __init__(
        self,
        agent: "EgoAgent",
        static_obstacles: List["StaticObstacle"],
        dynamic_obstacles: List["SimulatedDynamicObstacle"],
        video_path: Optional[Path] = None,
    ):
        self.agent = agent
        self.static_obstacles = static_obstacles
        self.dynamic_obstacles = dynamic_obstacles

        self.video_path = video_path

        if self.video_path:
            os.makedirs(self.video_path, exist_ok=True)

        self.num_frames = 0

        self.PLOT_SIZE_DELTA = 10

        # Create the plot
        _, axes = plt.subplots()
        axes: plt.Axes

        axes.set_aspect("equal")

        self.agent_patch = axes.add_patch(self.agent.geometry.create_patch())

        # add agent id to plot
        self.agent_id = axes.text(
            self.agent.state[0],
            self.agent.state[1],
            f"Agent ({self.agent.state[0]:.2f}, {self.agent.state[1]:.2f})",
            fontsize=12,
            color="black",
        )

        self.obstacle_patches = []
        self.obstacle_plots = []

        for obstacle in self.dynamic_obstacles:
            self.obstacle_patches.append(
                axes.add_patch(obstacle.geometry.create_patch())
            )

        for obstacle in self.static_obstacles:
            axes.add_patch(obstacle.geometry.create_patch())

        self.static_obstacle_ids = [
            # axes.text(
            #     obstacle.state[0],
            #     obstacle.state[1],
            #     f"Obstacle {obstacle.id}",
            #     fontsize=12,
            #     color="black",
            # )
            # for obstacle in self.static_obstacles
        ]

        self.dynamic_obstacle_ids = [
            axes.text(
                obstacle.state[0],
                obstacle.state[1],
                f"Obstacle {obstacle.id}",
                fontsize=12,
                color="black",
            )
            for obstacle in self.dynamic_obstacles
        ]

        self.goal_plot = axes.plot(
            self.agent.goal_state[0], self.agent.goal_state[1], marker="x", color="r"
        )[0]

        # self.final_goal_plot = axes.plot(
        #     self.agent.goal_state[0], self.agent.goal_state[1], marker="x", color="g"
        # )[0]

        self.states_plot = axes.plot(
            self.agent.states_matrix[0, 1:],
            self.agent.states_matrix[1, 1:],
            marker=".",
            color="blue",
            # s=1.5,
        )[0]

        # plot velocity arrows for dynamic obstacles
        # self.dynamic_obstacle_plots = [
        #     axes.arrow(
        #         obstacle.state[0],
        #         obstacle.state[1],
        #         obstacle.linear_velocity * np.cos(obstacle.state[2]),
        #         obstacle.linear_velocity * np.sin(obstacle.state[2]),
        #         head_width=0.2,
        #         head_length=0.2,
        #         fc="g",
        #         ec="g",
        #     )
        #     for obstacle in self.dynamic_obstacles
        # ]

        self.dynamic_obstacle_plots = [
            axes.plot(
                obstacle.states_matrix[0, 1:],
                obstacle.states_matrix[1, 1:],
                marker=".",
                color="green",
                # s=1.5,
            )[0]
            for obstacle in self.dynamic_obstacles
        ]

        self.goal_plot.set_data([self.agent.goal_state[0]], [self.agent.goal_state[1]])
        # self.final_goal_plot.set_data(waypoints[-1][0], waypoints[-1][1])

        self.goal_plot_text = axes.text(
            self.agent.goal_state[0],
            self.agent.goal_state[1],
            f"({self.agent.goal_state[0]:.2f}, {self.agent.goal_state[1]:.2f})",
            fontsize=12,
            color="black",
        )
        # self.final_goal_plot_text = axes.text(
        #     self.agent.goal_state[0],
        #     self.agent.goal_state[1],
        #     f"({self.agent.goal_state[0]:.2f}, {self.agent.goal_state[1]:.2f})",
        #     fontsize=12,
        #     color="black",
        # )
        # self.recenter_plot()

    def reset_plot_limits(self):
        # Remove limits
        axes = plt.gca()
        axes.autoscale()

    def freeze_plot(self):
        plt.show()

    def update_goal(self, waypoints):
        self.goal_plot.set_data([self.agent.goal_state[0]], [self.agent.goal_state[1]])
        self.goal_plot_text.set_position(
            (self.agent.goal_state[0], self.agent.goal_state[1])
        )
        self.goal_plot_text.set_text(
            f"({self.agent.goal_state[0]:.2f}, {self.agent.goal_state[1]:.2f})"
        )
        # if len(waypoints) > 0:
        #     self.final_goal_plot.set_data(waypoints[-1][0], waypoints[-1][1])
        #     self.final_goal_plot_text.set_position((waypoints[-1][0], waypoints[-1][1]))

    @property
    def obstacles(self):
        return self.static_obstacles + self.dynamic_obstacles

    def recenter_plot(self):
        # Center plot to agent
        axes = plt.gca()
        axes.set_aspect("equal")
        axes.set_xlim(
            self.agent.state[0] - self.PLOT_SIZE_DELTA,
            self.agent.state[0] + self.PLOT_SIZE_DELTA,
        )
        axes.set_ylim(
            self.agent.state[1] - self.PLOT_SIZE_DELTA,
            self.agent.state[1] + self.PLOT_SIZE_DELTA,
        )

    def save_frame(self):
        # Save frame to video
        plt.gcf().savefig(self.video_path / f"frame_{(self.num_frames + 1):04d}.png")

    def update_plot(self, waypoint):
        # Plot using matplotlib
        plt.pause(0.01)

        self.recenter_plot()

        self.agent.geometry.update_patch(self.agent_patch)

        self.agent_id.set_position((self.agent.state[0], self.agent.state[1]))
        self.agent_id.set_text(
            f"Agent ({self.agent.state[0]:.2f}, {self.agent.state[1]:.2f})"
        )

        for index, obstacle in enumerate(self.dynamic_obstacles):
            self.dynamic_obstacle_ids[index].set_position(
                (obstacle.state[0], obstacle.state[1])
            )
            obstacle.geometry.update_patch(self.obstacle_patches[index])

        self.states_plot.set_data(
            self.agent.states_matrix[0, 1:], self.agent.states_matrix[1, 1:]
        )

        for obstacle_plot, obstacle in zip(
            self.dynamic_obstacle_plots, self.dynamic_obstacles
        ):
            # obstacle_plot.set_data(
            #     x=obstacle.state[0],
            #     y=obstacle.state[1],
            #     dx=obstacle.linear_velocity * np.cos(obstacle.state[2]),
            #     dy=obstacle.linear_velocity * np.sin(obstacle.state[2]),
            # )
            obstacle_plot.set_data(
                obstacle.states_matrix[0, 1:],
                obstacle.states_matrix[1, 1:],
            )

        self.update_goal(waypoint)

        if self.video_path:
            self.save_frame()

        self.num_frames += 1

    def collapse_frames_to_video(self):
        frame_pattern = f"{self.video_path}/frame_%04d.png"
        video_path = f"{self.video_path}/video.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-framerate",
                "20",
                "-i",
                frame_pattern,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                video_path,
            ]
        )
        # Delete frames
        for file in os.listdir(self.video_path):
            if file.endswith(".png"):
                os.remove(os.path.join(self.video_path, file))

    def update_static_obstacles(self, static_obstacles):
        self.static_obstacles = static_obstacles
        # Remove old patches
        for patch in self.obstacle_patches:
            patch.remove()

        for plot in self.obstacle_plots:
            plot.remove()

        self.obstacle_patches = []
        self.obstacle_plots = []

        for obstacle in self.static_obstacles:
            self.obstacle_patches.append(
                plt.gca().add_patch(obstacle.geometry.create_patch())
            )
            self.obstacle_plots.append(
                plt.gca().plot(
                    obstacle.states_matrix[0, 1:],
                    obstacle.states_matrix[1, 1:],
                    marker=".",
                    color="green",
                    # s=1.5,
                )[0]
            )

        # for index, obstacle in enumerate(self.static_obstacles):
        # self.static_obstacle_ids[index].set_position(
        #     (obstacle.state[0], obstacle.state[1])
        # )
        # obstacle.geometry.update_patch(self.obstacle_patches[index])

    def close(self):
        plt.pause(2)
        plt.close()
