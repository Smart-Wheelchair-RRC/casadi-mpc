# import for type hinting
from typing import TYPE_CHECKING, List, Optional, Tuple, Union, cast

import casadi as ca
import numpy as np

if TYPE_CHECKING:
    from mpc.dynamic_obstacle import DynamicObstacle, SimulatedDynamicObstacle
    from mpc.obstacle import StaticObstacle
from mpc.geometry import Circle


def MX_horzcat(*args: ca.MX) -> ca.MX:
    return ca.horzcat(*args)


def SX_horzcat(*args: ca.SX) -> ca.SX:
    return ca.horzcat(*args)


def DM_horzcat(*args: ca.DM) -> ca.DM:
    return ca.horzcat(*args)


def MX_vertcat(*args: ca.MX) -> ca.MX:
    return ca.vertcat(*args)


def SX_vertcat(*args: ca.SX) -> ca.SX:
    return ca.vertcat(*args)


def DM_vertcat(*args: ca.DM) -> ca.DM:
    return ca.vertcat(*args)


def create_symbolic_scalar(name: str, size: Optional[Tuple[int, int]] = None) -> ca.SX:
    return ca.SX.sym(name, size) if size else ca.SX.sym(name)


def create_symbolic_matrix(name: str, size: Optional[Tuple[int, int]] = None) -> ca.MX:
    return ca.MX.sym(name, size) if size else ca.MX.sym(name)


class MotionPlanner:
    def __init__(
        self,
        time_step: float,
        horizon: int,
    ):
        self.time_step = time_step
        self.horizon = horizon

        # States
        self.symbolic_states = SX_vertcat(
            create_symbolic_scalar("x"),
            create_symbolic_scalar("y"),
            create_symbolic_scalar("theta"),
        )
        self.num_states = cast(int, self.symbolic_states.numel())

        # Controls
        self.symbolic_controls = SX_vertcat(
            create_symbolic_scalar("a"),
            create_symbolic_scalar("alpha"),
        )
        self.num_controls = cast(int, self.symbolic_controls.numel())

        # Weight matrix for goal cost
        self.weight_matrix = ca.DM(ca.diagcat(100, 100, 0))

        self.angular_acceleration_weight = ca.DM(30)
        self.linear_acceleration_weight = ca.DM(50)
        # Obstacle cost weight
        # self.obstacle_cost_weight = ca.DM(10000)

        # Matrix of states over the prediction horizon
        # (contains an extra column for the initial state)
        self.symbolic_states_matrix = create_symbolic_matrix(
            "X", (self.num_states, self.horizon + 1)
        )

        # Matrix of controls over the prediction horizon
        self.symbolic_controls_matrix = create_symbolic_matrix(
            "U", (self.num_controls, self.horizon)
        )

        # Initial state and Goal state vector
        self.symbolic_terminal_states_vector = create_symbolic_matrix(
            "P", (self.num_states + self.num_states, 1)
        )

        # Optimization variables
        self.symbolic_optimization_variables = MX_vertcat(
            self.symbolic_states_matrix.reshape((-1, 1)),
            self.symbolic_controls_matrix.reshape((-1, 1)),
        )

    def update_orientation_weight(self, orientation_weight: float):
        self.weight_matrix = ca.DM(ca.diagcat(100, 100, orientation_weight))

    def _get_symbolic_goal_cost(self) -> ca.MX:
        error = cast(
            ca.MX,
            self.symbolic_states_matrix[:, 1:-1]
            - self.symbolic_terminal_states_vector[self.num_states :],
        )
        cost = cast(
            ca.MX,
            ca.sum2(cast(ca.MX, cast(ca.MX, (error.T @ self.weight_matrix)) * error.T)),
        )
        return cast(
            ca.MX,
            ca.sum1(cost),
        )

    def _get_symbolic_angular_acceleration_cost(self) -> ca.MX:
        squared_angular_acceleration = cast(
            ca.MX, self.symbolic_controls_matrix[1, :] ** 2
        )
        return (
            cast(
                ca.MX,
                ca.sum1(ca.sum2(squared_angular_acceleration)),
            )
            * self.angular_acceleration_weight
        )

    def _get_symbolic_linear_acceleration_cost(self) -> ca.MX:
        squared_linear_acceleration = cast(
            ca.MX, self.symbolic_controls_matrix[0, :] ** 2
        )
        return (
            cast(
                ca.MX,
                ca.sum1(ca.sum2(squared_linear_acceleration)),
            )
            * self.linear_acceleration_weight
        )

    def _get_symbolic_costs(
        self,
        # visible_obstacles: List[Obstacle],
        # inflation_radius: float,
    ) -> ca.MX:
        return (
            self._get_symbolic_goal_cost()
            + self._get_symbolic_angular_acceleration_cost()
            + self._get_symbolic_linear_acceleration_cost()
            # + self._get_symbolic_obstacle_cost(visible_obstacles, inflation_radius)
        )

    def _get_lane_bounds(
        self, left_right_lane_bounds: Tuple[float, float]
    ) -> Tuple[ca.DM, ca.DM]:
        lane_lower_bounds = cast(
            ca.DM,
            ca.repmat(
                DM_vertcat(
                    left_right_lane_bounds[0],
                    -ca.inf,
                    -ca.inf,
                ),
                (1, self.horizon + 1),
            ),
        )
        lane_upper_bounds = cast(
            ca.DM,
            ca.repmat(
                DM_vertcat(
                    left_right_lane_bounds[1],
                    ca.inf,
                    ca.inf,
                ),
                (1, self.horizon + 1),
            ),
        )

        return lane_lower_bounds, lane_upper_bounds

    def _get_control_bounds(
        self,
        linear_acceleration_bounds: Tuple[float, float],
        angular_acceleration_bounds: Tuple[float, float],
    ) -> Tuple[ca.DM, ca.DM]:
        control_lower_bounds = cast(
            ca.DM,
            ca.repmat(
                DM_vertcat(
                    linear_acceleration_bounds[0],
                    angular_acceleration_bounds[0],
                ),
                (1, self.horizon),
            ),
        )
        control_upper_bounds = cast(
            ca.DM,
            ca.repmat(
                DM_vertcat(
                    linear_acceleration_bounds[1],
                    angular_acceleration_bounds[1],
                ),
                (1, self.horizon),
            ),
        )

        return control_lower_bounds, control_upper_bounds

    def _get_optimization_variable_bounds(
        self,
        left_right_lane_bounds: Tuple[float, float],
        linear_acceleration_bounds: Tuple[float, float],
        angular_acceleration_bounds: Tuple[float, float],
    ) -> Tuple[ca.DM, ca.DM]:
        lower_lane_bounds, upper_lane_bounds = self._get_lane_bounds(
            left_right_lane_bounds
        )
        lower_control_bounds, upper_control_bounds = self._get_control_bounds(
            linear_acceleration_bounds=linear_acceleration_bounds,
            angular_acceleration_bounds=angular_acceleration_bounds,
        )

        optimization_variable_lower_bounds = DM_vertcat(
            lower_lane_bounds.reshape((-1, 1)), lower_control_bounds.reshape((-1, 1))
        )
        optimization_variable_upper_bounds = DM_vertcat(
            upper_lane_bounds.reshape((-1, 1)), upper_control_bounds.reshape((-1, 1))
        )

        return optimization_variable_lower_bounds, optimization_variable_upper_bounds

    def _get_state_constraints_bounds(self) -> Tuple[ca.DM, ca.DM]:
        return (
            cast(ca.DM, ca.DM.zeros((3, self.horizon + 1))),
            cast(ca.DM, ca.DM.zeros((3, self.horizon + 1))),
        )

    def _get_symbolic_states_constraints(
        self, current_linear_velocity: float, current_angular_velocity: float
    ) -> ca.MX:
        current_velocities = cast(
            ca.MX,
            self.symbolic_controls_matrix * ca.DM(self.time_step)
            + DM_horzcat(
                DM_vertcat(
                    ca.DM(current_linear_velocity),
                    ca.DM(current_angular_velocity),
                ),
                ca.MX.zeros((2, self.horizon - 1)),
            ),
        )
        current_velocities = cast(
            ca.MX,
            ca.cumsum(
                current_velocities,
                1,
            ),
        )
        next_states_bounds = MX_vertcat(
            self.symbolic_states_matrix[0, :-1]
            + (
                current_velocities[0, :]
                * ca.cos(self.symbolic_states_matrix[2, :-1])
                * ca.DM(self.time_step)
            ),
            self.symbolic_states_matrix[1, :-1]
            + (
                current_velocities[0, :]
                * ca.sin(self.symbolic_states_matrix[2, :-1])
                * ca.DM(self.time_step)
            ),
            self.symbolic_states_matrix[2, :-1]
            + (current_velocities[1, :] * ca.DM(self.time_step)),
        )
        return MX_horzcat(
            self.symbolic_states_matrix[:, 0]
            - self.symbolic_terminal_states_vector[: self.num_states],
            self.symbolic_states_matrix[:, 1:] - next_states_bounds,
        )

    def _get_symbolic_obstacle_constraints(
        self,
        static_obstacles: List["StaticObstacle"],
        dynamic_obstacles: List[Union["DynamicObstacle", "SimulatedDynamicObstacle"]],
    ) -> ca.MX:
        all_obstacles = static_obstacles + dynamic_obstacles

        if (
            not all(
                isinstance(all_obstacles[i].geometry, Circle)
                for i in range(len(all_obstacles))
            )
            or len(all_obstacles) == 0
        ):
            return MX_horzcat(
                *[
                    obstacle.calculate_symbolic_matrix_distance(
                        symbolic_states_matrix=self.symbolic_states_matrix[:, 1:]
                    )
                    for obstacle in all_obstacles
                ]
            )

        num_obstacles = len(all_obstacles)

        centers = DM_horzcat(
            *[all_obstacles[i].geometry.center for i in range(len(all_obstacles))]
        )

        x_differences = cast(
            ca.MX,
            ca.repmat(centers[0, :].T, 1, self.horizon)
            - ca.repmat(self.symbolic_states_matrix[0, 1:], num_obstacles, 1),
        )

        y_differences = cast(
            ca.MX,
            ca.repmat(centers[1, :].T, 1, self.horizon)
            - ca.repmat(self.symbolic_states_matrix[1, 1:], num_obstacles, 1),
        )

        distances = cast(
            ca.MX,
            ca.sqrt(x_differences**2 + y_differences**2),
        ).T

        if len(static_obstacles) > 0:
            static_radius_vector = ca.repmat(
                ca.DM(static_obstacles[0].geometry.radius),
                (self.horizon, len(static_obstacles)),
            )
        else:
            static_radius_vector = ca.DM.zeros((self.horizon, 0))

        if len(dynamic_obstacles) > 0:
            dynamic_radius_vector = ca.repmat(
                ca.DM(dynamic_obstacles[0].geometry.radius),
                (self.horizon, len(dynamic_obstacles)),
            )
        else:
            dynamic_radius_vector = ca.DM.zeros((self.horizon, 0))

        distances = distances - ca.horzcat(
            static_radius_vector,
            dynamic_radius_vector,
        )

        return distances

    def _get_obstacle_constraints_bounds(
        self, inflation_radius: float, num_obstacles: int
    ) -> Tuple[ca.DM, ca.DM]:
        constraints_lower_bound = cast(
            ca.DM,
            ca.repmat(
                ca.DM(inflation_radius),
                (1, self.horizon * num_obstacles),
            ),
        )
        constraints_upper_bound = cast(
            ca.DM,
            ca.repmat(
                ca.DM(ca.inf),
                (1, self.horizon * num_obstacles),
            ),
        )
        return constraints_lower_bound, constraints_upper_bound

    def _get_symbolic_linear_velocity_constraints(
        self, current_linear_velocity: float
    ) -> ca.MX:
        current_velocities = cast(
            ca.MX,
            self.symbolic_controls_matrix[0, :] * ca.DM(self.time_step)
            + ca.horzcat(
                ca.DM(current_linear_velocity),
                ca.MX.zeros((1, self.horizon - 1)),
            ),
        )
        return cast(
            ca.MX,
            ca.cumsum(
                current_velocities,
                1,
            ),
        )

    def _get_linear_velocity_constraints_bounds(
        self, linear_velocity_bounds: Tuple[float, float]
    ) -> Tuple[ca.DM, ca.DM]:
        constraints_lower_bound = cast(
            ca.DM,
            ca.repmat(ca.DM(linear_velocity_bounds[0]), (1, self.horizon)),
        )
        constraints_upper_bound = cast(
            ca.DM,
            ca.repmat(ca.DM(linear_velocity_bounds[1]), (1, self.horizon)),
        )

        return constraints_lower_bound, constraints_upper_bound

    def _get_symbolic_angular_velocity_constraints(
        self, current_angular_velocity: float
    ) -> ca.MX:
        current_velocities = cast(
            ca.MX,
            self.symbolic_controls_matrix[1, :] * ca.DM(self.time_step)
            + ca.horzcat(
                ca.DM(current_angular_velocity),
                ca.MX.zeros((1, self.horizon - 1)),
            ),
        )
        return cast(
            ca.MX,
            ca.cumsum(
                current_velocities,
                1,
            ),
        )

    def _get_angular_velocity_constraints_bounds(
        self, angular_velocity_bounds: Tuple[float, float]
    ) -> Tuple[ca.DM, ca.DM]:
        constraints_lower_bound = cast(
            ca.DM,
            ca.repmat(ca.DM(angular_velocity_bounds[0]), (1, self.horizon)),
        )
        constraints_upper_bound = cast(
            ca.DM,
            ca.repmat(ca.DM(angular_velocity_bounds[1]), (1, self.horizon)),
        )

        return constraints_lower_bound, constraints_upper_bound

    def _get_symbolic_constraints(
        self,
        current_linear_velocity: float,
        current_angular_velocity: float,
        static_obstacles: List["StaticObstacle"],
        dynamic_obstacles: List[Union["DynamicObstacle", "SimulatedDynamicObstacle"]],
    ) -> ca.MX:
        symbolic_constraints = MX_vertcat(
            self._get_symbolic_states_constraints(
                current_linear_velocity=current_linear_velocity,
                current_angular_velocity=current_angular_velocity,
            ).reshape((-1, 1)),
            self._get_symbolic_linear_velocity_constraints(
                current_linear_velocity
            ).reshape((-1, 1)),
            self._get_symbolic_angular_velocity_constraints(
                current_angular_velocity
            ).reshape((-1, 1)),
        )

        if len(static_obstacles + dynamic_obstacles) > 0:
            symbolic_constraints = MX_vertcat(
                symbolic_constraints,
                self._get_symbolic_obstacle_constraints(
                    static_obstacles=static_obstacles,
                    dynamic_obstacles=dynamic_obstacles,
                ).reshape((-1, 1)),
            )

        return symbolic_constraints

    def _get_constraints_bounds(
        self,
        linear_velocity_bounds: Tuple[float, float],
        angular_velocity_bounds: Tuple[float, float],
        inflation_radius: float = 0,
        num_obstacles: int = 0,
    ) -> Tuple[ca.DM, ca.DM]:
        (
            state_constraints_lower_bound,
            state_constraints_upper_bound,
        ) = self._get_state_constraints_bounds()

        (
            linear_velocity_constraints_lower_bound,
            linear_velocity_constraints_upper_bound,
        ) = self._get_linear_velocity_constraints_bounds(linear_velocity_bounds)

        (
            angular_velocity_constraints_lower_bound,
            angular_velocity_constraints_upper_bound,
        ) = self._get_angular_velocity_constraints_bounds(angular_velocity_bounds)

        constraints_lower_bound = DM_vertcat(
            state_constraints_lower_bound.reshape((-1, 1)),
            linear_velocity_constraints_lower_bound.reshape((-1, 1)),
            angular_velocity_constraints_lower_bound.reshape((-1, 1)),
        )
        constraints_upper_bound = DM_vertcat(
            state_constraints_upper_bound.reshape((-1, 1)),
            linear_velocity_constraints_upper_bound.reshape((-1, 1)),
            angular_velocity_constraints_upper_bound.reshape((-1, 1)),
        )

        if num_obstacles > 0:
            (
                obstacle_constraints_lower_bound,
                obstacle_constraints_upper_bound,
            ) = self._get_obstacle_constraints_bounds(
                inflation_radius=inflation_radius, num_obstacles=num_obstacles
            )
            constraints_lower_bound = DM_vertcat(
                constraints_lower_bound,
                obstacle_constraints_lower_bound.reshape((-1, 1)),
            )
            constraints_upper_bound = DM_vertcat(
                constraints_upper_bound,
                obstacle_constraints_upper_bound.reshape((-1, 1)),
            )

        return constraints_lower_bound, constraints_upper_bound

    def solve(
        self,
        current_state: np.ndarray,
        current_linear_velocity: float,
        current_angular_velocity: float,
        goal_state: np.ndarray,
        states_matrix: np.ndarray,
        controls_matrix: np.ndarray,
        left_right_lane_bounds: Tuple[float, float],
        linear_velocity_bounds: Tuple[float, float],
        angular_velocity_bounds: Tuple[float, float],
        linear_acceleration_bounds: Tuple[float, float],
        angular_acceleration_bounds: Tuple[float, float],
        static_obstacles: List["StaticObstacle"] = [],
        dynamic_obstacles: List[
            Union["DynamicObstacle", "SimulatedDynamicObstacle"]
        ] = [],
        inflation_radius: Optional[float] = None,
    ):
        non_linear_program = {
            "x": self.symbolic_optimization_variables,
            "f": self._get_symbolic_costs(
                # visible_obstacles=(obstacles if obstacles is not None else []),
                # inflation_radius=(
                #     inflation_radius if inflation_radius is not None else 0
                # ),
            ),
            "g": self._get_symbolic_constraints(
                current_linear_velocity=current_linear_velocity,
                current_angular_velocity=current_angular_velocity,
                static_obstacles=static_obstacles,
                dynamic_obstacles=dynamic_obstacles,
            ),
            "p": self.symbolic_terminal_states_vector,
        }

        solver_options = {
            "ipopt": {
                "max_iter": 2000,
                "print_level": 0,
                "acceptable_tol": 1e-8,
                "acceptable_obj_change_tol": 1e-6,
            },
            "print_time": 0,
        }

        solver = ca.nlpsol("solver", "ipopt", non_linear_program, solver_options)

        (
            constraints_lower_bounds,
            constraints_upper_bounds,
        ) = self._get_constraints_bounds(
            linear_velocity_bounds=linear_velocity_bounds,
            angular_velocity_bounds=angular_velocity_bounds,
            inflation_radius=(inflation_radius if inflation_radius is not None else 0),
            num_obstacles=len(static_obstacles + dynamic_obstacles),
        )

        (
            optimization_variable_lower_bounds,
            optimization_variable_upper_bounds,
        ) = self._get_optimization_variable_bounds(
            left_right_lane_bounds=left_right_lane_bounds,
            linear_acceleration_bounds=linear_acceleration_bounds,
            angular_acceleration_bounds=angular_acceleration_bounds,
        )

        solution = solver(
            x0=DM_vertcat(
                ca.reshape(
                    ca.DM(states_matrix),
                    (self.num_states * (self.horizon + 1), 1),
                ),
                ca.reshape(
                    ca.DM(controls_matrix),
                    (self.num_controls * self.horizon, 1),
                ),
            ),
            lbx=optimization_variable_lower_bounds,
            ubx=optimization_variable_upper_bounds,
            lbg=constraints_lower_bounds,
            ubg=constraints_upper_bounds,
            p=DM_vertcat(ca.DM(current_state), ca.DM(goal_state)),
        )
        updated_states_matrix = cast(
            ca.DM,
            ca.reshape(
                solution["x"][: self.num_states * (self.horizon + 1)],
                (self.num_states, self.horizon + 1),
            ),
        )
        updated_controls_matrix = cast(
            ca.DM,
            ca.reshape(
                solution["x"][self.num_states * (self.horizon + 1) :],
                (self.num_controls, self.horizon),
            ),
        )

        return np.array(updated_states_matrix.full()), np.array(
            updated_controls_matrix.full()
        )

    # def _get_symbolic_obstacle_cost(
    #     self, visible_obstacles: List[Obstacle], inflation_radius: float
    # ) -> ca.MX:
    #     # Calculate cost as proportional to the distance within the inflation radius of each obstacle
    #     distance_from_obstacle = MX_horzcat(
    #         *[
    #             obstacle.calculate_symbolic_matrix_distance(
    #                 symbolic_states_matrix=self.symbolic_states_matrix[:, 1:]
    #             )
    #             for obstacle in visible_obstacles
    #         ]
    #     )

    #     # Utilize self.obstacle_cost_weight to scale the cost
    #     scaled_cost = ca.sum1(
    #         ca.sum2(
    #             self.obstacle_cost_weight
    #             * ca.fmax(0, inflation_radius - distance_from_obstacle)
    #         )
    #     )
    #     return scaled_cost

    # @property
    # def lane_cost(self, lane_bounds_x: ca.SX):
    #     cost = 0
    #     for timestep in range(self.horizon+1):
    #         state = self.symbolic_states_matrix[0, timestep]
    #         cost += (state - lane_bounds_x)**2
    #     return cost
