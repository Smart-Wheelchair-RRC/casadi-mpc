from typing import TYPE_CHECKING, List, Optional, Tuple, Union, cast

import casadi as ca
import numpy as np

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
    def __init__(self, time_step: float, horizon: int):
        self.time_step = time_step
        self.horizon = horizon
        
        self.symbolic_states = SX_vertcat(
            create_symbolic_scalar("x"),
            create_symbolic_scalar("y"),
            create_symbolic_scalar("theta")
        ) 
        self.num_states = self.symbolic_states.numel()
        
        self.symbolic_controls = SX_vertcat(
            create_symbolic_scalar("vx"),
            create_symbolic_scalar("vz")
        )
        self.num_controls = self.symbolic_controls.numel()
        
        self.weight_matrix = ca.DM(ca.diagcat(100, 100, 0))
        
        self.linear_velocity_weight = ca.DM(50)
        self.angular_velocity_weight = ca.DM(30)
        
        self.symbolic_states_matrix = create_symbolic_matrix(
            "X", (self.num_states, self.horizon + 1)
        )
        
        self.symbolic_controls_matrix = create_symbolic_matrix(
            "U", (self.num_controls, self.horizon)
        )
        
        self.symbolic_terminal_states_vector = create_symbolic_matrix(
            "P", (self.num_states + self.num_states, 1)
        )
        
        self.symbolic_optimization_variables = MX_vertcat(
            self.symbolic_states_matrix.reshape((-1, 1)),
            self.symbolic_controls_matrix.reshape((-1, 1)),
        )
        
    def get_symbolic_goal_cost(self) -> ca.MX:
        error = self.symbolic_states_matrix[:, 1: -1] - self.symbolic_terminal_states_vector[self.num_states :]
        
        cost = ca.sum2((ca.transpose(error) @ self.weight_matrix) * ca.transpose(error))
        return ca.sum1(cost)
    
    def get_symbolic_linear_velocity_cost(self) -> ca.MX:
        squared_linear_velocity = self.symbolic_controls_matrix[0, :] ** 2
        return(
            (ca.sum1(ca.sum2(squared_linear_velocity))) * self.linear_velocity_weight
               )
        
    def get_symbolic_angular_velocity_cost(self) -> ca.MX:
        squared_angular_velocity = self.symbolic_controls_matrix[1, :] ** 2
        return(
            (ca.sum1(ca.sum2(squared_angular_velocity))) * self.angular_velocity_weight
               )
        
    def get_symbolic_costs(
        self,
    ) -> ca.MX:
        return(
            self.get_symbolic_goal_cost()
            + self.get_symbolic_linear_velocity_cost()
            + self.get_symbolic_angular_velocity_cost()
        )
    def get_state_bounds(
        self, state_bounds: Tuple[float, float]
    ) -> Tuple[ca.DM, ca.DM]:
        state_lower_bounds = ca.repmat(DM_vertcat(state_bounds[0], -ca.inf, -ca.inf), (1, self.horizon +1),)
        state_upper_bounds = ca.repmat(DM_vertcat(state_bounds[1], ca.inf, ca.inf), (1, self.horizon +1),)
        return state_lower_bounds, state_upper_bounds
    
    def get_control_bounds(
        self,
        linear_velocity_bounds: Tuple[float, float],
        angular_velocity_bounds: Tuple[float, float],
    ) -> Tuple[ca.DM, ca.DM]:
        control_lower_bounds = ca.repmat(
            DM_vertcat(
                linear_velocity_bounds[0],
                angular_velocity_bounds[0],
            ),
            (1, self.horizon),
        )
        control_upper_bounds = ca.repmat(
            DM_vertcat(
                linear_velocity_bounds[1],
                angular_velocity_bounds[1],
            ),
            (1, self.horizon),
        )
        return control_lower_bounds, control_upper_bounds
    
    def get_optimization_variable_bounds(
        self,
        state_bounds: Tuple[float, float],
        linear_velocity_bounds: Tuple[float, float],
        angular_velocity_bounds: Tuple[float, float]
    ) -> Tuple[ca.DM, ca.DM]:
        lower_state_bounds, upper_state_bounds = self.get_state_bounds(state_bounds)
        lower_control_bounds, upper_control_bounds = self.get_control_bounds(
            linear_velocity_bounds=linear_velocity_bounds,
            angular_velocity_bounds=angular_velocity_bounds
        )
        optimization_variable_lower_bounds = DM_vertcat(
            lower_state_bounds.reshape((-1, 1)), lower_control_bounds.reshape((-1, 1))
        )
        optimization_variable_upper_bounds = DM_vertcat(
            upper_state_bounds.reshape((-1, 1)), upper_control_bounds.reshape((-1, 1))
        )
        return optimization_variable_lower_bounds, optimization_variable_upper_bounds
    
    def get_state_constrains_bounds(self) -> Tuple[ca.DM, ca.DM]:
        return (
            ca.DM.zeros((3, self.horizon + 1)),
            ca.DM.zeros((3, self.horizon + 1)),
        )
    def get_symbolic_state_constrains(
        self, 
        current_linear_velocity: float, 
        current_angular_velocity: float) -> ca.MX:

            current_velocities = self.symbolic_controls_matrix
            
            next_states = MX_vertcat(
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
            initial_state_constraint = (
                self.symbolic_states_matrix[:, 0]
                - self.symbolic_terminal_states_vector[: self.num_states]
            )
            
            state_update_constraints = (
                self.symbolic_states_matrix[:, 1:] - next_states
            )
            return MX_horzcat(
                initial_state_constraint,
                state_update_constraints
            )
        
    def get_symbolic_constraints(
        self,
        current_linear_velocity: float,
        current_angular_velocity: float,
    ) -> ca.MX:
        symbolic_constraints = MX_vertcat(
            self.get_symbolic_state_constrains(
                current_linear_velocity=current_linear_velocity,
                current_angular_velocity=current_angular_velocity
            ).reshape((-1, 1)),
        )
        return symbolic_constraints
    
    def get_constraints_bounds(
        self,
        linear_velocity_bounds: Tuple[float, float],
        angular_velocity_bounds: Tuple[float, float],
        # inflation_radius: float = 0
    ) -> Tuple[ca.DM, ca.DM]:
        (
            state_constraints_lower_bound,
            state_constraints_upper_bound,
        ) = self.get_state_constrains_bounds()
        
        constraints_lower_bound = DM_vertcat(
            state_constraints_lower_bound.reshape((-1, 1))
        )
        constraints_upper_bound = DM_vertcat(
            state_constraints_upper_bound.reshape((-1, 1))
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
        state_bounds: np.ndarray,
        linear_velocity_bounds: np.ndarray,
        angular_velocity_bounds: np.ndarray,
    ):
        non_linear_program = {
            "x": self.symbolic_optimization_variables,
            "f": self.get_symbolic_costs(),
            "g": self.get_symbolic_constraints(
                current_linear_velocity=current_linear_velocity,
                current_angular_velocity=current_angular_velocity,
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
        ) = self.get_constraints_bounds(
            linear_velocity_bounds=linear_velocity_bounds,
            angular_velocity_bounds=angular_velocity_bounds
        )
        
        (
            optimization_variable_lower_bounds,
            optimization_variable_upper_bounds,
        ) = self.get_optimization_variable_bounds(
            state_bounds=state_bounds,
            linear_velocity_bounds=linear_velocity_bounds,
            angular_velocity_bounds=angular_velocity_bounds
        )
        
        solution = solver(
            x0=DM_vertcat(
                ca.reshape(
                    ca.DM(states_matrix),
                    (self.num_states *(self.horizon + 1), 1),
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
        updated_states_matrix = ca.reshape(
            solution["x"][: self.num_states * (self.horizon + 1)],
            (self.num_states, self.horizon + 1)
        )
        updated_controls_matrix = ca.reshape(
            solution["x"][self.num_states * (self.horizon + 1) :],
            (self.num_controls, self.horizon),
        )
        return np.array(updated_states_matrix.full()), np.array(updated_controls_matrix.full())