from typing import TYPE_CHECKING, List, Optional, Tuple, Union, cast

import casadi as ca
import numpy as np

if TYPE_CHECKING:
    from mpc.dynamic_obstacle import DynamicObstacle, SimulatedDynamicObstacle
    from mpc.obstacle import StaticObstacle
from mpc.geometry import Circle


def MX_horzcat
