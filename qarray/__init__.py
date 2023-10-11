"""
Qarray is a Python package for simulating quantum dot arrays.
"""
__version__ = "1.0.0"

from .classes import (DotArray, GateVoltageComposer, ChargeSensedDotArray)
from .functions import (optimal_Vg, compute_threshold, convert_to_maxwell, dot_occupation_changes, lorentzian)
from .python_core import (ground_state_open_python, ground_state_closed_python)
from .rust_core import (ground_state_open_rust, ground_state_closed_rust, closed_charge_configurations_rust)