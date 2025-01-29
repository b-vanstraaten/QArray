"""
Qarray, a GPU accelerated quantum dot array simulator, leveraging parallelised Rust and JAX XLA acceleration
to compute charge stability diagrams of large both open and closed arrays in milliseconds.
"""
__version__ = "1.5.0"

from .DotArrays import (DotArray, GateVoltageComposer, ChargeSensedDotArray)
from .functions import (_optimal_Vg, dot_occupation_changes, charge_state_changes,
                        charge_state_to_scalar, compute_optimal_virtual_gate_matrix, charge_state_dot_product)
from .latching_models import *
from .noise_models import *

__all__ = [
    'DotArray', 'GateVoltageComposer', 'ChargeSensedDotArray',
    '_optimal_Vg', 'dot_occupation_changes', 'charge_state_changes',
    'dot_occupation_changes', 'charge_state_dot_product',
    'charge_state_to_scalar', 'compute_optimal_virtual_gate_matrix',
]

submodules = ['latching_models', 'noise_models']
