from dataclasses import dataclass

import numpy as np

from .GateVoltageComposer import GateVoltageComposer
from ._helper_functions import (check_algorithm_and_implementation,
                                check_and_warn_user, convert_to_maxwell)
from .ground_state import _ground_state_open, _ground_state_closed
from ..functions import optimal_Vg, compute_threshold, compute_optimal_virtual_gate_matrix
from ..latching_models import LatchingBaseModel
from ..qarray_types import Cdd as CddType  # to avoid name clash with dataclass cdd
from ..qarray_types import CgdNonMaxwell, CddNonMaxwell, VectorList, Cgd_holes, Cgd_electrons, PositiveValuedMatrix, \
    NegativeValuedMatrix


def both_none(a, b):
    return a is None and b is None


def all_positive(a):
    return np.all(a >= 0)


def all_negative(a):
    return np.all(a <= 0)


def all_positive_or_negative(a):
    return all_positive(a) or all_negative(a)


@dataclass
class DotArray:

    """
    A class to represent a quantum dot array. The class is initialized with the following parameters:

    Cdd: CddNonMaxwell | None = None  # an (n_dot, n_dot) the capacitive coupling between dots
    Cgd: CgdNonMaxwell | None = None  # an (n_dot, n_gate) the capacitive coupling between gates and dots

    cdd: Cdd | None = None  # an (n_dot, n_dot) the capacitive coupling between dots
    cgd: PositiveValuedMatrix | NegativeValuedMatrix | None = None # an (n_dot, n_gate) the capacitive coupling between gates and dots

    algorithm: str | None = 'default'  # which algorithm to use
    implementation: str | None = 'rust'

    threshold: float | str = 1.  # if the threshold algorithm is used the user needs to pass the threshold
    max_charge_carriers: int | None = None  # if the brute force algorithm is used the user needs to pass the maximum number of charge carriers

    charge_carrier: str = 'hole'  # a string of either 'electron' or 'hole' to specify the charge carrie
    T: float | int = 0.  # the temperature of the system for if there is thermal broadening

    """

    Cdd: CddNonMaxwell | None = None  # an (n_dot, n_dot)the capacitive coupling between dots
    Cgd: CgdNonMaxwell | None = None  # an (n_dot, n_gate) the capacitive coupling between gates and dots
    cdd: CddType | None = None
    cgd: PositiveValuedMatrix | NegativeValuedMatrix | None = None
    cdd_inv: np.ndarray | None = None

    algorithm: str | None = 'default'  # which algorithm to use
    implementation: str | None = 'rust'  # which implementation of the algorithm to use
    threshold: float | str = 1.  # if the threshold algorithm is used the user needs to pass the threshold
    max_charge_carriers: int | None = None  # if the brute force algorithm is used the user needs to pass the maximum number of charge carriers

    charge_carrier: str = 'hole'  # a string of either 'electron' or 'hole' to specify the charge carrie
    T: float | int = 0.  # the temperature of the system, only used for jax and jax_brute_force cores
    batch_size: int = 10000
    polish: bool = True  # a bool specifying whether to polish the result of the ground state computation

    latching_model: LatchingBaseModel | None = None  # a latching model to add latching to the dot occupation vector
    gate_voltage_composer: GateVoltageComposer | None = None  # a gate voltage composer to create gate voltage arrays

    def update_capacitance_matrices(self, Cdd: CddNonMaxwell, Cgd: CgdNonMaxwell):
        """
        Updates the capacitance matrices of the dot array

        :param Cdd: the new Cdd matrix
        :param Cgd: the new Cgd matrix

        :return: None
        """

        self.Cdd = PositiveValuedMatrix(Cdd)
        self.Cgd = PositiveValuedMatrix(Cgd)
        self.cdd, self.cdd_inv, self.cgd = convert_to_maxwell(self.Cdd, self.Cgd)
        self._process_capacitance_matricies()

    def _process_capacitance_matricies(self):
        """
        Processes the capacitance matrices of the dot array

        :return: None
        """

        if self.cdd_inv is None:
            self.cdd_inv = np.linalg.inv(self.cdd)

        # by now in the code, the cdd and cgd matrices have been initialized as their specified types. These
        # types enforce most of the constraints on the matrices like positive-definitness for cdd for example,
        # but not all. The following asserts check the remainder.
        self.n_dot = self.cdd.shape[0]
        self.n_gate = self.cgd.shape[1]
        assert self.cgd.shape[0] == self.n_dot, 'The number of dots must be the same as the number of rows in cgd'

        self.gate_voltage_composer = GateVoltageComposer(n_gate=self.n_gate, n_dot=self.n_dot)
        self.gate_voltage_composer.virtual_gate_origin = np.zeros(self.n_gate)
        self.gate_voltage_composer.virtual_gate_matrix = -np.linalg.pinv(self.cdd_inv @ self.cgd)

        # setting the cdd_inv attribute as the inverse of cdd
        self.cdd_inv = np.linalg.inv(self.cdd)

        # checking that the cgd matrix has all positive or all negative elements
        # so that the sign can be matched to the charge carrier
        assert all_positive_or_negative(self.cgd), 'The elements of cgd must all be positive or all be negative'

        # matching the sign of the cgd matrix to the charge carrier
        match self.charge_carrier.lower():
            case 'electrons' | 'electron' | 'e' | '-':
                self.charge_carrier = 'electrons'
                # the cgd matrix is positive for electrons
                self.cgd = Cgd_electrons(np.abs(self.cgd))
            case 'holes' | 'hole' | 'h' | '+':
                self.charge_carrier = 'holes'
                # the cgd matrix is negative for holes
                self.cgd = Cgd_holes(-np.abs(self.cgd))
            case _:
                raise ValueError(f'charge_carrier must be either "electrons" or "holes {self.charge_carrier}"')

    def __post_init__(self):
        """
        This function is called after the initialization of the dataclass. It checks that the capacitance matrices
        are valid and sets the cdd_inv attribute as the inverse of cdd.
        """

        # checking that either cdd and cgd or cdd and cgd are specified
        non_maxwell_pair_passed = not both_none(self.Cdd, self.Cgd)
        maxwell_pair_passed = not both_none(self.cdd, self.cgd)
        assertion_message = 'Either cdd and cgd or cdd and cgd must be specified'
        assert (non_maxwell_pair_passed or maxwell_pair_passed), assertion_message

        # if the non maxwell pair is passed, convert it to maxwell
        if non_maxwell_pair_passed:
            self.update_capacitance_matrices(self.Cdd, self.Cgd)
        else:
            self.cdd = CddType(self.cdd)
            self.cgd = np.array(self.cgd)
            self._process_capacitance_matricies()

        # type casting the temperature to a float
        self.T = float(self.T)

        # checking the passed algorithm and implementation
        check_algorithm_and_implementation(self.algorithm, self.implementation)
        if self.algorithm == 'threshold':
            assert self.threshold is not None, 'The threshold must be specified when using the thresholded algorithm'

        if self.algorithm == 'brute_force':
            assert self.max_charge_carriers is not None, 'The maximum number of charge carriers must be specified'

        if self.latching_model is None:
            self.latching_model = LatchingBaseModel()

        if self.algorithm in ['thresholded', 'default']:
            check_and_warn_user(self)

        self.gate_voltage_composer = GateVoltageComposer(n_gate=self.n_gate, n_dot=self.n_dot)
        self.gate_voltage_composer.virtual_gate_origin = np.zeros(self.n_gate)
        self.gate_voltage_composer.virtual_gate_matrix = -np.linalg.pinv(self.cdd_inv @ self.cgd)



    def optimal_Vg(self, n_charges: VectorList, rcond: float = 1e-3) -> np.ndarray:
        """
        Computes the optimal dot voltages for a given charge configuration, of shape (n_charge,).
        :param n_charges: the charge configuration
        :param rcond: the rcond parameter for the least squares solver
        :return: the optimal dot voltages of shape (n_gate,)
        """
        return optimal_Vg(cdd_inv=self.cdd_inv, cgd=self.cgd, n_charges=n_charges, rcond=rcond)

    def ground_state_open(self, vg: VectorList | np.ndarray) -> np.ndarray:
        """
        Computes the ground state for an open dot array.
        :param vg: (..., n_gate) array of dot voltages to compute the ground state for
        :return: (..., n_dot) array of ground state charges
        """
        return _ground_state_open(self, vg)

    def ground_state_closed(self, vg: VectorList | np.ndarray, n_charges: int) -> np.ndarray:
        """
        Computes the ground state for a closed dot array.
        :param vg: (..., n_gate) array of dot voltages to compute the ground state for
        :param n_charges: the number of charges to be confined in the dot array
        :return: (..., n_dot) array of the number of charges to compute the ground state for
        """
        return _ground_state_closed(self, vg, n_charges)

    def free_energy(self, n, vg):
        """
        Computes the free energy of the change configurations
        """
        n_cont_min = np.einsum('ij, ...i', self.cgd, vg)
        delta = n[np.newaxis, np.newaxis, :] - n_cont_min[..., np.newaxis, :]
        # computing the free energy of the change configurations
        F = np.einsum('...i, ij, ...j', delta, self.cdd_inv, delta)
        return F

    def compute_threshold_estimate(self):
        """
        Computes the threshold estimate for the dot array for the thresholded algorithm
        """
        return compute_threshold(self.cdd)

    def do1d_open(self, gate: int | str, min: float, max: float, res: int) -> np.ndarray:
        """
        Performs a 1D sweep of the dot array with the gate

        :param gate: the gate to sweep
        :param min: the minimum value of the gate to sweep
        :param max: the maximum value of the gate to sweep
        :param res: the number of res to sweep the gate over

        returns the ground state of the dot array which is a np.ndarray of shape (res, n_dot)
        """

        vg = self.gate_voltage_composer.do1d(gate, min, max, res)
        return self.ground_state_open(vg)

    def do1d_closed(self, gate: int | str, min: float, max: float, res: int, n_charges: int) -> np.ndarray:
        """
        Performs a 1D sweep of the dot array with the gate

        :param gate: the gate to sweep
        :param min: the minimum value of the gate to sweep
        :param max: the maximum value of the gate to sweep
        :param res: the number of res to sweep the gate over
        :param n_charges: the number of charges to be confined in the dot array

        returns the ground state of the dot array which is a np.ndarray of shape (res, n_dot)
        """

        vg = self.gate_voltage_composer.do1d(gate, min, max, res)
        return self.ground_state_closed(vg, n_charges)

    def do2d_open(self, x_gate: int | str, x_min: float, x_max: float, x_res: int,
                  y_gate: int | str, y_min: float, y_max: float, y_res: int) -> np.ndarray:
        """
        Performs a 2D sweep of the dot array with the gates x_gate and y_gate

        :param x_gate: the gate to sweep in the x direction
        :param x_min: the minimum value of the gate to sweep
        :param x_max: the maximum value of the gate to sweep
        :param x_res: the number of res to sweep the gate over
        :param y_gate: the gate to sweep in the y direction
        :param y_min: the minimum value of the gate to sweep
        :param y_max: the maximum value of the gate to sweep
        :param y_res: the number of res to sweep the

        returns the ground state of the dot array which is a np.ndarray of shape (x_res, y_res, n_dot)
        """

        vg = self.gate_voltage_composer.do2d(x_gate, x_min, x_max, x_res, y_gate, y_min, y_max, y_res)
        return self.ground_state_open(vg)

    def do2d_closed(self, x_gate: int | str, x_min: float, x_max: float, x_res: int,
                    y_gate: int | str, y_min: float, y_max: float, y_res: int, n_charges: int) -> np.ndarray:
        """
        Performs a 2D sweep of the dot array with the gates x_gate and y_gate

        :param x_gate: the gate to sweep in the x direction
        :param x_min: the minimum value of the gate to sweep
        :param x_max: the maximum value of the gate to sweep
        :param x_res: the number of res to sweep the gate over
        :param y_gate: the gate to sweep in the y direction
        :param y_min: the minimum value of the gate to sweep
        :param y_max: the maximum value of the gate to sweep
        :param y_res: the number of res to sweep the gate over
        :param n_charges: the number of charges to be confined in the dot array

        returns the ground state of the dot array which is a np.ndarray of shape (x_res, y_res, n_dot)
        """
        vg = self.gate_voltage_composer.do2d(x_gate, x_min, x_max, x_res, y_gate, y_min, y_max, y_res)
        return self.ground_state_closed(vg, n_charges)

    def compute_optimal_virtual_gate_matrix(self):
        """
        Computes the optimal virtual gate matrix for the dot array and sets it as the virtual gate matrix
        in the gate voltage composer.

        The virtual gate matrix is computed as the pseudo inverse of the dot to dot capacitance matrix times the dot to gate capacitance matrix.

        returns np.ndarray: the virtual gate matrix
        """
        virtual_gate_matrix = compute_optimal_virtual_gate_matrix(self.cdd_inv, self.cgd)
        self.gate_voltage_composer.virtual_gate_matrix = virtual_gate_matrix

        if self.charge_carrier == 'electrons':
            virtual_gate_matrix = -virtual_gate_matrix

        return virtual_gate_matrix

    def run_gui(self, port=9000, print_compute_time: bool = False, initial_dac_values=None):
        """
        Creates a GUI for the dot array

        :param port: the port to run the GUI on
        :param print_compute_time: a bool specifying whether to print the compute time
        :param initial_dac_values: the initial dac values to set the GUI to as a numpy array of length n_gate

        """
        # importing the run_gui function here to avoid circular imports
        from ..gui import run_gui
        run_gui(self, port=port, print_compute_time=print_compute_time, initial_dac_values=initial_dac_values)
