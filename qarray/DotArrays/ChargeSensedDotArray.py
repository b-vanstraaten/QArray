"""
A class that represents a charge sensed dot array. The class has methods to compute the ground state of the dot array
and the charge sensor output for both open and closed dot arrays.
"""

from dataclasses import dataclass

import numpy as np
from ..functions import compute_optimal_virtual_gate_matrix

from .GateVoltageComposer import GateVoltageComposer
from ._helper_functions import check_algorithm_and_implementation, \
    check_and_warn_user, lorentzian, convert_to_maxwell, _convert_to_maxwell_with_sensor, suppress_output
from .ground_state import _ground_state_open, _ground_state_closed
from ..functions import optimal_Vg, compute_threshold
from ..latching_models import LatchingBaseModel
from ..noise_models import BaseNoiseModel
from ..python_implementations.helper_functions import free_energy
from ..qarray_types import CddNonMaxwell, CgdNonMaxwell, VectorList, CdsNonMaxwell, CgsNonMaxwell, Vector, \
    PositiveValuedMatrix



@dataclass
class ChargeSensedDotArray:
    """
    A class that represents a charge sensed dot array. The class has methods to compute the ground state of the dot array
    and the charge sensor output for both open and closed dot arrays.

    The class has the following attributes:

    - Cdd: an (n_dot, n_dot) array of the capacitive coupling between dots
    - Cgd: an (n_dot, n_gate) array of the capacitive coupling between gates and dots
    - Cds: an (n_sensor, n_dot) array of the capacitive coupling between dots and sensors
    - Cgs: an (n_sensor, n_gate) array of the capacitive coupling between gates and dots

    - algorithm: the algorithm to use to compute the ground state
    - implementation: the implementation to use to compute the ground state
    - threshold: the threshold to use if the threshold algorithm is used
    - max_charge_carriers: the maximum number of charge carriers to use if the brute force algorithm is used
    - polish: a bool specifying whether to polish the result of the ground state computation by the default or thresholded algorithm

    - coulomb_peak_width: the width of the lorentzian peaks

    - noise_model: the noise model to use to add noise to the charge sensor output


    """

    Cdd: CddNonMaxwell  # an (n_dot, n_dot) array of the capacitive coupling between dots
    Cgd: CgdNonMaxwell  # an (n_dot, n_gate) array of the capacitive coupling between gates and dots

    Cds: CdsNonMaxwell  # an (n_sensor, n_dot) array of the capacitive coupling between dots and sensors
    Cgs: CgsNonMaxwell  # an (n_sensor, n_gate) array of the capacitive coupling between gates and dots

    algorithm: str | None = 'default'  # which algorithm to use
    implementation: str | None = 'rust'  # which implementation of the algorithm to use

    threshold: float | str = 1.  # if the threshold algorithm is used the user needs to pass the threshold
    max_charge_carriers: int | None = None  # if the brute force algorithm is used the user needs to pass the maximum number of charge carriers
    polish: bool = True  # a bool specifying whether to polish the result of the ground state computation
    max_charge_carriers: int | None = None  # need if using a brute_force algorithm
    batch_size: int | None = None  # needed if using jax implementation
    charge_carrier: str = 'h'

    T: float | int = 0.  # the temperature of the system in mK
    n_peak: int = 5

    coulomb_peak_width: float = 0.1  # the width of the lorentzian peaks
    noise_model: BaseNoiseModel | None = None
    latching_model: LatchingBaseModel | None = None

    def update_capacitance_matrices(self, Cdd: CddNonMaxwell, Cgd: CgdNonMaxwell, Cds: CdsNonMaxwell,
                                    Cgs: CgsNonMaxwell):

        self.Cdd = PositiveValuedMatrix(Cdd)
        self.Cgd = PositiveValuedMatrix(Cgd)
        self.Cds = PositiveValuedMatrix(Cds)
        self.Cgs = PositiveValuedMatrix(Cgs)

        self.n_dot = self.Cdd.shape[0]
        self.n_sensor = self.Cds.shape[0]
        self.n_gate = self.Cgd.shape[1]
        self._assert_shape()

        self.cdd, self.cdd_inv, self.cgd = convert_to_maxwell(self.Cdd, self.Cgd)
        self.cdd_full, self.cdd_inv_full, self.cgd_full = _convert_to_maxwell_with_sensor(self.Cdd, self.Cgd, self.Cds, self.Cgs)

        self.cgs = self.Cgs
        self.cds = self.Cds


    def __post_init__(self):

        # converting to the non-maxwellian capacitance matrices to their respective type. This
        # is done to ensure that the capacitance matrices are of the correct type and the values are correct. Aka
        # the capacitance matrices are positive and the diagonal elements are zero.

        self.update_capacitance_matrices(self.Cdd, self.Cgd, self.Cds, self.Cgs)

        # type casting the temperature to a float
        self.T = float(self.T)

        # checking the passed algorithm and implementation
        check_algorithm_and_implementation(self.algorithm, self.implementation)
        if self.algorithm == 'threshold':
            assert self.threshold is not None, 'The threshold must be specified when using the thresholded algorithm'

        if self.algorithm == 'brute_force':
            assert self.max_charge_carriers is not None, 'The maximum number of charge carriers must be specified'

        if self.noise_model is None:
            # this is the default noise model adds no noise
            self.noise_model = BaseNoiseModel()

        if self.latching_model is None:
            self.latching_model = LatchingBaseModel()

        if self.algorithm in ['thresholded', 'default']:
            check_and_warn_user(self)

        self.gate_voltage_composer = GateVoltageComposer(n_gate=self.n_gate, n_dot=self.n_dot, n_sensor=self.n_sensor)
        self.gate_voltage_composer.virtual_gate_matrix = -np.linalg.pinv(self.cdd_inv_full @ self.cgd_full)
        self.gate_voltage_composer.virtual_gate_origin = np.zeros(self.n_gate)

    def do1d_open(self, gate: int | str, min: float, max: float, points: int) -> np.ndarray:
        """
        Performs a 1D sweep of the dot array with the gate in the open configuration

        :param gate: the gate to sweep
        :param min: the minimum value of the gate to sweep
        :param max: the maximum value of the gate to sweep
        :param points: the number of res to sweep the gate over

        returns the ground state of the dot array which is a np.ndarray of shape (res, n_dot) in the open configuration
        """

        vg = self.gate_voltage_composer.do1d(gate, min, max, points)
        return self.charge_sensor_open(vg)

    def gui(self, port=9001, run=True, print_compute_time=True, initial_dac_values=None, initial_virtual_gate_matrix=None):
        """
        A function to open the GUI for the ChargeSensedDotArray class
        """
        from ..gui.gui_charge_sensor import run_gui_charge_sensor

        run_gui_charge_sensor(self, port = port, run = run, print_compute_time = print_compute_time, initial_dac_values = initial_dac_values, initial_virtual_gate_matrix = initial_virtual_gate_matrix)

    def compute_optimal_virtual_gate_matrix(self):
        """
        Computes the optimal virtual gate matrix for the dot array and sets it as the virtual gate matrix
        in the gate voltage composer.

        The virtual gate matrix is computed as the pseudo inverse of the dot to dot capacitance matrix times the dot to gate capacitance matrix.

        returns np.ndarray: the virtual gate matrix
        """
        virtual_gate_matrix = compute_optimal_virtual_gate_matrix(self.cdd_inv_full, self.cgd_full)
        self.gate_voltage_composer.virtual_gate_matrix = virtual_gate_matrix

        if self.charge_carrier == 'electrons':
            virtual_gate_matrix = -virtual_gate_matrix

        return virtual_gate_matrix

    def compute_optimal_sensor_virtual_gate_matrix(self):
        assert self.n_sensor == 1, 'currently not implemented for more than one sensor'
        virtual_gate_matrix = compute_optimal_virtual_gate_matrix(self.cdd_inv_full, self.cgd_full)

        device_virtual_gates = virtual_gate_matrix[:-1, :-1]
        sensor_gates = virtual_gate_matrix[-1, :-1]

        inv = np.linalg.inv(device_virtual_gates)
        sensor_virtual_matrix = np.eye(self.n_gate)
        sensor_virtual_matrix[-1, :-1] = inv @ sensor_gates

        self.gate_voltage_composer.virtual_gate_matrix = sensor_virtual_matrix
        return sensor_virtual_matrix



    def do1d_closed(self, gate: int | str, min: float, max: float, points: int, n_charge: int) -> np.ndarray:
        """
        Performs a 1D sweep of the dot array with the gate in the closed configuration

        :param gate: the gate to sweep
        :param min: the minimum value of the gate to sweep
        :param max: the maximum value of the gate to sweep
        :param points: the number of res to sweep the gate over

        returns the ground state of the dot array which is a np.ndarray of shape (res, n_dot) in the closed configuration
        """
        vg = self.gate_voltage_composer.do1d(gate, min, max, points)
        return self.charge_sensor_closed(vg, n_charge)

    def do2d_open(self, x_gate: int | str, x_min: float, x_max: float, x_points: int,
                  y_gate: int | str, y_min: float, y_max: float, y_points: int) -> np.ndarray:
        """
        Performs a 2D sweep of the dot array with the gates x_gate and y_gate in the open configuration

        :param x_gate: the gate to sweep in the x direction
        :param x_min: the minimum value of the gate to sweep
        :param x_max: the maximum value of the gate to sweep
        :param x_points: the number of res to sweep the gate over
        :param y_gate: the gate to sweep in the y direction
        :param y_min: the minimum value of the gate to sweep
        :param y_max: the maximum value of the gate to sweep
        :param y_points: the number of res to sweep

        returns the ground state of the dot array which is a np.ndarray of shape (x_res, y_res, n_dot) in the open
        configuration
        """

        vg = self.gate_voltage_composer.do2d(x_gate, x_min, x_max, x_points, y_gate, y_min, y_max, y_points)
        return self.charge_sensor_open(vg)

    def do2d_closed(self, x_gate: int | str, x_min: float, x_max: float, x_points: int,
                    y_gate: int | str, y_min: float, y_max: float, y_points: int, n_charge: int) -> np.ndarray:
        """
        Performs a 2D sweep of the dot array with the gates x_gate and y_gate in the open configuration

        :param x_gate: the gate to sweep in the x direction
        :param x_min: the minimum value of the gate to sweep
        :param x_max: the maximum value of the gate to sweep
        :param x_points: the number of res to sweep the gate over
        :param y_gate: the gate to sweep in the y direction
        :param y_min: the minimum value of the gate to sweep
        :param y_max: the maximum value of the gate to sweep
        :param y_points: the number of res to sweep

        returns the ground state of the dot array which is a np.ndarray of shape (x_res, y_res, n_dot)
        in the closed configuration
        """


    def optimal_Vg(self, n_charges: VectorList, rcond: float = 1e-3) -> np.ndarray:
        """
        Computes the optimal dot voltages for a given charge configuration, of shape (n_dot + n_sensor,).
        :param n_charges: the charge configuration
        :param rcond: the rcond parameter for the least squares solver
        :return: the optimal dot voltages of shape (n_gate,)
        """
        n_charges = Vector(n_charges)
        assert n_charges.shape == (
            self.n_dot + self.n_sensor,), 'The n_charge vector must be of shape (n_dot + n_sensor)'
        return optimal_Vg(cdd_inv=self.cdd_inv_full, cgd=self.cgd_full, n_charges=n_charges, rcond=rcond)

    def ground_state_open(self, vg: VectorList | np.ndarray) -> np.ndarray:
        """
        Computes the ground state for an open dot array.
        :param vg: (..., n_gate) array of dot voltages to compute the ground state for
        :return: (..., n_dot) array of ground state charges
        """
        return _ground_state_open(self, vg)

    def charge_sensor_open(self, vg: VectorList | np.ndarray) -> np.ndarray:
        """
        Computes the charge sensor output for an open dot array.

        :param vg: (..., n_gate) array of dot voltages to compute the charge sensor output for
        :return: (..., n_sensor) array of the charge sensor output

        Noise is added according to the noise model passed to the ChargeSensedDotArray class.
        """

        n_open = self.ground_state_open(vg)

        # computing the continuous minimum charge state (open)
        N_cont = np.einsum('ij, ...j', self.cgd_full, vg)

        # computing the discrete state on the charge sensor
        N_sensor = np.round(N_cont[..., self.n_dot:])

        # computing the noise to be added to the charge sensor potential before it is used in as the input to the lorentzian
        input_noise = self.noise_model.sample_input_noise(N_sensor.shape)

        F = np.zeros(shape=(2 * self.n_peak + 1, *N_sensor.shape))
        for sensor in range(self.n_sensor):
            for i, n in enumerate(range(-self.n_peak, self.n_peak + 1)):
                perturbed_N_sensor = N_sensor.copy()
                perturbed_N_sensor[..., sensor] = perturbed_N_sensor[..., sensor] + n
                N_full = np.concatenate([n_open, perturbed_N_sensor + input_noise], axis=-1)
                F[i, ..., sensor] = free_energy(self.cdd_inv_full, self.cgd_full, vg, N_full)

        signal = lorentzian(np.diff(F, axis=0), 0, self.coulomb_peak_width).sum(axis=0)
        output_noise = self.noise_model.sample_output_noise(N_sensor.shape)
        return signal + output_noise, n_open

    def ground_state_closed(self, vg: VectorList | np.ndarray, n_charge: int) -> np.ndarray:
        """
        Computes the ground state for a closed dot array.
        :param vg: (..., n_gate) array of dot voltages to compute the ground state for
        :param n_charge: the number of charges to be confined in the dot array
        :return: (..., n_dot) array of the number of charges to compute the ground state for
        """
        return _ground_state_closed(self, vg, n_charge)

    def charge_sensor_closed(self, vg: VectorList | np.ndarray, n_charge) -> np.ndarray:
        """
        Computes the charge sensor output for a closed dot array.

        :param vg: (..., n_gate) array of dot voltages to compute the charge sensor output for
        :param n_charge: the number of charges to be confined in the dot array
        :return: (..., n_sensor) array of the charge sensor output

        Noise is added according to the noise model passed to the ChargeSensedDotArray class.
        """
        n_closed = self.ground_state_closed(vg, n_charge)

        # computing the continuous minimum charge state (open)
        N_cont = np.einsum('ij, ...j', self.cgd_full, vg)

        # computing the discrete state on the charge sensor
        N_sensor = np.round(N_cont[..., self.n_dot:])

        # computing the noise to be added to the charge sensor potential before it is used in as the input to the lorentzian
        input_noise = self.noise_model.sample_input_noise(N_sensor.shape)

        F = np.zeros(shape=(2 * self.n_peak + 1, *N_sensor.shape))
        for sensor in range(self.n_sensor):
            for i, n in enumerate(range(-self.n_peak, self.n_peak + 1)):
                perturbed_N_sensor = N_sensor.copy()
                perturbed_N_sensor[..., sensor] = perturbed_N_sensor[..., sensor] + n
                N_full = np.concatenate([n_closed, perturbed_N_sensor + input_noise], axis=-1)
                F[i, ..., sensor] = free_energy(self.cdd_inv_full, self.cgd_full, vg, N_full)

        signal = lorentzian(np.diff(F, axis=0), 0, self.coulomb_peak_width).sum(axis=0)
        output_noise = self.noise_model.sample_output_noise(N_sensor.shape)

        return signal + output_noise, n_closed

    def check_threshold(self):
        """
        Checks if the threshold is below the optimal threshold for the system
        """
        optimal_threshold = compute_threshold(self.cdd)

        if optimal_threshold > 1:
            print(f'Warning: The default nor thresholded algorithm is not recommended for this system')
            return

        match self.algorithm:
            case 'thresholded':

                if self.threshold < optimal_threshold:
                    print(f'Warning: The threshold is below the suggested threshold of {optimal_threshold}.')
                    return
            case 'default':
                if optimal_threshold > 1:
                    print(f'Warning: The default algorithm is not recommended for this system')

    def compute_threshold_estimate(self):
        """
        Computes the threshold estimate for the dot array for use with the thresholded algorithm
        """
        return compute_threshold(self.cdd)

    def _assert_shape(self):
        """
        A function to assert the shape of the capacitance matrices.
        """

        # checking the shape of the cgd matrix
        assert self.Cgd.shape[0] == self.n_dot, f'Cgd must be of shape (n_dot, n_gate) = ({self.n_dot}, {self.n_gate})'
        assert self.Cgd.shape[1] == self.n_gate, f'Cdd must be of shape (n_dot, n_gate) = ({self.n_dot}, {self.n_gate})'

        # checking the shape of the cds matrix
        assert self.Cds.shape[0] == self.n_sensor, 'Cds must be of shape (n_sensor, n_dot)'
        assert self.Cds.shape[1] == self.n_dot, 'Cds must be of shape (n_sensor, n_dot)'

        # checking the shape of the cgs matrix
        assert self.Cgs.shape[0] == self.n_sensor, 'Cgs must be of shape (n_sensor, n_gate)'
        assert self.Cgs.shape[1] == self.n_gate, 'Cgs must be of shape (n_sensor, n_gate)'
