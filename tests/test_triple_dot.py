"""
Tests to check the capacitance model_threshold_1 works for double dot arrays
"""

import unittest

import matplotlib.pyplot as plt
import numpy as np

from qarray.functions import compute_threshold, optimal_Vg
from qarray.jax_implementations.default_jax import ground_state_open_default_jax, ground_state_closed_default_jax
from qarray.python_implementations import ground_state_open_default_or_thresholded_python, \
    ground_state_closed_default_or_thresholded_python
from qarray.rust_implemenations import ground_state_open_default_or_thresholded_rust, \
    ground_state_closed_default_or_thresholded_rust
from .GLOBAL_OPTIONS import N_ITERATIONS, N_VOLTAGES
from .helper_functions import randomly_generate_matrices, too_different


class TripleDotTests(unittest.TestCase):
    def test_python_vs_rust_open(self):
        """
        Test that the python and rust open double dot ground state functions return the same result.

        The threshold is set to 1, so every nearest neighbour change state is considered
        """

        for _ in range(N_ITERATIONS):
            cdd, cdd_inv, cgd = randomly_generate_matrices(3)
            vg = np.random.uniform(-5, 5, size=(N_VOLTAGES, 3))
            n_rust = ground_state_open_default_or_thresholded_rust(vg, cgd, cdd_inv, 1)
            n_python = ground_state_open_default_or_thresholded_python(vg, cgd, cdd_inv, 1)
            n_jax = ground_state_open_default_jax(vg, cgd, cdd_inv)

            debug = False
            if debug:
                if not np.allclose(n_rust, n_python, n_jax):
                    print(cdd_inv)
                    fig, ax = plt.subplots(1, 3)
                    ax[0].imshow(n_rust, aspect='auto')
                    ax[1].imshow(n_python, aspect='auto')
                    ax[2].imshow(n_rust - n_python, aspect='auto')
                    plt.show()

            self.assertFalse(too_different(n_rust, n_python))
            self.assertFalse(too_different(n_rust, n_jax))

    def test_threshold(self):
        """
        Test that the python and rust open double dot ground state functions return the same result.

        The threshold is set to 1, so every nearest neighbour change state is considered
        """

        for _ in range(N_ITERATIONS):
            cdd, cdd_inv, cgd = randomly_generate_matrices(3)
            vg = np.random.uniform(-10, 5, size=(N_VOLTAGES, 3))
            n_threshold_of_1 = ground_state_open_default_or_thresholded_rust(vg, cgd, cdd_inv, 1.)

            threshold = compute_threshold(cdd)
            n_threshold_not_of_1 = ground_state_open_default_or_thresholded_rust(vg, cgd, cdd_inv, threshold)

            self.assertTrue(np.allclose(n_threshold_of_1, n_threshold_not_of_1),
                            msg=f"threshold {threshold}"
                            )

    def test_python_vs_rust_one_charge(self):
        """
        Test that the python and rust isolated double dot array with one change in it ground state functions return the
        same result.

        The threshold is set to 1, so every nearest neighbour change state is considered
        """

        for _ in range(N_ITERATIONS):
            cdd, cdd_inv, cgd = randomly_generate_matrices(3)
            vg = np.random.uniform(-5, 5, size=(N_VOLTAGES, 3))
            n_rust = ground_state_closed_default_or_thresholded_rust(vg, 1, cdd=cdd, cdd_inv=cdd_inv, cgd=cgd,
                                                                     threshold=1)
            n_python = ground_state_closed_default_or_thresholded_python(vg, 1, cdd=cdd, cdd_inv=cdd_inv, cgd=cgd,
                                                                         threshold=1)
            n_jax = ground_state_closed_default_jax(vg, n_charge=1, cdd=cdd, cdd_inv=cdd_inv, cgd=cgd)

            self.assertFalse(too_different(n_rust, n_python))
            self.assertFalse(too_different(n_rust, n_jax))

    def test_python_vs_rust_two_charge(self):
        """
        Test that the python and rust isolated double dot array with two change in it ground state functions return the same result.

        The threshold is set to 1, so every nearest neighbour change state is considered
        """

        for _ in range(N_ITERATIONS):
            cdd, cdd_inv, cgd = randomly_generate_matrices(3)
            vg = np.random.uniform(-5, 5, size=(N_VOLTAGES, 3))
            n_rust = ground_state_closed_default_or_thresholded_rust(vg, 2, cdd=cdd, cdd_inv=cdd_inv, cgd=cgd,
                                                                     threshold=1)
            n_python = ground_state_closed_default_or_thresholded_python(vg, 2, cdd=cdd, cdd_inv=cdd_inv, cgd=cgd,
                                                                         threshold=1)
            n_jax = ground_state_closed_default_jax(vg, n_charge=2, cdd=cdd, cdd_inv=cdd_inv, cgd=cgd)

            self.assertFalse(too_different(n_rust, n_python))
            self.assertFalse(too_different(n_rust, n_jax))

    def test_python_vs_rust_three_charge(self):
        """
        Test that the python and rust isolated double dot array with three changes in it ground state functions return the same result.

        The threshold is set to 1, so every nearest neighbour change state is considered
        """

        for _ in range(N_ITERATIONS):
            cdd, cdd_inv, cgd = randomly_generate_matrices(3)
            vg = np.random.uniform(-5, 5, size=(N_VOLTAGES, 3))
            n_rust = ground_state_closed_default_or_thresholded_rust(vg, 3, cdd=cdd, cdd_inv=cdd_inv, cgd=cgd,
                                                                     threshold=1)
            n_python = ground_state_closed_default_or_thresholded_python(vg, 3, cdd=cdd, cdd_inv=cdd_inv, cgd=cgd,
                                                                         threshold=1)

            n_jax = ground_state_closed_default_jax(vg, n_charge=3, cdd=cdd, cdd_inv=cdd_inv, cgd=cgd)

            self.assertFalse(too_different(n_rust, n_python))
            self.assertFalse(too_different(n_rust, n_jax))
    def test_optimal_vg(self):
        """

        Test of the optimal dot voltage function which computes the dot voltages which minimise the free energy of a
        particular change configuration. For double quantum dots with two gates this means that change state will be the
        ground state.
        """

        for _ in range(N_ITERATIONS):
            cdd, cdd_inv, cgd = randomly_generate_matrices(3)
            n_charges = np.random.choice(np.arange(0, 10), size=(N_VOLTAGES, 3)).astype(int)
            vg = optimal_Vg(cdd_inv=cdd_inv, cgd=cgd, n_charges=n_charges)
            n_rust = ground_state_open_default_or_thresholded_rust(vg, cgd, cdd_inv, 1).astype(int)
            self.assertTrue(np.allclose(n_rust, n_charges))


if __name__ == '__main__':
    unittest.main()
