"""
Tests to check the capacitance model_threshold_1 works for double dot arrays
"""

from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np

from qarray import (optimal_Vg,
                    GateVoltageComposer, charge_state_changes)
from .GLOBAL_OPTIONS import N_ITERATIONS, N_VOLTAGES
from .helper_functions import randomly_generate_model


class ThresholdTests(TestCase):

    def test_threshold_double_dot_open_python(self):
        """
        Test that the python and rust open double dot ground state functions return the same result.

        The threshold is set to 1, so every nearest neighbour change state is considered
        """

        n_dot = 2
        n_gate = 2

        models = randomly_generate_model(n_dot, n_gate, N_ITERATIONS)
        voltage_composer = GateVoltageComposer(n_gate=n_gate)

        for model in models:
            model.implementation = 'python'
            model.algorithm = 'thresholded'
            vg = voltage_composer.do2d(1, -5, 5, N_VOLTAGES, 2, -5, 5, N_VOLTAGES)
            n_threshold_not_of_1 = model.ground_state_open(vg)

            model.threshold = 1.
            n_threshold_of_1 = model.ground_state_open(vg)
            self.assertTrue(np.allclose(n_threshold_of_1, n_threshold_not_of_1))

    def test_threshold_double_dot_closed_python(self):
        """
        Test that the python and rust open double dot ground state functions return the same result.

        The threshold is set to 1, so every nearest neighbour change state is considered
        """

        n_dot = 2
        n_gate = 2

        models = randomly_generate_model(n_dot, n_gate, N_ITERATIONS)
        voltage_composer = GateVoltageComposer(n_gate=n_gate)

        for model in models:
            model.implementation = 'python'
            model.algorithm = 'thresholded'
            vg = voltage_composer.do2d(1, -10, 5, N_VOLTAGES, 2, -10, 5, N_VOLTAGES)
            n_threshold_not_of_1 = model.ground_state_closed(vg, n_charges=5)

            model.threshold = 1.
            n_threshold_of_1 = model.ground_state_closed(vg, n_charges=5)
            self.assertTrue(np.allclose(n_threshold_of_1, n_threshold_not_of_1))

    def test_threshold_double_dot_open_jax(self):
        """
        Test that the python and rust open double dot ground state functions return the same result.

        The threshold is set to 1, so every nearest neighbour change state is considered
        """

        n_dot = 2
        n_gate = 2

        models = randomly_generate_model(n_dot, n_gate, N_ITERATIONS)
        voltage_composer = GateVoltageComposer(n_gate=n_gate)

        for model in models:
            model.implementation = 'jax'
            model.algorithm = 'default'
            vg = voltage_composer.do2d(1, -5, 5, N_VOLTAGES, 2, -5, 5, N_VOLTAGES)
            n_threshold_not_of_1 = model.ground_state_open(vg)

            model.threshold = 1.
            n_threshold_of_1 = model.ground_state_open(vg)
            self.assertTrue(np.allclose(n_threshold_of_1, n_threshold_not_of_1))

    def test_threshold_double_dot_closed_jax(self):
        """
        Test that the python and rust open double dot ground state functions return the same result.

        The threshold is set to 1, so every nearest neighbour change state is considered
        """

        n_dot = 2
        n_gate = 2

        models = randomly_generate_model(n_dot, n_gate, N_ITERATIONS)
        voltage_composer = GateVoltageComposer(n_gate=n_gate)

        for model in models:
            model.implementation = 'jax'
            model.algorithm = 'default'
            vg = voltage_composer.do2d(1, -10, 5, N_VOLTAGES, 2, -10, 5, N_VOLTAGES)
            n_threshold_not_of_1 = model.ground_state_closed(vg, n_charges=5)

            model.threshold = 1.
            n_threshold_of_1 = model.ground_state_closed(vg, n_charges=5)
            self.assertTrue(np.allclose(n_threshold_of_1, n_threshold_not_of_1))


    def test_threshold_double_dot_open(self):
        """
        Test that the python and rust open double dot ground state functions return the same result.

        The threshold is set to 1, so every nearest neighbour change state is considered
        """

        n_dot = 2
        n_gate = 2

        models = randomly_generate_model(n_dot, n_gate, N_ITERATIONS)
        voltage_composer = GateVoltageComposer(n_gate=n_gate)

        for model in models:
            vg = voltage_composer.do2d(1, -5, 5, N_VOLTAGES, 2, -5, 5, N_VOLTAGES)
            n_threshold_not_of_1 = model.ground_state_open(vg)

            model.threshold = 1.
            n_threshold_of_1 = model.ground_state_open(vg)
            self.assertTrue(np.allclose(n_threshold_of_1, n_threshold_not_of_1))

    def test_threshold_double_dot_closed(self):
        """
        Test that the python and rust open double dot ground state functions return the same result.

        The threshold is set to 1, so every nearest neighbour change state is considered
        """

        n_dot = 2
        n_gate = 2

        models = randomly_generate_model(n_dot, n_gate, N_ITERATIONS)
        voltage_composer = GateVoltageComposer(n_gate=n_gate)

        for model in models:
            vg = voltage_composer.do2d(1, -10, 5, N_VOLTAGES, 2, -10, 5, N_VOLTAGES)
            n_threshold_not_of_1 = model.ground_state_closed(vg, n_charges=5)

            model.threshold = 1.
            n_threshold_of_1 = model.ground_state_closed(vg, n_charges=5)
            self.assertTrue(np.allclose(n_threshold_of_1, n_threshold_not_of_1))

    def test_threshold_triple_dot(self):
        """
        Test that the python and rust open double dot ground state functions return the same result.

        The threshold is set to 1, so every nearest neighbour change state is considered
        """
        n_dot = 3
        n_gate = 3

        models = randomly_generate_model(n_dot, n_gate, N_ITERATIONS)
        voltage_composer = GateVoltageComposer(n_gate=n_gate)

        for model in models:
            vg = voltage_composer.do2d(1, -10, 5, N_VOLTAGES, 2, -10, 5, N_VOLTAGES)
            n_threshold_not_of_1 = model.ground_state_open(vg)

            model.threshold = 1.
            n_threshold_of_1 = model.ground_state_open(vg)

            if not np.allclose(n_threshold_of_1, n_threshold_not_of_1):
                fig, ax = plt.subplots(1, 3)
                ax[0].imshow(charge_state_changes(n_threshold_of_1), aspect='auto', cmap='Greys')
                ax[1].imshow(charge_state_changes(n_threshold_not_of_1), aspect='auto', cmap='Greys')

                diff = np.abs(n_threshold_of_1 - n_threshold_not_of_1).sum(axis=-1)
                ax[2].imshow(diff, aspect='auto')
                plt.show()

            self.assertTrue(np.allclose(n_threshold_of_1, n_threshold_not_of_1))

    def test_threshold_triple_dot_closed(self):
        """
        Test that the python and rust open double dot ground state functions return the same result.

        The threshold is set to 1, so every nearest neighbour change state is considered
        """

        n_dot = 3
        n_gate = 3

        models = randomly_generate_model(n_dot, n_gate, N_ITERATIONS)
        voltage_composer = GateVoltageComposer(n_gate=n_gate)

        for model in models:
            vg = voltage_composer.do2d(1, -20, 5, N_VOLTAGES, 3, -20, 5, N_VOLTAGES)
            n_threshold_not_of_1 = model.ground_state_closed(vg, n_charges=2)

            model.threshold = 1.
            n_threshold_of_1 = model.ground_state_closed(vg, n_charges=2)

            if not np.allclose(n_threshold_of_1, n_threshold_not_of_1):
                fig, ax = plt.subplots(1, 3)
                ax[0].imshow(charge_state_changes(n_threshold_of_1), aspect='auto', cmap='Greys')
                ax[1].imshow(charge_state_changes(n_threshold_not_of_1), aspect='auto', cmap='Greys')

                diff = np.abs(n_threshold_of_1 - n_threshold_not_of_1).sum(axis=-1)
                ax[2].imshow(diff, aspect='auto')
                plt.show()


    def test_threshold_quadruple_dot(self):
        """
        Test that the python and rust open double dot ground state functions return the same result.

        The threshold is set to 1, so every nearest neighbour change state is considered
        """
        n_dot = 4
        n_gate = 4

        models = randomly_generate_model(n_dot, n_gate, N_ITERATIONS)
        voltage_composer = GateVoltageComposer(n_gate=n_gate)

        for model in models:
            vg = voltage_composer.do2d(1, -5, 0, N_VOLTAGES, 4, -5, 0, N_VOLTAGES)
            n_threshold_not_of_1 = model.ground_state_open(vg)

            model.threshold = 1.
            n_threshold_of_1 = model.ground_state_open(vg)

            if not np.allclose(n_threshold_of_1, n_threshold_not_of_1):
                fig, ax = plt.subplots(1, 3)
                ax[0].imshow(charge_state_changes(n_threshold_of_1), aspect='auto', cmap='Greys')
                ax[1].imshow(charge_state_changes(n_threshold_not_of_1), aspect='auto', cmap='Greys')

                diff = np.abs(n_threshold_of_1 - n_threshold_not_of_1).sum(axis=-1)
                ax[2].imshow(diff, aspect='auto')
                plt.show()

            self.assertTrue(np.allclose(n_threshold_of_1, n_threshold_not_of_1))

    def test_threshold_quadruple_dot_closed(self):
        """
        Test that the python and rust open double dot ground state functions return the same result.

        The threshold is set to 1, so every nearest neighbour change state is considered
        """
        n_dot = 4
        n_gate = 4

        models = randomly_generate_model(n_dot, n_gate, N_ITERATIONS)
        voltage_composer = GateVoltageComposer(n_gate=n_gate)

        for model in models:
            vg = voltage_composer.do2d(1, -4, 0, N_VOLTAGES, 4, -4, 0, N_VOLTAGES)
            n_threshold_not_of_1 = model.ground_state_closed(vg, 4)

            model.threshold = 1.
            n_threshold_of_1 = model.ground_state_closed(vg, 4)

            if not np.allclose(n_threshold_of_1, n_threshold_not_of_1):
                fig, ax = plt.subplots(1, 3)

                ax[0].imshow(charge_state_changes(n_threshold_of_1), aspect='auto', cmap='Greys')
                ax[1].imshow(charge_state_changes(n_threshold_not_of_1), aspect='auto', cmap='Greys')

                diff = np.abs(n_threshold_of_1 - n_threshold_not_of_1).sum(axis=-1)
                ax[2].imshow(diff, aspect='auto')
                plt.show()

            self.assertTrue(np.allclose(n_threshold_of_1, n_threshold_not_of_1))


    def test_threshold_five_dot(self):
        """
        Test that the python and rust open double dot ground state functions return the same result.

        The threshold is set to 1, so every nearest neighbour change state is considered
        """
        n_dot = 5
        n_gate = 5

        models = randomly_generate_model(n_dot, n_gate, N_ITERATIONS)
        voltage_composer = GateVoltageComposer(n_gate=n_gate)

        for model in models:
            vg = voltage_composer.do2d(1, -5, 5, N_VOLTAGES, 5, -5, 5, N_VOLTAGES)
            vg = vg + optimal_Vg(model.cdd_inv, model.cgd, np.random.uniform(1, 5, size=n_gate))

            n_threshold_not_of_1 = model.ground_state_open(vg)

            model.threshold = 1.
            n_threshold_of_1 = model.ground_state_open(vg)

            if not np.allclose(n_threshold_of_1, n_threshold_not_of_1):
                fig, ax = plt.subplots(1, 3)
                ax[0].imshow(charge_state_changes(n_threshold_of_1), aspect='auto', cmap='Greys', origin='lower')
                ax[1].imshow(charge_state_changes(n_threshold_not_of_1), aspect='auto', cmap='Greys', origin='lower')

                diff = np.abs(n_threshold_of_1 - n_threshold_not_of_1).sum(axis=-1)
                ax[2].imshow(diff, aspect='auto', origin='lower')
                plt.show()

            self.assertTrue(np.allclose(n_threshold_of_1, n_threshold_not_of_1))

    def test_threshold_quadruple_dot_closed(self):
        """
        Test that the python and rust open double dot ground state functions return the same result.

        The threshold is set to 1, so every nearest neighbour change state is considered
        """
        n_dot = 5
        n_gate = 5

        models = randomly_generate_model(n_dot, n_gate, N_ITERATIONS)
        voltage_composer = GateVoltageComposer(n_gate=n_gate)

        for model in models:
            print(model.threshold)
            vg = voltage_composer.do2d(1, -20, 0, N_VOLTAGES, 5, -20, 0, N_VOLTAGES)
            n_threshold_not_of_1 = model.ground_state_closed(vg, 4)

            model.threshold = 1.
            n_threshold_of_1 = model.ground_state_closed(vg, 4)

            if not np.allclose(n_threshold_of_1, n_threshold_not_of_1):
                fig, ax = plt.subplots(1, 3)

                ax[0].imshow(charge_state_changes(n_threshold_of_1), aspect='auto', cmap='Greys')
                ax[1].imshow(charge_state_changes(n_threshold_not_of_1), aspect='auto', cmap='Greys')

                diff = np.abs(n_threshold_of_1 - n_threshold_not_of_1).sum(axis=-1)
                ax[2].imshow(diff, aspect='auto')
                plt.show()

            self.assertTrue(np.allclose(n_threshold_of_1, n_threshold_not_of_1))
