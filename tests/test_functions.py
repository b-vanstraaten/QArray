import unittest

import numpy as np


class TestFunctions(unittest.TestCase):

    def test_charge_state_contrast(self):
        from qarray.functions import charge_state_dot_product

        n = np.array([[0, 1], [1, 0], [0, 0], [1, 1]])
        values = np.array([1, 2])

        with self.assertRaises(AssertionError):
            charge_state_dot_product(n, values)

        result = charge_state_dot_product(n[np.newaxis, ...], values)
        expected = np.array([2, 1, 0, 3])
        self.assertTrue(np.allclose(result, expected))

    def test_unique_last_index(self):
        from qarray.gui.helper_functions import unique_last_axis

        a = np.array([[[1, 2], [3, 4]], [[1, 2], [3, 4]], [[1, 2], [3, 4]]])
        b = np.array([[[1, 2], [3, 4]], [[1, 2], [3, 4]], [[1, 2], [3, 4]]])
        c = np.array([[[1, 2], [3, 4]], [[1, 2], [3, 4]], [[1, 2], [3, 4]]])

        d = np.array([[[1, 2], [3, 4]], [[1, 2], [3, 4]], [[1, 2], [3, 4]], [[1, 2], [3, 4]]])

        result = unique_last_axis(a)
        expected = np.array([[[1, 2], [3, 4]]])
        self.assertTrue(np.allclose(result, expected))

        result = unique_last_axis(b)
        expected = np.array([[[1, 2], [3, 4]]])
        self.assertTrue(np.allclose(result, expected))

        result = unique_last_axis(c)
        expected = np.array([[[1, 2], [3, 4]]])
        self.assertTrue(np.allclose(result, expected))

        result = unique_last_axis(d)
        expected = np.array([[[1, 2], [3, 4]]])
        self.assertTrue(np.allclose(result, expected))
