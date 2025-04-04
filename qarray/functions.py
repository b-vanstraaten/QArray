"""
This file contains functions that are used elesewhere in the code.
"""

from warnings import warn

import numpy as np
import scipy.linalg

from .qarray_types import (CddInv, Cgd_holes, Cdd, VectorList, Tetrad,
                           Vector)


def compute_optimal_virtual_gate_matrix(
        cdd_inv: CddInv, cgd: Cgd_holes, rcond: float = 1e-4) -> np.ndarray:
    """
    Function to compute the optimal virtual gate matrix.

    :param cdd_inv: the inverse of the dot to dot capacitance matrix
    :param cgd: the dot to gate capacitance matrix
    :param rcond: the rcond parameter for the pseudo inverse
    :return: the optimal virtual gate matrix

    """
    n_dot = cdd_inv.shape[0]
    n_gate = cgd.shape[1]
    virtual_gate_matrix = -np.linalg.pinv(cdd_inv @ cgd, rcond=rcond)

    # if the number of dots is less than the number of gates then we pad with zeros
    if n_dot < n_gate:
        virtual_gate_matrix = np.pad(virtual_gate_matrix, ((0, 0), (0, n_gate - n_dot)), mode='constant')

    return virtual_gate_matrix



def charge_state_to_scalar(n: Tetrad | np.ndarray) -> int:
    """
    Function to convert the charge state to a unique index, using the binary representation.

    Function to convert the charge state to a random index.
    This is useful for plotting the change state. So that the
    vector of charges in converted to a random integer value.
    As such nearby charge states will have similar colors.

    For plotting we recommend taking, if the output of this function is z, then
    we reccomend plotting np.log2(z + 1) to get a better color map.

    :param n: the charge state of the dots of shape (n_dot)

    :return: the unique index
    """

    if not isinstance(n, Tetrad):
        n = Tetrad(n)

    return np.sum(2 ** np.arange(n.shape[-1])[np.newaxis, np.newaxis] * n, axis = -1)


def charge_state_dot_product(n: Tetrad | np.ndarray, values: Vector | np.ndarray) -> VectorList:
    """
    Function which computes the dot product between the dot change state and the
    values in "values", thereby assigning a scalar value to each charge state.

    :param n: the dot occupation vector of shape (..., n_dot)
    :param values: the values to assign to each charge state of shape (n_dot)

    :return: the dot product between the dot change state and the values in "values" of shape (...)
    """

    if not isinstance(n, Tetrad):
        n = Tetrad(n)

    if not isinstance(values, Vector):
        values = Vector(values)

    values = values[np.newaxis, np.newaxis, :]
    return (n * values).sum(axis=-1)


def charge_state_changes(n: Tetrad | np.ndarray, dot = 'any') -> np.ndarray:
    """
    If the user passes the np.ndarray n array of shape (Nx, Ny, N_dot) which is outputted from a do2d funciton
    then this function will return a boolean array of shape (Nx-1, Ny-1, N_dot) which will be true if the
    charge state of the dot changes between in either the x or y direction.

    :param n: the dot occupation rank 3 tensor of shape (res_y, res_x, n_dot)

    :return: the boolean array of whether the charge state changes (rank 2 tensor of shape (res_y - 1, res_x - 1))

    """

    # ensure n is of rank three
    if not isinstance(n, Tetrad):
        n = Tetrad(n)

    if dot == 'any':
        change_in_x = np.logical_not(np.isclose(n[1:, :-1, :], n[:-1, :-1, :], atol=1e-3)).any(axis=-1)
        change_in_y = np.logical_not(np.isclose(n[:-1, 1:, :], n[:-1, :-1, :], atol=1e-3)).any(axis=-1)
    else:
        change_in_x = np.logical_not(np.isclose(n[1:, :-1, dot], n[:-1, :-1, dot], atol=1e-3))
        change_in_y = np.logical_not(np.isclose(n[:-1, 1:, dot], n[:-1, :-1, dot], atol=1e-3))

        # Create arrays of the same shape as n[:, :, 0] (2D grid), initialised with False
    change = np.zeros(n.shape[:2], dtype=bool)

    # Set the top-left part where the comparison was valid
    change[:-1, :-1] = np.logical_or(change_in_x, change_in_y)

    return change


def dot_occupation_changes(n: Tetrad | np.ndarray) -> np.ndarray:
    """

    If the user passes the np.ndarray n array of shape (Nx, Ny, N_dot) which is outputted from a do2d funciton
    then this function will return a boolean array of shape (Nx-1, Ny-1, N_dot) which will be true if the
    charge state of the dot changes between in either the x or y direction.

    :param n: the dot occupation rank 3 tensor of shape (res_y, res_x, n_dot)

    :return: the boolean array of whether the charge state changes (rank 2 tensor of shape (res_y - 1, res_x - 1))

    """
    warn("This function is deprecated, use charge_state_changes instead", DeprecationWarning)
    return charge_state_changes(n)


def optimal_Vg(cdd_inv: CddInv, cgd: Cgd_holes, n_charges: VectorList, rcond: float = 1e-3):
    '''
    calculate voltage that minimises charge state's energy

    :param cdd_inv: the inverse of the dot to dot capacitance matrix
    :param cgd: the dot to gate capacitance matrix
    :param n_charges: the charge state of the dots of shape (n_dot)
    :return:
    '''
    R = np.linalg.cholesky(cdd_inv).T
    M = np.linalg.pinv(R @ cgd, rcond=rcond) @ R
    return np.einsum('ij, ...j', M, n_charges)


def compute_threshold(cdd: Cdd) -> float:
    """
    Function to compute the threshold for the ground state calculation
    :param cdd:
    :return:
    """
    cdd_diag = np.diag(cdd)
    c = (cdd - np.diag(cdd_diag)) @ np.linalg.inv(np.diag(cdd_diag))
    return scipy.linalg.norm(c, ord='fro')
