"""
This module contains the functions for computing the ground state of an open array using jax.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jaxopt import BoxOSQP

from .charge_configuration_generators import open_charge_configurations_jax
from ..typing_classes import VectorList, CddInv, Cgd, Vector

qp = BoxOSQP(check_primal_dual_infeasability=False, verbose=False)


def compute_analytical_solution_open(cgd: Cgd, vg: Vector) -> Vector | np.ndarray:
    """
    Computes the analytical solution for the continuous charge distribution for an open array.
    :param cgd: the gate to dot capacitance matrix
    :param vg: the gate voltage coordinate vector
    :return: the continuous charge distribution
    """
    return cgd @ vg


def numerical_solver_open(cdd_inv: CddInv, cgd: Cgd, vg: VectorList) -> VectorList | np.ndarray:
    """
    Solve the quadratic program for the continuous charge distribution for an open array.
    :param cdd_inv: the inverse of the dot to dot capacitance matrix
    :param cgd: the gate to dot capacitance matrix
    :param vg: the gate voltage coordinate vector
    :return: the continuous charge distribution
    """
    n_dot = cdd_inv.shape[0]
    P = cdd_inv
    q = -cdd_inv @ cgd @ vg
    A = jnp.eye(n_dot)
    l = jnp.zeros(n_dot)
    u = jnp.full(n_dot, fill_value=jnp.inf)
    params = qp.run(params_obj=(P, q), params_eq=A, params_ineq=(l, u)).params
    return params.primal[0]


def compute_continuous_solution_open(cdd_inv: CddInv, cgd: Cgd, vg):
    """
    Computes the continuous solution for an open array. If the analytical solution is valid, it is returned, otherwise
    the numerical solution is returned.
    :param cdd_inv: the inverse of the dot to dot capacitance matrix
    :param cgd: the gate to dot capacitance matrix
    :param vg: the gate voltage coordinate vector
    :return: the continuous charge distribution
    """
    analytical_solution = compute_analytical_solution_open(cgd, vg)
    return jax.lax.cond(
        jnp.all(analytical_solution >= 0),
        lambda: analytical_solution,
        lambda: numerical_solver_open(cdd_inv=cdd_inv, cgd=cgd, vg=vg),
    )



def ground_state_open_jax(vg: VectorList, cgd: Cgd, cdd_inv: CddInv) -> VectorList:
    """
    A jax implementation for the ground state function that takes in numpy arrays and returns numpy arrays.
    :param vg: the gate voltage coordinate vectors to evaluate the ground state at
    :param cgd: the gate to dot capacitance matrix
    :param cdd_inv: the inverse of the dot to dot capacitance matrix
    :return: the lowest energy charge configuration for each gate voltage coordinate vector
    """

    f = partial(_ground_state_open_0d, cgd=cgd, cdd_inv=cdd_inv)
    match jax.local_device_count():
        case 0:
            raise ValueError('Must have at least one device')
        case 1:
            f = jax.vmap(f)
        case _:
            f = jax.pmap(f)
    return f(vg)


@jax.jit
def _ground_state_open_0d(vg: jnp.ndarray, cgd: jnp.ndarray, cdd_inv: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the ground state for an open array.
    :param vg: the gate voltage coordinate vector
    :param cgd: the gate to dot capacitance matrix
    :param cdd_inv: the inverse of the dot to dot capacitance matrix
    :return: the lowest energy charge configuration
    """
    n_continuous = compute_continuous_solution_open(cdd_inv=cdd_inv, cgd=cgd, vg=vg)
    n_continuous = jnp.clip(n_continuous, 0, None)
    # eliminating the possibly of negative numbers of change carriers
    return compute_argmin_open(n_continuous=n_continuous, cdd_inv=cdd_inv, cgd=cgd, Vg=vg)


def compute_argmin_open(n_continuous, cdd_inv, cgd, Vg):
    """
    Computes the lowest energy charge configuration for an open array.
    :param n_continuous: the continuous charge distribution
    :param cdd_inv: the inverse of the dot to dot capacitance matrix
    :param cgd: the gate to dot capacitance matrix
    :param Vg: the gate voltage coordinate vector
    :return: the lowest energy charge configuration
    """

    # computing the remainder
    n_list = open_charge_configurations_jax(n_continuous)
    v_dash = cgd @ Vg
    # computing the free energy of the change configurations
    F = jnp.einsum('...i, ij, ...j', n_list - v_dash, cdd_inv, n_list - v_dash)
    # returning the lowest energy change configuration
    return n_list[jnp.argmin(F), :]