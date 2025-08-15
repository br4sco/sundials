import itertools
from dataclasses import dataclass
from typing import (
    Any,
    Union,
    Callable,
    TypeVar,
    List,
    Iterator,
    Tuple,
    Optional,
)
import inspect

import numpy as np
import sympy as sp
from sympy.physics.vector import dynamicsymbols
from sympy.tensor.array.expressions import ArraySymbol

import jax
import jax.numpy as jnp
from jax import config
from pydantic import BaseModel, validate_call
from pydantic_core import core_schema

import dd

config.update("jax_enable_x64", True)

t = dynamicsymbols._t


@validate_call
def mkvars(names: Union[str, List[str]], level=0):
    return dynamicsymbols(names, level=level, is_real=True)


@validate_call
def mkparams(names: Union[str, List[str]]):
    return sp.symbols(names, is_real=True)


def _diff_on_offsets(exprs: List[Any], offsets: List[int]):
    assert len(exprs) == len(offsets)
    return [
        [sp.diff(expr, (t, k)) for k in range(ofs + 1)]
        for expr, ofs in zip(exprs, offsets)
    ]


def _vectorize_residual(
    residual: Union[List[Any], List[List[Any]]],
    variables: List[List[Any]],
    parameters: List[Any],
):
    residual_list = list(itertools.chain.from_iterable(residual))
    variables_list = list(itertools.chain.from_iterable(variables))
    Y = sp.MatrixSymbol("Y", len(variables_list), 1)
    P = sp.MatrixSymbol("P", len(parameters), 1)
    subs_dict = {var: Y[i, 0] for i, var in enumerate(variables_list)} | {
        param: P[i, 0] for i, param in enumerate(parameters)
    }
    return ([expr.subs(subs_dict) for expr in residual_list], Y, P)


def _lambdify_residual(Y, P, residual: List[Any], cse: bool, modules: str):
    return sp.lambdify(
        [t, Y, P],
        sp.Array(residual, (len(residual),)),
        cse=cse,
        modules=modules,
    )


class DAE:

    @validate_call
    def __init__(
        self,
        eqn_offsets: List[int],
        var_offsets: List[int],
        variables: List[Any],
        parameters: List[Any],
        residual: List[Any],
    ):
        if not len(eqn_offsets) == len(residual):
            raise ValueError(
                "The number of offsets and residuals does not match"
            )
        if not len(var_offsets) == len(variables):
            raise ValueError(
                "The number of offsets and variables does not match"
            )

        self.eqn_offsets = eqn_offsets
        self.var_offsets = var_offsets
        self.variables = variables
        self.parameters = parameters
        self.residual = residual

    def _gen_highest_order_resfn(self, cse=True):
        vector_residual, Y, P = _vectorize_residual(
            [
                [sp.diff(expr, (t, k))]
                for expr, k in zip(self.residual, self.eqn_offsets)
            ],
            _diff_on_offsets(self.variables, self.var_offsets),
            self.parameters,
        )
        return _lambdify_residual(Y, P, vector_residual, cse=cse, modules="jax")

    def _gen_low_index_resfn(self, cse=True):
        vector_residual, Y, P = _vectorize_residual(
            _diff_on_offsets(self.residual, self.eqn_offsets),
            _diff_on_offsets(self.variables, self.var_offsets),
            self.parameters,
        )
        return _lambdify_residual(Y, P, vector_residual, cse=cse, modules="jax")

    @staticmethod
    def _wrap_lambdified_residual(resfn_inner):
        def resfn(t, Y, P):
            return resfn_inner(t, Y.reshape(-1, 1), P.reshape(-1, 1))

        return resfn

    @validate_call
    def gen_highest_order_resfn(self, cse=True):
        return DAE._wrap_lambdified_residual(self._gen_highest_order_resfn(cse))

    @validate_call
    def gen_low_index_resfn(self, cse=True):
        return DAE._wrap_lambdified_residual(self._gen_low_index_resfn(cse))

    @validate_call
    def index_variable_map(self):
        variables = list(
            itertools.chain.from_iterable(
                _diff_on_offsets(self.variables, self.var_offsets)
            )
        )
        return (variables, {var: i for i, var in enumerate(variables)})

    @validate_call
    def gen_highest_order_variables(self):
        return [
            sp.diff(var, (t, ofs))
            for var, ofs in zip(self.variables, self.var_offsets)
        ]

    @validate_call
    def low_index_resfn_dim(self):
        return (
            np.sum(np.array(self.eqn_offsets) + 1),
            np.sum(np.array(self.var_offsets) + 1),
        )


x, y, lam = mkvars("x y ʎ")
m, l, g = mkparams("m l g")
residual = [
    m * x.diff().diff() + x * lam / l,
    m * y.diff().diff() + y * lam / l + m * g,
    x**2 + y**2 - l**2,
]
eqn_offsets = [0, 0, 2]
var_offsets = [2, 2, 0]

dae = DAE(eqn_offsets, var_offsets, [x, y, lam], [m, l, g], residual)

highest_order_resfn = dae.gen_highest_order_resfn()
low_index_resfn = dae.gen_low_index_resfn()

index_map, variable_map = dae.index_variable_map()
highest_order_variables = dae.gen_highest_order_variables()

jacfn_basis_vectors = jnp.eye(len(index_map), dtype=float)
jacfn0_basis_vectors = jacfn_basis_vectors[
    [variable_map[var] for var in highest_order_variables], :
]

M, N = dae.low_index_resfn_dim()


@jax.jit
def jax_jacfn0(t, Y, P):
    def jvp(v):
        return jax.jvp(lambda Y: highest_order_resfn(t, Y, P), (Y,), (v,))[1]

    return jax.vmap(jvp)(jacfn0_basis_vectors)


def jacfn0(t, Y, J, P):
    J.np_array[:] = jax_jacfn0(t, Y.np_array, P)
    return 0


@jax.jit
def jax_resfn(t, Y, P):
    return low_index_resfn(t, Y, P)


def resfn(t, Y, R, P):
    R.np_array[0:M] = jax_resfn(t, Y.np_array, P)
    return 0


@jax.jit
def jax_jacfn(t, Y, P):
    def jvp(v):
        return jax.jvp(lambda Y: jax_resfn(t, Y, P), (Y,), (v,))[1]

    return jax.vmap(jvp)(jacfn_basis_vectors)


def jacfn(t, Y, R, J, P, tmp1, tmp2, tmp3):
    J.np_array[:, 0:M] = jax_jacfn(t, Y.np_array, P)
    return 0


def solve():
    print_trace = False

    sunctx = dd.SUNContext()
    n = len(dae.var_offsets)

    params = {m: 1.1, l: 1.2, g: 1.3}
    P = jnp.array([params[param] for param in dae.parameters], dtype=float)

    t0 = 0.0
    theta0 = np.pi / 5 + np.pi / 2
    Y = dd.NVectorSerial.from_list(
        [
            params[l] * np.sin(theta0),
            0.0,
            -params[g] * np.sin(2.0 * theta0) / 2.0,
            -params[l] * np.cos(theta0),
            0.0,
            -params[g] * np.sin(theta0) * np.sin(theta0),
            params[m] * params[g] * np.cos(theta0),
        ],
        sunctx,
    )

    J0 = dd.DDMatrixDense(dd.SUNDenseMatrix.empty(n, n, sunctx))
    jacfn0(t0, Y, J0.sunmatrix, P)

    structure = dd.DAEStructure(dae.eqn_offsets, dae.var_offsets)

    ddmem = dd.DDMem(sunctx)
    ddmem.init(structure, 0, jacfn0, J0, resfn, t0, Y)
    ddmem.set_user_data(P)
    ddmem.set_tolerances(1e-9, 1e-9)

    J = dd.SUNDenseMatrix.empty(N, N, sunctx)
    linsol = dd.SUNLinSol_Dense(Y, J, sunctx)
    ddmem.set_linear_solver(linsol, J)
    ddmem.set_jacfn(dd.DDJacFn1(jacfn))

    def print_trace_header():
        if print_trace:
            print(f'{",".join(["t"] + [f"Y[{i}]" for i in range(N)])}')

    def print_trace(t, Y):
        if print_trace:
            print(
                f"{",".join([f"{t:.20f}"] + [f"{Y[i]:.20f}" for i in range(N)])}"
            )

    ddmem.set_stop_time(100.0)
    # print_trace_header()
    status, t = dd.IDAStatus.SUCCESS, t0
    while status == dd.IDAStatus.SUCCESS:
        # print_trace(t, Y)
        pr = ddmem.pivot()
        assert pr != dd.DDPivotResult.PIVOT_FAIL
        t += 0.1
        status, t = ddmem.solve_normal(t, Y)
