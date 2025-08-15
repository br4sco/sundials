import copy
from dataclasses import dataclass

from dd import (
    SUNContext,
    NVector,
    NVectorSerial,
    SUNDenseMatrix,
    DDMatrixDense,
    DAEStructure,
    DDMem,
    SUNLinSol_Dense,
    DDJacFn1,
    IDAStatus,
    DDPivotResult,
)

import numpy as np
import pytest
from hypothesis import given, strategies as st

rng = np.random.default_rng()

WRITE_DAE_TRACE_FILE = False

## --- Test NVector ---


def np_array_equal_ignore_nan(a, b):
    return np.array_equal(a[~np.isnan(a)], b[~np.isnan(b)])


@given(st.integers(min_value=0, max_value=1000))
def test_nvector_serial_empty(size):
    nvector = NVectorSerial.empty(size, SUNContext())
    assert len(nvector) == size


@given(st.lists(st.floats(), max_size=1000))
def test_nvector_serial_from_list(float_list):
    nvector = NVectorSerial.from_list(float_list, SUNContext())
    assert len(nvector) == len(float_list)


@given(st.lists(st.floats(), max_size=1000))
def test_nvector_serial_from_array(float_list):
    np_array = np.asarray(float_list)
    nvector = NVectorSerial.from_list(np_array, SUNContext())

    assert len(nvector) == len(float_list)
    assert np_array_equal_ignore_nan(np_array, nvector.np_array)


max_value = 10


@given(st.lists(st.floats(max_value=max_value), max_size=1000))
def test_nvector_serial_deepcopy(float_list):
    np_array = np.asarray(float_list)
    nvector = NVectorSerial.from_list(np_array, SUNContext())
    nvector_copy = copy.deepcopy(nvector)

    assert np_array_equal_ignore_nan(nvector.np_array, nvector_copy.np_array)
    size = len(np_array)
    if size > 0:
        nvector.np_array[rng.integers(low=0, high=size)] = max_value + 1
        assert not np_array_equal_ignore_nan(
            nvector.np_array, nvector_copy.np_array
        )


## --- Test DDMem ---

T0 = 0.0
TSTOP = 100.0

## Lotka-Volterra Model

lv_eqn_offsets = [0, 0]
lv_var_offsets = [1, 1]


@dataclass
class LVParams:
    prey_growth: float
    prey_death: float
    predator_growth: float
    predator_death: float


def lv_resfn(t, Y, R, p):
    x = Y[0]
    dx = Y[1]
    y = Y[2]
    dy = Y[3]
    R[0] = dx - p.prey_growth * x + p.prey_death * x * y
    R[1] = dy - p.predator_death * x * y + p.predator_death * y
    return 0


def lv_jacfn0(t, Y, J, p):
    J[0, 0] = 1
    J[1, 1] = 1
    return 0


def lv_jacfn(t, Y, R, J, p, tmp1, tmp2, tmp3):
    x = Y[0]
    y = Y[2]
    J[0, 0] = -p.prey_growth + p.prey_death * y
    J[0, 1] = 1
    J[0, 2] = p.prey_death * x
    J[1, 0] = -p.predator_death * y
    J[1, 2] = -p.predator_death * x + p.predator_death
    J[1, 3] = 1
    return 0


lv_params = LVParams(1.5, 1, 1, 3)

lv_Y0 = [
    1,
    lv_params.prey_growth - lv_params.prey_death,
    1,
    lv_params.prey_death - lv_params.predator_growth,
]

p_eqn_offsets = [0, 0, 2]
p_var_offsets = [2, 2, 0]


@dataclass
class PParams:
    m: float
    l: float
    g: float


def p_resfn(t, Y, R, p):
    x = Y[0]
    dx = Y[1]
    ddx = Y[2]
    y = Y[3]
    dy = Y[4]
    ddy = Y[5]
    lam = Y[6]
    R[0] = p.m * ddx + lam / p.l * x
    R[1] = p.m * ddy + lam / p.l * y + p.m * p.g
    R[2] = x**2 + y**2 - p.l**2
    R[3] = 2 * (x * dx + y * dy)
    R[4] = 2 * (x * ddx + dx**2 + y * ddy + dy**2)
    return 0


def p_jacfn0(t, Y, J, p):
    x = Y[0]
    y = Y[3]
    J[0, 0] = p.m
    J[0, 2] = x / p.l
    J[1, 1] = p.m
    J[1, 2] = y / p.l
    J[2, 0] = 2 * x
    J[2, 1] = 2 * y
    return 0


def p_jacfn(t, Y, R, J, p, tmp1, tmp2, tmp3):
    x = Y[0]
    dx = Y[1]
    ddx = Y[2]
    y = Y[3]
    dy = Y[4]
    ddy = Y[5]
    lam = Y[6]
    # column 0
    J[0, 0] = lam / p.l
    J[2, 0] = 2 * x
    J[3, 0] = 2 * dx
    J[4, 0] = 2 * ddx
    # column 1
    J[3, 1] = 2 * x
    J[4, 1] = 4 * dx
    # column 2
    J[0, 2] = p.m
    J[4, 2] = 2 * x
    # column 3
    J[1, 3] = lam / p.l
    J[2, 3] = 2 * y
    J[3, 3] = 2 * dy
    J[4, 3] = 2 * ddy
    # column 4
    J[3, 4] = 2 * y
    J[4, 4] = 4 * dy
    # column 6
    J[1, 5] = p.m
    J[4, 5] = 2 * y
    # column 6
    J[0, 6] = x / p.l
    J[1, 6] = y / p.l
    return 0


p_params = PParams(1.1, 1.2, 1.3)

p_theta0 = np.pi / 5 + np.pi / 2
p_Y0 = [
    p_params.l * np.sin(p_theta0),
    0,
    -p_params.g * np.sin(2 * p_theta0) / 2,
    -p_params.l * np.cos(p_theta0),
    0,
    -p_params.g * np.sin(p_theta0) * np.sin(p_theta0),
    p_params.m * p_params.g * np.cos(p_theta0),
]

MODELS_TO_TEST = [
    # (
    #     lv_eqn_offsets,
    #     lv_var_offsets,
    #     lv_params,
    #     lv_resfn,
    #     lv_jacfn0,
    #     lv_jacfn,
    #     lv_Y0,
    #     np.array(
    #         [
    #             1.02349899180322134740,
    #             -0.65638477491231705940,
    #             2.14131453002790461682,
    #             0.15095619776773375187,
    #         ]
    #     ),
    # ),
    (
        p_eqn_offsets,
        p_var_offsets,
        p_params,
        p_resfn,
        p_jacfn0,
        p_jacfn,
        p_Y0,
        np.array(
            [
                -1.14493821778094639896,
                -0.28401660598107553168,
                0.34388924963112110778,
                0.35932781337462627036,
                -0.90497160133034260099,
                -1.40792632317967858491,
                0.39647013477536663384,
            ]
        ),
    ),
]


@pytest.mark.parametrize(
    "eqn_offsets, var_offsets, params, resfn, jacfn0, jacfn, Y0, Ystop",
    MODELS_TO_TEST,
)
def test_solver_accuracy(
    eqn_offsets, var_offsets, params, resfn, jacfn0, jacfn, Y0, Ystop
):
    n = len(eqn_offsets)
    N = sum([d + 1 for d in var_offsets])

    if WRITE_DAE_TRACE_FILE:
        trace_file = open(f"test_trace_{resfn.__name__}.csv", "w")

    def maybe_write_trace_header():
        if WRITE_DAE_TRACE_FILE:
            trace_file.write(
                f'{",".join(["t"] + [f"Y[{i}]" for i in range(N)])}\n'
            )

    def maybe_write_trace(t, Y):
        if WRITE_DAE_TRACE_FILE:
            trace_file.write(
                f"{",".join([f"{t:.20f}"] + [f"{Y[i]:.20f}" for i in range(N)])}\n"
            )

    structure = DAEStructure(eqn_offsets, var_offsets)
    sunctx = SUNContext()
    ddmem = DDMem(sunctx)
    J0 = DDMatrixDense(SUNDenseMatrix.empty(n, n, sunctx))
    Y = NVectorSerial.from_list(Y0, sunctx)

    R = NVectorSerial.empty(N, sunctx)
    resfn(T0, Y, R, params)
    print(Y)
    print(R)
    assert False

    jacfn0(T0, Y, J0.sunmatrix, params)
    ddmem.init(structure, 0, jacfn0, J0, resfn, T0, Y)
    ddmem.set_user_data(params)
    ddmem.set_stop_time(TSTOP)
    ddmem.set_tolerances(1e-9, 1e-9)

    J = SUNDenseMatrix.empty(N, N, sunctx)
    linsol = SUNLinSol_Dense(Y, J, sunctx)
    ddmem.set_linear_solver(linsol, J)
    ddmem.set_jacfn(DDJacFn1(jacfn))

    maybe_write_trace_header()
    maybe_write_trace(T0, Y)

    status, t = IDAStatus.SUCCESS, T0
    while status == IDAStatus.SUCCESS:
        pr = ddmem.pivot()
        assert pr != DDPivotResult.PIVOT_FAIL
        t += 0.1
        status, t = ddmem.solve_normal(t, Y)
        maybe_write_trace(t, Y)

    assert np.allclose(Y.np_array, Ystop, rtol=1e-09, atol=1e-09)
