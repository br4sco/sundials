#include <idas/idas.h>
#include <stdlib.h>
#include <string.h>
#include <sundials/sundials_core.h>
#include <sunlinsol/sunlinsol_dense.h>
#include <sunmatrix/sunmatrix_dense.h>

#include "dd.h"
#include "dd_err.h"
#include "dd_staged_pivot_impl.h"
#include "dd_staged_pivot_matrix.h"
#include "models.h"
#include "static_info.h"
#include "test.h"

#define ZERO SUN_RCONST(0.0)

/* (b): unconditionally proposes an all-algebraic spec, always reporting a
   change. */
static int AlwaysChangeUpdate(DDStatePivot self,
                              sunrealtype t,
                              N_Vector Y,
                              void* user_data,
                              uint8_t* spec,
                              sunbooleantype* spec_changed)
{
  (void)self;
  (void)t;
  (void)Y;
  (void)user_data;
  memset(spec, 0, PENDULUM_N * sizeof(*spec));
  *spec_changed = SUNTRUE;
  return 0;
}

/* (c): always fails. */
static int AlwaysFailUpdate(DDStatePivot self,
                            sunrealtype t,
                            N_Vector Y,
                            void* user_data,
                            uint8_t* spec,
                            sunbooleantype* spec_changed)
{
  (void)self;
  (void)t;
  (void)Y;
  (void)user_data;
  (void)spec;
  (void)spec_changed;
  return -1;
}

int main(void)
{
  SUNContext sunctx;
  TEST_ASSERT(SUNContext_Create(SUN_COMM_NULL, &sunctx) == SUN_SUCCESS);

  /* (b) and (c): concrete DDStatePivot mocks built on the empty shell. */
  DDStatePivot always_change = DDSPNewEmpty();
  TEST_ASSERT(always_change != NULL);
  always_change->ops->update = AlwaysChangeUpdate;

  DDStatePivot always_fail = DDSPNewEmpty();
  TEST_ASSERT(always_fail != NULL);
  always_fail->ops->update = AlwaysFailUpdate;

  PendulumData data = {.m     = SUN_RCONST(1.0),
                       .param = {SUN_RCONST(1.0), SUN_RCONST(1.0)}};

  DDStaticInfo si = DDStaticInfoCreate(PENDULUM_N,
                                       PENDULUM_C,
                                       PENDULUM_D,
                                       PENDULUM_VAR_IDX_MAP,
                                       NULL,
                                       NULL);
  TEST_ASSERT(si != NULL);

  /* Compute the correct initial spec by pivoting, same as every other
     driver -- don't hand-guess it. */
  uint8_t spec[3];
  {
    N_Vector Y0 = N_VNew_Serial(si->N_all_orders, sunctx);
    TEST_ASSERT(Y0 != NULL);
    PendulumY0(&data, ZERO, Y0);

    SUNMatrix J0 = SUNDenseMatrix(si->N, si->N, sunctx);
    TEST_ASSERT(J0 != NULL);
    DDStagedPivotMatrix pJ0 = DDStagedPivotMatWrapDense(J0);
    TEST_ASSERT(pJ0 != NULL);

    DDStagedPivot pm = DDSPCreateStaged(sunctx, si, pJ0, PendulumJacf0);
    TEST_ASSERT(pm != NULL);

    sunbooleantype spec_changed;
    TEST_ASSERT(DDSPPivotStaged(pm, ZERO, ZERO, Y0, &data, &spec_changed) ==
                SUN_SUCCESS);
    memcpy(spec, pm->spec, sizeof(spec));

    DDSPDestroyStaged(&pm);
    DDStagedPivotMatDestroy(pJ0);
    SUNMatDestroy(J0);
    N_VDestroy(Y0);
  }

  /* (a) No DDStatePivot registered -- DDSolve() behaves exactly as before. */
  {
    N_Vector Y = N_VNew_Serial(si->N_all_orders, sunctx);
    TEST_ASSERT(Y != NULL);
    PendulumY0(&data, ZERO, Y);

    DDMem dd_mem = DDCreate(sunctx);
    TEST_ASSERT(dd_mem != NULL);
    TEST_ASSERT(DDInit(dd_mem, si, PendulumRes, spec, ZERO, Y) == IDA_SUCCESS);
    TEST_ASSERT(DDSetUserData(dd_mem, &data) == IDA_SUCCESS);
    TEST_ASSERT(DDSSTolerances(dd_mem, SUN_RCONST(1.0e-9), SUN_RCONST(1.0e-9)) ==
                IDA_SUCCESS);

    SUNMatrix J = SUNDenseMatrix(si->N_all_orders, si->N_all_orders, sunctx);
    TEST_ASSERT(J != NULL);
    SUNLinearSolver LS = SUNLinSol_Dense(Y, J, sunctx);
    TEST_ASSERT(LS != NULL);
    TEST_ASSERT(DDSetLinearSolver(dd_mem, LS, J) == IDA_SUCCESS);
    TEST_ASSERT(
      DDSetJacFn(dd_mem,
                 (DDLsJacFn){.id = DD_JAC_1, .fn.jacfn1 = PendulumJacfn_Dense}) ==
      IDA_SUCCESS);

    sunrealtype tret;
    TEST_ASSERT(DDSolve(dd_mem, SUN_RCONST(0.1), &tret, Y, IDA_NORMAL) >= 0);

    DDFree(&dd_mem);
    SUNLinSolFree(LS);
    SUNMatDestroy(J);
    N_VDestroy(Y);
  }

  /* (b) update() reports spec_changed -- DDSetSpec() is applied. */
  {
    N_Vector Y = N_VNew_Serial(si->N_all_orders, sunctx);
    TEST_ASSERT(Y != NULL);
    PendulumY0(&data, ZERO, Y);

    DDMem dd_mem = DDCreate(sunctx);
    TEST_ASSERT(dd_mem != NULL);
    TEST_ASSERT(DDInit(dd_mem, si, PendulumRes, spec, ZERO, Y) == IDA_SUCCESS);
    TEST_ASSERT(DDSetUserData(dd_mem, &data) == IDA_SUCCESS);
    TEST_ASSERT(DDSSTolerances(dd_mem, SUN_RCONST(1.0e-9), SUN_RCONST(1.0e-9)) ==
                IDA_SUCCESS);

    SUNMatrix J = SUNDenseMatrix(si->N_all_orders, si->N_all_orders, sunctx);
    TEST_ASSERT(J != NULL);
    SUNLinearSolver LS = SUNLinSol_Dense(Y, J, sunctx);
    TEST_ASSERT(LS != NULL);
    TEST_ASSERT(DDSetLinearSolver(dd_mem, LS, J) == IDA_SUCCESS);
    TEST_ASSERT(
      DDSetJacFn(dd_mem,
                 (DDLsJacFn){.id = DD_JAC_1, .fn.jacfn1 = PendulumJacfn_Dense}) ==
      IDA_SUCCESS);

    TEST_ASSERT(DDSetStatePivot(dd_mem, always_change) == SUN_SUCCESS);

    uint8_t before[3];
    memcpy(before, DDGetSpec(dd_mem), sizeof(before));

    sunrealtype tret;
    TEST_ASSERT(DDSolve(dd_mem, SUN_RCONST(0.1), &tret, Y, IDA_NORMAL) >= 0);

    const uint8_t* after = DDGetSpec(dd_mem);
    TEST_ASSERT(memcmp(before, after, sizeof(before)) != 0);
    TEST_ASSERT(after[0] == 0 && after[1] == 0 && after[2] == 0);

    DDFree(&dd_mem);
    SUNLinSolFree(LS);
    SUNMatDestroy(J);
    N_VDestroy(Y);
  }

  /* (c) update() fails -- DDSolve() returns DD_ERR_STATE_PIVOT_FAIL without
     calling IDASolve(). */
  {
    N_Vector Y = N_VNew_Serial(si->N_all_orders, sunctx);
    TEST_ASSERT(Y != NULL);
    PendulumY0(&data, ZERO, Y);

    DDMem dd_mem = DDCreate(sunctx);
    TEST_ASSERT(dd_mem != NULL);
    TEST_ASSERT(DDInit(dd_mem, si, PendulumRes, spec, ZERO, Y) == IDA_SUCCESS);
    TEST_ASSERT(DDSetUserData(dd_mem, &data) == IDA_SUCCESS);
    TEST_ASSERT(DDSSTolerances(dd_mem, SUN_RCONST(1.0e-9), SUN_RCONST(1.0e-9)) ==
                IDA_SUCCESS);

    SUNMatrix J = SUNDenseMatrix(si->N_all_orders, si->N_all_orders, sunctx);
    TEST_ASSERT(J != NULL);
    SUNLinearSolver LS = SUNLinSol_Dense(Y, J, sunctx);
    TEST_ASSERT(LS != NULL);
    TEST_ASSERT(DDSetLinearSolver(dd_mem, LS, J) == IDA_SUCCESS);
    TEST_ASSERT(
      DDSetJacFn(dd_mem,
                 (DDLsJacFn){.id = DD_JAC_1, .fn.jacfn1 = PendulumJacfn_Dense}) ==
      IDA_SUCCESS);

    TEST_ASSERT(DDSetStatePivot(dd_mem, always_fail) == SUN_SUCCESS);

    sunrealtype tret = SUN_RCONST(-1.0);
    int flag         = DDSolve(dd_mem, SUN_RCONST(0.1), &tret, Y, IDA_NORMAL);
    TEST_ASSERT(flag == DD_ERR_STATE_PIVOT_FAIL);
    TEST_ASSERT(tret == SUN_RCONST(-1.0));

    DDFree(&dd_mem);
    SUNLinSolFree(LS);
    SUNMatDestroy(J);
    N_VDestroy(Y);
  }

  DDSPFreeEmpty(always_change);
  DDSPFreeEmpty(always_fail);
  DDStaticInfoDestroy(si);
  SUNContext_Free(&sunctx);

  return EXIT_SUCCESS;
}
