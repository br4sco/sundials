#include <idas/idas.h>
#include <nvector/nvector_serial.h>
#include <stddef.h>
#include <stdio.h>
#include <sundials/sundials_core.h>
#include <sundials/sundials_matrix.h>
#include <sundials/sundials_nvector.h>
#include <sunlinsol/sunlinsol_dense.h>
#include <sunlinsol/sunlinsol_klu.h>
#include <sunmatrix/sunmatrix_dense.h>
#include <sunmatrix/sunmatrix_sparse.h>

#include "dd.h"
#include "dd_math.h"
#include "dd_staged_pivot.h"
#include "dd_staged_pivot_matrix.h"
#include "models.h"
#include "static_info.h"
#include "sundials/sundials_types.h"
#include "test.h"

/* -----------------------------------------------------------------------------
 * Verifies that DDReInitB() back to a backward problem's ORIGINAL final time
 * T (not an interior point, unlike dd_test_reinitb_interior.c) works after
 * the backward problem has already been driven down to t0 once. Unlike
 * dd_test_reinitb_interior.c, the trajectory here (m=1, l=1, g=9.81,
 * theta0=PI/4) causes DDSetSpec() pivots during the forward pass, so this
 * exercises reinit across DD checkpoint segment boundaries.
 *
 * Strategy: (A) take one backward step from a fresh terminal condition at T
 * as a baseline. Separately, (B) drive the SAME backward problem down to t0
 * once, then DDReInitB() it back to T with a fresh terminal condition,
 * DDQuadReInitB(), and take one backward step. Compare (A) and (B); since
 * both reuse the SAME forward checkpoints (no fresh forward re-run, no
 * pivot's step/order history disturbed), they must agree BIT FOR BIT.
 *
 * Parametrized over interpolation type (argv[1] = 'h' for IDA_HERMITE, 'p'
 * for IDA_POLYNOMIAL) and over the forward problem's Jacobian callback
 * (argv[2] = 'n' no callback/DQ, 'd' dense, 'r' CSR, 'c' CSC, matching
 * model_pendulum.c's mat_type convention): all combinations must agree BIT
 * FOR BIT.
 * ---------------------------------------------------------------------------*/

#define ZERO SUN_RCONST(0.0)
#define ONE  SUN_RCONST(1.0)
#define TWO  SUN_RCONST(2.0)
#define FIVE SUN_RCONST(5.0)

int main(int argc, char* argv[])
{
  enum
  {
    HERMITE    = 'h',
    POLYNOMIAL = 'p'
  } interp_arg;

  enum
  {
    NONE  = 'n',
    DENSE = 'd',
    CSR   = 'r',
    CSC   = 'c'
  } mat_type;

  TEST_ASSERT(argc > 2);
  switch (argv[1][0])
  {
  case HERMITE: interp_arg = HERMITE; break;
  case POLYNOMIAL: interp_arg = POLYNOMIAL; break;
  default: TEST_ASSERT(0);
  }
  const int interp = (interp_arg == HERMITE) ? IDA_HERMITE : IDA_POLYNOMIAL;

  switch (argv[2][0])
  {
  case NONE: mat_type = NONE; break;
  case DENSE: mat_type = DENSE; break;
  case CSR: mat_type = CSR; break;
  case CSC: mat_type = CSC; break;
  default: TEST_ASSERT(0);
  }

  const sunrealtype t0    = ZERO;
  const sunrealtype tstep = SUN_RCONST(0.1);
  const sunrealtype tout  = SUN_RCONST(5.0);
  const int Nd            = 50;

  SUNContext sunctx;
  TEST_ASSERT(SUNContext_Create(SUN_COMM_NULL, &sunctx) == SUN_SUCCESS);

  DDStaticInfo si = DDStaticInfoCreate(PENDULUM_N,
                                       PENDULUM_C,
                                       PENDULUM_D,
                                       PENDULUM_VAR_IDX_MAP,
                                       PENDULUM_EQN_NAMES,
                                       PENDULUM_VAR_NAMES);
  TEST_ASSERT(si);

  /* ------------------------------------------------------------------------
   * Setup Forward Problem
   * ------------------------------------------------------------------------ */

  SUNMatrix J0 = SUNDenseMatrix(si->N, si->N, sunctx);
  TEST_ASSERT(J0);
  DDStagedPivotMatrix pJ0 = DDStagedPivotMatWrapDense(J0);
  TEST_ASSERT(pJ0);
  N_Vector Y = N_VNew_Serial(si->N_all_orders, sunctx);
  TEST_ASSERT(Y);

  const sunrealtype m = SUN_RCONST(1.0), l = SUN_RCONST(1.0),
                    g = SUN_RCONST(9.81);
  PendulumData* data  = malloc(sizeof(*data));
  data->m             = m;
  data->param[0]      = l;
  data->param[1]      = g;

  sunrealtype theta0 = SUN_RCONST(PI) / SUN_RCONST(4.0);
  PendulumY0(data, theta0, Y);

  DDStatePivot sp = DDSPStaged(sunctx, si, pJ0, PendulumJacf0, ZERO);
  TEST_ASSERT(sp != NULL);

  uint8_t spec[si->N];
  sunbooleantype spec_changed;
  TEST_ASSERT(DDSPUpdate(sp, t0, Y, data, spec, &spec_changed) >= 0);

  DDMem dd_mem = DDCreate(sunctx);
  TEST_ASSERT(dd_mem);

  TEST_ASSERT(DDInit(dd_mem, si, PendulumRes, spec, t0, Y) == IDA_SUCCESS);
  TEST_ASSERT(DDSetStatePivot(dd_mem, sp) == SUN_SUCCESS);

  TEST_ASSERT(DDAdjInit(dd_mem, Nd, interp) == IDA_SUCCESS);

  TEST_ASSERT(DDSetUserData(dd_mem, data) == IDA_SUCCESS);

  TEST_ASSERT(DDSSTolerances(dd_mem, SUN_RCONST(1.0e-9), SUN_RCONST(1.0e-9)) ==
              IDA_SUCCESS);

  const sunindextype N = si->N_all_orders;
  SUNMatrix J          = NULL;
  SUNLinearSolver LS   = NULL;

  switch (mat_type)
  {
  case NONE:
  case DENSE:
    J = SUNDenseMatrix(N, N, sunctx);
    TEST_ASSERT(J);
    LS = SUNLinSol_Dense(Y, J, sunctx);
    TEST_ASSERT(LS);
    break;
  case CSR:
  case CSC:
    J = SUNSparseMatrix(N,
                        N,
                        PENDULUM_JAC_NNZ + 0 + 4,
                        mat_type == CSR ? CSR_MAT : CSC_MAT,
                        sunctx);
    TEST_ASSERT(J);
    LS = SUNLinSol_KLU(Y, J, sunctx);
    TEST_ASSERT(LS);
    break;
  }

  TEST_ASSERT(DDSetLinearSolver(dd_mem, LS, J) == IDA_SUCCESS);

  switch (mat_type)
  {
  case NONE: break;
  case DENSE:
    TEST_ASSERT(
      DDSetJacFn(dd_mem,
                 (DDLsJacFn){.id = DD_JAC_1, .fn.jacfn1 = PendulumJacfn_Dense}) ==
      IDA_SUCCESS);
    break;
  case CSR:
    TEST_ASSERT(
      DDSetJacFn(dd_mem,
                 (DDLsJacFn){.id = DD_JAC_1, .fn.jacfn1 = PendulumJacfn_CSR}) ==
      IDA_SUCCESS);
    break;
  case CSC:
    TEST_ASSERT(
      DDSetJacFn(dd_mem,
                 (DDLsJacFn){.id = DD_JAC_2, .fn.jacfn2 = PendulumJacfn_CSC}) ==
      IDA_SUCCESS);
    break;
  }

  /* ------------------------------------------------------------------------
   * Integrate Forward
   * ------------------------------------------------------------------------ */

  sunrealtype t    = t0;
  sunrealtype tret = ZERO;
  int ncheck       = 0;

  while (SUNTRUE)
  {
    t += tstep;
    if (t >= tout) { break; }

    TEST_ASSERT(DDSolveF(dd_mem, t, &tret, Y, IDA_NORMAL, &ncheck) == IDA_SUCCESS);
  }

  printf("Forward integration done: tret=%.4f, ncheck=%d\n", tret, ncheck);

  /* ------------------------------------------------------------------------
   * Setup Backwards Problem
   * ------------------------------------------------------------------------ */

  N_Vector yyBT = N_VNew_Serial(PENDULUM_ADJ_N, sunctx);
  TEST_ASSERT(yyBT);
  N_Vector ypBT = N_VClone(yyBT);
  TEST_ASSERT(ypBT);
  PendulumYyBT(data, Y, yyBT, ypBT);

  N_Vector yyB = N_VClone(yyBT);
  N_Vector ypB = N_VClone(yyBT);

  int indexB;
  TEST_ASSERT(DDCreateB(dd_mem, &indexB) == IDA_SUCCESS);

  N_VScale(ONE, yyBT, yyB);
  N_VScale(ONE, ypBT, ypB);
  TEST_ASSERT(DDInitB(dd_mem, indexB, PendulumResB, tret, yyB, ypB) ==
              IDA_SUCCESS);

  TEST_ASSERT(
    DDSStolerancesB(dd_mem, indexB, SUN_RCONST(1.e-6), SUN_RCONST(1.e-6)) ==
    IDA_SUCCESS);

  TEST_ASSERT(DDSetUserDataB(dd_mem, indexB, data) == IDA_SUCCESS);

  SUNMatrix AB = SUNDenseMatrix(PENDULUM_ADJ_N, PENDULUM_ADJ_N, sunctx);
  TEST_ASSERT(AB);
  SUNLinearSolver LSB = SUNLinSol_Dense(yyB, AB, sunctx);
  TEST_ASSERT(LSB);

  TEST_ASSERT(DDSetLinearSolverB(dd_mem, indexB, LSB, AB) == IDA_SUCCESS);
  TEST_ASSERT(DDSetJacFnB(dd_mem, indexB, PendulumJacFnB) == IDA_SUCCESS);

  N_Vector qB = N_VNew_Serial(PENDULUM_NP + 1, sunctx);
  TEST_ASSERT(qB);
  N_VConst(ZERO, qB);
  TEST_ASSERT(DDQuadInitB(dd_mem, indexB, PendulumQuadRhsFnB, qB) == IDA_SUCCESS);

  /* ------------------------------------------------------------------------
   * (A) Baseline: from a fresh terminal condition at T, take a single
   *     backward step to T - tstep.
   * ------------------------------------------------------------------------ */

  sunrealtype tgot_baseline = tret;
  int flag                  = DDSolveB(dd_mem, tret - tstep, IDA_NORMAL);
  TEST_ASSERT(flag >= 0);
  TEST_ASSERT(DDGetB(dd_mem, indexB, &tgot_baseline, yyB, ypB) == IDA_SUCCESS);

  N_Vector yyB_baseline = N_VClone(yyB);
  N_VScale(ONE, yyB, yyB_baseline);

  TEST_ASSERT(DDGetQuadB(dd_mem, indexB, &tgot_baseline, qB) == IDA_SUCCESS);
  N_Vector qB_baseline = N_VClone(qB);
  N_VScale(ONE, qB, qB_baseline);

  printf("Baseline single backward step reached t=%.6f\n", tgot_baseline);

  /* ------------------------------------------------------------------------
   * Full backward pass: T -> t0, stepping by tstep.
   * ------------------------------------------------------------------------ */

  N_VScale(ONE, yyBT, yyB);
  N_VScale(ONE, ypBT, ypB);
  TEST_ASSERT(DDReInitB(dd_mem, indexB, tret, yyB, ypB) == IDA_SUCCESS);

  N_VConst(ZERO, qB);
  TEST_ASSERT(DDQuadReInitB(dd_mem, indexB, qB) == IDA_SUCCESS);

  sunrealtype tB = tret;
  while (tB > t0)
  {
    tB -= tstep;
    if (tB < t0) { break; }
    flag = DDSolveB(dd_mem, tB, IDA_NORMAL);
    TEST_ASSERT(flag >= 0);
    TEST_ASSERT(DDGetB(dd_mem, indexB, &tB, yyB, ypB) == IDA_SUCCESS);
    TEST_ASSERT(DDGetQuadB(dd_mem, indexB, &tB, qB) == IDA_SUCCESS);
  }

  printf("Full backward pass reached t=%.6f\n", tB);

  /* ------------------------------------------------------------------------
   * Reinitialize the SAME backward problem back to its ORIGINAL final time T
   * (not an interior point) using a fresh terminal condition, then resume
   * solving one step.
   * ------------------------------------------------------------------------ */

  N_Vector yyBT2 = N_VClone(yyBT);
  N_Vector ypBT2 = N_VClone(yyBT);
  PendulumYyBT(data, Y, yyBT2, ypBT2);

  TEST_ASSERT(DDReInitB(dd_mem, indexB, tret, yyBT2, ypBT2) == IDA_SUCCESS);

  N_Vector qB2 = N_VClone(qB);
  N_VConst(ZERO, qB2);
  TEST_ASSERT(DDQuadReInitB(dd_mem, indexB, qB2) == IDA_SUCCESS);

  printf("Reinit-to-original-final-time call returned OK; resuming solve...\n");

  sunrealtype tgot_resumed = tret;
  flag                     = DDSolveB(dd_mem, tret - tstep, IDA_NORMAL);
  TEST_ASSERT(flag >= 0);
  TEST_ASSERT(DDGetB(dd_mem, indexB, &tgot_resumed, yyB, ypB) == IDA_SUCCESS);
  TEST_ASSERT(DDGetQuadB(dd_mem, indexB, &tgot_resumed, qB2) == IDA_SUCCESS);

  printf("Resumed backward solve reached t=%.6f\n", tgot_resumed);

  /* ------------------------------------------------------------------------
   * Compare (A) baseline vs. (B) reproduction
   * ------------------------------------------------------------------------ */

  sunrealtype* d_arr = N_VGetArrayPointer(yyB_baseline);
  sunrealtype* r_arr = N_VGetArrayPointer(yyB);

  sunrealtype maxerr = ZERO;
  for (int i = 0; i < PENDULUM_ADJ_N; ++i)
  {
    sunrealtype err = SUNRabs(d_arr[i] - r_arr[i]);
    printf("  lambda[%d]: baseline=% .12f  resumed=% .12f  |diff|=%.3e\n",
           i,
           d_arr[i],
           r_arr[i],
           err);
    if (err > maxerr) { maxerr = err; }
  }
  printf("max |baseline - resumed| (state) = %.3e\n", maxerr);

  sunrealtype* qd_arr = N_VGetArrayPointer(qB_baseline);
  sunrealtype* qr_arr = N_VGetArrayPointer(qB2);

  sunrealtype maxerr_q = ZERO;
  for (int i = 0; i < PENDULUM_NP + 1; ++i)
  {
    sunrealtype err = SUNRabs(qd_arr[i] - qr_arr[i]);
    printf("  qB[%d]: baseline=% .12f  resumed=% .12f  |diff|=%.3e\n",
           i,
           qd_arr[i],
           qr_arr[i],
           err);
    if (err > maxerr_q) { maxerr_q = err; }
  }
  printf("max |baseline - resumed| (quad)  = %.3e\n", maxerr_q);

  int ok = (maxerr == ZERO) && (maxerr_q == ZERO);

  /* Cleanup */
  DDAdjFree(dd_mem);
  DDFree(&dd_mem);
  DDSPDestroy(sp);
  DDStagedPivotMatDestroy(pJ0);
  N_VDestroy(Y);
  N_VDestroy(yyB);
  N_VDestroy(ypB);
  N_VDestroy(yyBT);
  N_VDestroy(ypBT);
  N_VDestroy(yyBT2);
  N_VDestroy(ypBT2);
  N_VDestroy(yyB_baseline);
  N_VDestroy(qB);
  N_VDestroy(qB2);
  N_VDestroy(qB_baseline);
  DDStaticInfoDestroy(si);
  SUNContext_Free(&sunctx);
  SUNLinSolFree(LS);
  SUNLinSolFree(LSB);
  SUNMatDestroy(AB);
  SUNMatDestroy(J);
  SUNMatDestroy(J0);
  free(data);

  if (ok) { printf("SUCCESS\n"); }
  else
  {
    printf("FAIL\n");
  }

  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
