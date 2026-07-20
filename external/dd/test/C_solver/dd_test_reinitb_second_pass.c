#include <idas/idas.h>
#include <nvector/nvector_serial.h>
#include <stddef.h>
#include <stdio.h>
#include <sundials/sundials_core.h>
#include <sundials/sundials_matrix.h>
#include <sundials/sundials_nvector.h>
#include <sunlinsol/sunlinsol_dense.h>
#include <sunmatrix/sunmatrix_dense.h>

#include "dd.h"
#include "dd_math.h"
#include "matrix.h"
#include "models.h"
#include "pivot.h"
#include "static_info.h"
#include "sundials/sundials_types.h"
#include "test.h"

/* -----------------------------------------------------------------------------
 * Verifies that a SECOND full backward pass -- DDReInitB() back to T after
 * the backward problem has already been driven down to t0 once, then driven
 * down to t0 again with the same terminal condition, stepping by tstep like
 * the first pass -- completes successfully and reproduces the same result as
 * the first pass. Both passes start from the same tret with the same
 * consistent terminal condition, so they must agree exactly (within
 * tolerance).
 *
 * Additionally verifies a THIRD pass that exercises DDAdjReInit(): the
 * forward problem is reset to t0 via DDReInit(), the checkpoints are
 * discarded and re-primed via DDAdjReInit(), and a fresh forward solve is
 * driven from t0 to T to regenerate checkpoints from scratch. The backward
 * problem is then reinitialized with DDReInitB() using the terminal
 * condition from this new forward solve and driven back down to t0. Since
 * the forward trajectory and terminal condition are numerically identical to
 * the first pass, the result must again agree with the first pass (within
 * tolerance).
 * ---------------------------------------------------------------------------*/

#define ZERO SUN_RCONST(0.0)
#define ONE  SUN_RCONST(1.0)
#define TWO  SUN_RCONST(2.0)
#define FIVE SUN_RCONST(5.0)

int main(void)
{
  const sunrealtype t0    = ZERO;
  const sunrealtype tstep = SUN_RCONST(0.1);
  const sunrealtype tout  = SUN_RCONST(5.0);
  const int Nd            = 50;
  const sunrealtype eps   = SUN_RCONST(1.0e-4);

  SUNContext sunctx;
  TEST_ASSERT(SUNContext_Create(SUN_COMM_NULL, &sunctx) == SUN_SUCCESS);

  DDStaticInfo si = DDStaticInfoCreate(sunctx,
                                       PENDULUM_N,
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
  PIVMatrix pJ0 = PIVMatWrapDense(J0);
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

  PivMem pm = PIVCreate(sunctx, si, pJ0, PendulumJacf0);
  TEST_ASSERT(pm);
  TEST_ASSERT(PIVSetUserData(pm, data) == SUN_SUCCESS);
  sunbooleantype spec_changed;
  TEST_ASSERT(PIVPivot(pm, ZERO, t0, Y, &spec_changed) == SUN_SUCCESS);

  DDMem dd_mem = DDCreate(sunctx);
  TEST_ASSERT(dd_mem);

  TEST_ASSERT(DDInit(dd_mem, si, PendulumRes, PIVGetSpec(pm), t0, Y) ==
              IDA_SUCCESS);

  TEST_ASSERT(DDAdjInit(dd_mem, Nd, IDA_HERMITE) == IDA_SUCCESS);

  TEST_ASSERT(DDSetUserData(dd_mem, data) == IDA_SUCCESS);

  TEST_ASSERT(DDSSTolerances(dd_mem, SUN_RCONST(1.0e-9), SUN_RCONST(1.0e-9)) ==
              IDA_SUCCESS);

  SUNMatrix J = SUNDenseMatrix(si->N_all_orders, si->N_all_orders, sunctx);
  TEST_ASSERT(J);
  SUNLinearSolver LS = SUNLinSol_Dense(Y, J, sunctx);
  TEST_ASSERT(LS);

  TEST_ASSERT(DDSetLinearSolver(dd_mem, LS, J) == IDA_SUCCESS);

  /* ------------------------------------------------------------------------
   * Integrate Forward
   * ------------------------------------------------------------------------ */

  sunrealtype t    = t0;
  sunrealtype tret = ZERO;
  int ncheck       = 0;

  while (SUNTRUE)
  {
    TEST_ASSERT(PIVPivot(pm, ZERO, t, Y, &spec_changed) == SUN_SUCCESS);
    if (spec_changed)
    {
      TEST_ASSERT(DDSetSpec(dd_mem, PIVGetSpec(pm)) == SUN_SUCCESS);
    }

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
   * First full backward pass: T -> t0, stepping by tstep.
   * ------------------------------------------------------------------------ */

  int flag;
  sunrealtype tB = tret;
  while (tB > t0)
  {
    sunrealtype tnext = tB - tstep;
    if (tnext < t0) { tnext = t0; }
    flag = DDSolveB(dd_mem, tnext, IDA_NORMAL);
    TEST_ASSERT(flag >= 0);
    tB = tnext;
    TEST_ASSERT(DDGetB(dd_mem, indexB, &tB, yyB, ypB) == IDA_SUCCESS);
    TEST_ASSERT(DDGetQuadB(dd_mem, indexB, &tB, qB) == IDA_SUCCESS);
  }

  printf("First backward pass reached t=%.6f\n", tB);

  /* ------------------------------------------------------------------------
   * Reinitialize back to T with a fresh (but numerically identical)
   * terminal condition, then drive down to t0 a second time.
   * ------------------------------------------------------------------------ */

  N_Vector yyBT2 = N_VClone(yyBT);
  N_Vector ypBT2 = N_VClone(yyBT);
  PendulumYyBT(data, Y, yyBT2, ypBT2);

  TEST_ASSERT(DDReInitB(dd_mem, indexB, tret, yyBT2, ypBT2) == IDA_SUCCESS);

  N_Vector qB2 = N_VClone(qB);
  N_VConst(ZERO, qB2);
  TEST_ASSERT(DDQuadReInitB(dd_mem, indexB, qB2) == IDA_SUCCESS);

  N_Vector yyB2 = N_VClone(yyBT);
  N_Vector ypB2 = N_VClone(yyBT);
  N_VScale(ONE, yyBT2, yyB2);
  N_VScale(ONE, ypBT2, ypB2);

  sunrealtype tB2 = tret;
  while (tB2 > t0)
  {
    sunrealtype tnext = tB2 - tstep;
    if (tnext < t0) { tnext = t0; }
    flag = DDSolveB(dd_mem, tnext, IDA_NORMAL);
    TEST_ASSERT(flag >= 0);
    tB2 = tnext;
    TEST_ASSERT(DDGetB(dd_mem, indexB, &tB2, yyB2, ypB2) == IDA_SUCCESS);
    TEST_ASSERT(DDGetQuadB(dd_mem, indexB, &tB2, qB2) == IDA_SUCCESS);
  }

  printf("Second backward pass reached t=%.6f\n", tB2);

  /* ------------------------------------------------------------------------
   * Compare pass 1 (yyB/qB, already at t0) vs. pass 2 (yyB2/qB2, also at t0)
   * ------------------------------------------------------------------------ */

  sunrealtype* d_arr = N_VGetArrayPointer(yyB);
  sunrealtype* r_arr = N_VGetArrayPointer(yyB2);

  sunrealtype maxerr = ZERO;
  for (int i = 0; i < PENDULUM_ADJ_N; ++i)
  {
    sunrealtype err = SUNRabs(d_arr[i] - r_arr[i]);
    printf("  lambda[%d]: pass1=% .12f  pass2=% .12f  |diff|=%.3e\n",
           i,
           d_arr[i],
           r_arr[i],
           err);
    if (err > maxerr) { maxerr = err; }
  }
  printf("max |pass1 - pass2| (state) = %.3e\n", maxerr);

  sunrealtype* qd_arr = N_VGetArrayPointer(qB);
  sunrealtype* qr_arr = N_VGetArrayPointer(qB2);

  sunrealtype maxerr_q = ZERO;
  for (int i = 0; i < PENDULUM_NP + 1; ++i)
  {
    sunrealtype err = SUNRabs(qd_arr[i] - qr_arr[i]);
    printf("  qB[%d]: pass1=% .12f  pass2=% .12f  |diff|=%.3e\n",
           i,
           qd_arr[i],
           qr_arr[i],
           err);
    if (err > maxerr_q) { maxerr_q = err; }
  }
  printf("max |pass1 - pass2| (quad)  = %.3e\n", maxerr_q);

  /* ------------------------------------------------------------------------
   * Third pass: exercise DDAdjReInit(). Reset the forward problem to t0,
   * discard and re-prime the checkpoints, then drive a fresh forward
   * solution from t0 to T (regenerating checkpoints), and finally reinit and
   * re-drive the backward problem down to t0 using those new checkpoints.
   * ------------------------------------------------------------------------ */

  PendulumY0(data, theta0, Y);
  TEST_ASSERT(PIVPivot(pm, ZERO, t0, Y, &spec_changed) == SUN_SUCCESS);
  TEST_ASSERT(DDReInit(dd_mem, PIVGetSpec(pm), t0, Y) == IDA_SUCCESS);
  TEST_ASSERT(DDAdjReInit(dd_mem) == IDA_SUCCESS);

  sunrealtype t3    = t0;
  sunrealtype tret3 = ZERO;
  int ncheck3       = 0;

  while (SUNTRUE)
  {
    TEST_ASSERT(PIVPivot(pm, ZERO, t3, Y, &spec_changed) == SUN_SUCCESS);
    if (spec_changed)
    {
      TEST_ASSERT(DDSetSpec(dd_mem, PIVGetSpec(pm)) == SUN_SUCCESS);
    }

    t3 += tstep;
    if (t3 >= tout) { break; }

    TEST_ASSERT(DDSolveF(dd_mem, t3, &tret3, Y, IDA_NORMAL, &ncheck3) ==
                IDA_SUCCESS);
  }

  printf("Third pass forward integration done: tret=%.4f, ncheck=%d\n", tret3,
         ncheck3);

  N_Vector yyBT3 = N_VClone(yyBT);
  N_Vector ypBT3 = N_VClone(yyBT);
  PendulumYyBT(data, Y, yyBT3, ypBT3);

  TEST_ASSERT(DDReInitB(dd_mem, indexB, tret3, yyBT3, ypBT3) == IDA_SUCCESS);

  N_Vector qB3 = N_VClone(qB);
  N_VConst(ZERO, qB3);
  TEST_ASSERT(DDQuadReInitB(dd_mem, indexB, qB3) == IDA_SUCCESS);

  N_Vector yyB3 = N_VClone(yyBT);
  N_Vector ypB3 = N_VClone(yyBT);
  N_VScale(ONE, yyBT3, yyB3);
  N_VScale(ONE, ypBT3, ypB3);

  sunrealtype tB3 = tret3;
  while (tB3 > t0)
  {
    sunrealtype tnext = tB3 - tstep;
    if (tnext < t0) { tnext = t0; }
    flag = DDSolveB(dd_mem, tnext, IDA_NORMAL);
    TEST_ASSERT(flag >= 0);
    tB3 = tnext;
    TEST_ASSERT(DDGetB(dd_mem, indexB, &tB3, yyB3, ypB3) == IDA_SUCCESS);
    TEST_ASSERT(DDGetQuadB(dd_mem, indexB, &tB3, qB3) == IDA_SUCCESS);
  }

  printf("Third backward pass reached t=%.6f\n", tB3);

  /* ------------------------------------------------------------------------
   * Compare pass 1 (yyB/qB, already at t0) vs. pass 3 (yyB3/qB3, also at t0)
   * ------------------------------------------------------------------------ */

  sunrealtype* r3_arr = N_VGetArrayPointer(yyB3);

  sunrealtype maxerr3 = ZERO;
  for (int i = 0; i < PENDULUM_ADJ_N; ++i)
  {
    sunrealtype err = SUNRabs(d_arr[i] - r3_arr[i]);
    printf("  lambda[%d]: pass1=% .12f  pass3=% .12f  |diff|=%.3e\n",
           i,
           d_arr[i],
           r3_arr[i],
           err);
    if (err > maxerr3) { maxerr3 = err; }
  }
  printf("max |pass1 - pass3| (state) = %.3e\n", maxerr3);

  sunrealtype* qr3_arr = N_VGetArrayPointer(qB3);

  sunrealtype maxerr_q3 = ZERO;
  for (int i = 0; i < PENDULUM_NP + 1; ++i)
  {
    sunrealtype err = SUNRabs(qd_arr[i] - qr3_arr[i]);
    printf("  qB[%d]: pass1=% .12f  pass3=% .12f  |diff|=%.3e\n",
           i,
           qd_arr[i],
           qr3_arr[i],
           err);
    if (err > maxerr_q3) { maxerr_q3 = err; }
  }
  printf("max |pass1 - pass3| (quad)  = %.3e\n", maxerr_q3);

  int ok = (maxerr <= eps) && (maxerr_q <= eps) && (maxerr3 <= eps) &&
           (maxerr_q3 <= eps);

  /* Cleanup */
  DDAdjFree(dd_mem);
  DDFree(&dd_mem);
  PIVDestroy(&pm);
  PIVMatDestroy(pJ0);
  N_VDestroy(Y);
  N_VDestroy(yyB);
  N_VDestroy(ypB);
  N_VDestroy(yyBT);
  N_VDestroy(ypBT);
  N_VDestroy(yyBT2);
  N_VDestroy(ypBT2);
  N_VDestroy(yyB2);
  N_VDestroy(ypB2);
  N_VDestroy(qB);
  N_VDestroy(qB2);
  N_VDestroy(yyBT3);
  N_VDestroy(ypBT3);
  N_VDestroy(yyB3);
  N_VDestroy(ypB3);
  N_VDestroy(qB3);
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
