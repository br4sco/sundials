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
#include "dd_staged_pivot.h"
#include "dd_staged_pivot_matrix.h"
#include "models.h"
#include "static_info.h"
#include "sundials/sundials_types.h"
#include "test.h"

/* -----------------------------------------------------------------------------
 * Verifies that DDReInitB (a thin wrapper over IDAReInitB) can re-initialize
 * a backward problem at an ARBITRARY interior time point of the forward
 * solution interval, not just at the initial backward time T.
 *
 * Model: the index-3 pendulum ASA problem already used by
 * model_pendulum_ASA.c (adjoint depends on the interpolated forward
 * trajectory, so a wrong checkpoint lookup after reinit would show up as a
 * measurable disagreement).
 *
 * Strategy: solve the backward problem directly from T down to t0 in one
 * shot (baseline). Separately, reinitialize the SAME backward problem back
 * to T, solve down to an interior point t_mid, grab the (consistent) state
 * there, reinitialize the backward problem AT t_mid with that state, and
 * continue solving down to t0. Compare the two t0 results.
 * ---------------------------------------------------------------------------*/

#define ZERO SUN_RCONST(0.0)
#define ONE  SUN_RCONST(1.0)
#define TWO  SUN_RCONST(2.0)
#define FIVE SUN_RCONST(5.0)

int main(void)
{
  const sunrealtype t0     = ZERO;
  const sunrealtype tstep  = SUN_RCONST(0.1);
  const sunrealtype tout   = SUN_RCONST(15.0);
  const sunrealtype t_mid  = SUN_RCONST(9.0); /* interior reinit point */
  const sunrealtype t_targ = SUN_RCONST(3.0); /* interior target point */
  const int Nd             = 20;
  const sunrealtype eps    = SUN_RCONST(1.0e-4);

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

  const sunrealtype m = SUN_RCONST(1.1), l = SUN_RCONST(1.2), g = SUN_RCONST(1.3);
  PendulumData* data = malloc(sizeof(*data));
  data->m            = m;
  data->param[0]     = l;
  data->param[1]     = g;

  sunrealtype theta0 = SUN_RCONST(PI) / SUN_RCONST(6.0);
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

  TEST_ASSERT(DDAdjInit(dd_mem, Nd, IDA_POLYNOMIAL) == IDA_SUCCESS);

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

  /* Backward quadrature: dG/dm, dG/dl, dG/dg, zero at t = T. */
  N_Vector qB = N_VNew_Serial(PENDULUM_NP + 1, sunctx);
  TEST_ASSERT(qB);
  N_VConst(ZERO, qB);
  TEST_ASSERT(DDQuadInitB(dd_mem, indexB, PendulumQuadRhsFnB, qB) == IDA_SUCCESS);

  /* ------------------------------------------------------------------------
   * (A) Direct backward solve: T -> t_targ, stepping by tstep (mirrors how
   *     model_pendulum_ASA.c drives DDSolveB -- a single large jump can hit
   *     IDA's per-call mxsteps cap for this stiff, frequently-repivoted
   *     problem).
   * ------------------------------------------------------------------------ */

  sunrealtype tgot = tret;
  int flag;
  while (tgot > t_targ)
  {
    sunrealtype tnext = tgot - tstep;
    if (tnext < t_targ) { tnext = t_targ; }
    flag = DDSolveB(dd_mem, tnext, IDA_NORMAL);
    TEST_ASSERT(flag >= 0);
    tgot = tnext;
    TEST_ASSERT(DDGetB(dd_mem, indexB, &tgot, yyB, ypB) == IDA_SUCCESS);
  }

  N_Vector yyB_direct = N_VClone(yyB);
  N_VScale(ONE, yyB, yyB_direct);

  TEST_ASSERT(DDGetQuadB(dd_mem, indexB, &tgot, qB) == IDA_SUCCESS);
  N_Vector qB_direct = N_VClone(qB);
  N_VScale(ONE, qB, qB_direct);

  printf("Direct backward solve reached t=%.6f\n", tgot);

  /* ------------------------------------------------------------------------
   * (B) Reinit back to T, solve to interior t_mid, reinit AT t_mid using the
   *     state obtained there, then continue solving to t_targ
   * ------------------------------------------------------------------------ */

  N_VScale(ONE, yyBT, yyB);
  N_VScale(ONE, ypBT, ypB);
  TEST_ASSERT(DDReInitB(dd_mem, indexB, tret, yyB, ypB) == IDA_SUCCESS);

  N_VConst(ZERO, qB);
  TEST_ASSERT(DDQuadReInitB(dd_mem, indexB, qB) == IDA_SUCCESS);

  sunrealtype tmid_got = tret;
  while (tmid_got > t_mid)
  {
    sunrealtype tnext = tmid_got - tstep;
    if (tnext < t_mid) { tnext = t_mid; }
    flag = DDSolveB(dd_mem, tnext, IDA_NORMAL);
    TEST_ASSERT(flag >= 0);
    tmid_got = tnext;
    TEST_ASSERT(DDGetB(dd_mem, indexB, &tmid_got, yyB, ypB) == IDA_SUCCESS);
  }
  printf("Backward solve to interior t_mid reached t=%.6f\n", tmid_got);

  /* Grab the quadrature value accumulated so far (T -> t_mid); this is a
   * continuation, not a reset, so it must be fed back in, not zeroed. */
  TEST_ASSERT(DDGetQuadB(dd_mem, indexB, &tmid_got, qB) == IDA_SUCCESS);

  /* Re-initialize the backward problem AT THE INTERIOR POINT t_mid_got using
   * the (consistent) state the integrator produced there. */
  TEST_ASSERT(DDReInitB(dd_mem, indexB, tmid_got, yyB, ypB) == IDA_SUCCESS);
  TEST_ASSERT(DDQuadReInitB(dd_mem, indexB, qB) == IDA_SUCCESS);

  tgot = tmid_got;
  while (tgot > t_targ)
  {
    sunrealtype tnext = tgot - tstep;
    if (tnext < t_targ) { tnext = t_targ; }
    flag = DDSolveB(dd_mem, tnext, IDA_NORMAL);
    TEST_ASSERT(flag >= 0);
    tgot = tnext;
    TEST_ASSERT(DDGetB(dd_mem, indexB, &tgot, yyB, ypB) == IDA_SUCCESS);
  }
  printf("Reinit-at-interior-point backward solve reached t=%.6f\n", tgot);

  TEST_ASSERT(DDGetQuadB(dd_mem, indexB, &tgot, qB) == IDA_SUCCESS);

  /* ------------------------------------------------------------------------
   * Compare
   * ------------------------------------------------------------------------ */

  sunrealtype* d_arr = N_VGetArrayPointer(yyB_direct);
  sunrealtype* r_arr = N_VGetArrayPointer(yyB);

  sunrealtype maxerr = ZERO;
  for (int i = 0; i < PENDULUM_ADJ_N; ++i)
  {
    sunrealtype err = SUNRabs(d_arr[i] - r_arr[i]);
    printf("  lambda[%d]: direct=% .12f  reinit=% .12f  |diff|=%.3e\n",
           i,
           d_arr[i],
           r_arr[i],
           err);
    if (err > maxerr) { maxerr = err; }
  }
  printf("max |direct - reinit| (state) = %.3e\n", maxerr);

  sunrealtype* qd_arr = N_VGetArrayPointer(qB_direct);
  sunrealtype* qr_arr = N_VGetArrayPointer(qB);

  sunrealtype maxerr_q = ZERO;
  for (int i = 0; i < PENDULUM_NP + 1; ++i)
  {
    sunrealtype err = SUNRabs(qd_arr[i] - qr_arr[i]);
    printf("  qB[%d]: direct=% .12f  reinit=% .12f  |diff|=%.3e\n",
           i,
           qd_arr[i],
           qr_arr[i],
           err);
    if (err > maxerr_q) { maxerr_q = err; }
  }
  printf("max |direct - reinit| (quad)  = %.3e\n", maxerr_q);

  int ok = (maxerr <= eps) && (maxerr_q <= eps);

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
  N_VDestroy(yyB_direct);
  N_VDestroy(qB);
  N_VDestroy(qB_direct);
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
