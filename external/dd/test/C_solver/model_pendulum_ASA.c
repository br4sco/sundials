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

#define ZERO SUN_RCONST(0.0)
#define ONE  SUN_RCONST(1.0)
#define TWO  SUN_RCONST(2.0)
#define FOUR SUN_RCONST(4.0)
#define FIVE SUN_RCONST(5.0)

int main(void)
{
  const sunrealtype t0    = ZERO;
  const sunrealtype tstep = SUN_RCONST(0.1);
  const sunrealtype tout  = SUN_RCONST(100.0);
  const int Nd            = 200;

  /* Setup Sundials context. */
  SUNContext sunctx;
  TEST_ASSERT(SUNContext_Create(SUN_COMM_NULL, &sunctx) == SUN_SUCCESS);

  /* Compute DAE structure. */
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

  /* Allocate state and Jacobian data. */
  SUNMatrix J0 = SUNDenseMatrix(si->N, si->N, sunctx);
  TEST_ASSERT(J0);
  DDStagedPivotMatrix pJ0 = DDStagedPivotMatWrapDense(J0);
  TEST_ASSERT(pJ0);
  N_Vector Y = N_VNew_Serial(si->N_all_orders, sunctx);
  TEST_ASSERT(Y);

  /* Allocate Quadrature RHS */
  N_Vector Q = N_VNew_Serial(1, sunctx);
  N_VScale(ZERO, Q, Q);

  /* Set DAE parameters. */
  const sunrealtype m = SUN_RCONST(1.1), l = SUN_RCONST(1.2), g = SUN_RCONST(1.3);
  PendulumData* data = malloc(sizeof(*data));
  data->m            = m;
  data->param[0]     = l;
  data->param[1]     = g;

  /* Set initial values. */
  sunrealtype theta0 = SUN_RCONST(PI) / FIVE + SUN_RCONST(PI) / TWO;
  PendulumY0(data, theta0, Y);

  /* Create pivot policy and compute initial spec. */
  DDStatePivot sp = DDSPStaged(sunctx, si, pJ0, PendulumJacf0, ZERO);
  TEST_ASSERT(sp != NULL);

  uint8_t spec[si->N];
  sunbooleantype spec_changed;
  TEST_ASSERT(DDSPUpdate(sp, t0, Y, data, spec, &spec_changed) >= 0);

  /* Create solver session. */
  DDMem dd_mem = DDCreate(sunctx);
  TEST_ASSERT(dd_mem);

  TEST_ASSERT(DDInit(dd_mem, si, PendulumRes, spec, t0, Y) == IDA_SUCCESS);
  TEST_ASSERT(DDSetStatePivot(dd_mem, sp) == SUN_SUCCESS);

  TEST_ASSERT(DDAdjInit(dd_mem, Nd, IDA_POLYNOMIAL) == IDA_SUCCESS);

  TEST_ASSERT(DDQuadInit(dd_mem, PendulumG, Q) == IDA_SUCCESS);

  TEST_ASSERT(DDSetUserData(dd_mem, data) == IDA_SUCCESS);

  TEST_ASSERT(DDSSTolerances(dd_mem, SUN_RCONST(1.0e-9), SUN_RCONST(1.0e-9)) ==
              IDA_SUCCESS);

  /* Setup and set linear solver. */
  SUNMatrix J = SUNDenseMatrix(si->N_all_orders, si->N_all_orders, sunctx);
  TEST_ASSERT(J);
  SUNLinearSolver LS = SUNLinSol_Dense(Y, J, sunctx);
  TEST_ASSERT(LS);

  TEST_ASSERT(DDSetLinearSolver(dd_mem, LS, J) == IDA_SUCCESS);

  /* Set up result file */
  FILE* filef = fopen("pendulum_ASA_forward.csv", "w");
  TEST_ASSERT(filef);

  /* ------------------------------------------------------------------------
   * Integrate Forward
   * ------------------------------------------------------------------------ */

  /* Solve and output solution. */
  fprintf(filef, "t,x,y,λ,G,ΔL\n"); /* print header */

  int flag         = IDA_SUCCESS;
  sunrealtype t    = t0;
  sunrealtype tret = ZERO;
  int ncheck       = 0;

  while (SUNTRUE)
  {
    const sunrealtype x = P_Ith(Y, 0, 0), y = P_Ith(Y, 1, 0),
                      lam = P_Ith(Y, 2, 0);

    fprintf(filef,
            "%.20f,%.20f,%.20f,%.20f,%.20f,%.20f\n",
            t,
            x,
            y,
            lam,
            NV_Ith(Q, 0),
            x * x + y * y - l * l);

    t += tstep;

    if (t >= tout) { break; }

    TEST_ASSERT(DDSolveF(dd_mem, t, &tret, Y, IDA_NORMAL, &ncheck) == IDA_SUCCESS);

    TEST_ASSERT(DDGetQuad(dd_mem, &tret, Q) == IDA_SUCCESS);
  }

  /* ------------------------------------------------------------------------
   * Setup Backwards Problem
   * ------------------------------------------------------------------------ */

  /* Allocate state and set initial values */
  N_Vector yyB = N_VNew_Serial(PENDULUM_ADJ_N, sunctx);
  TEST_ASSERT(yyB);

  N_Vector ypB = N_VClone(yyB);
  TEST_ASSERT(ypB);

  PendulumYyBT(data, Y, yyB, ypB);

  N_Vector qB = N_VNew_Serial(PENDULUM_NP + 1, sunctx);
  N_VScale(ZERO, qB, qB);

  /* Initialize backwards problem. */
  int indexB;
  TEST_ASSERT(DDCreateB(dd_mem, &indexB) == IDA_SUCCESS);

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

  /* Initialize backwards quadrature. */
  TEST_ASSERT(DDQuadInitB(dd_mem, indexB, PendulumQuadRhsFnB, qB) == IDA_SUCCESS);

  /* Set up result file */
  FILE* fileb = fopen("pendulum_ASA_backwards.csv", "w");
  TEST_ASSERT(fileb);

  /* ------------------------------------------------------------------------
   * Integrate Backwards
   * ------------------------------------------------------------------------ */

  /* Print datafile header. */
  fprintf(fileb, "t,xB,xpB,xppB,yB,ypB,yppB,λB,dG/dm,dG/dl,dG/dg\n");

  const double* yyB_arr = N_VGetArrayPointer(yyB);
  const double* ypB_arr = N_VGetArrayPointer(ypB);
  const double* qB_arr  = N_VGetArrayPointer(qB);

  while (SUNTRUE)
  {
    fprintf(fileb,
            "%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n",
            tret,
            yyB_arr[0],
            yyB_arr[1],
            yyB_arr[2],
            ypB_arr[3],
            ypB_arr[4],
            ypB_arr[5],
            ypB_arr[6],
            qB_arr[0],
            qB_arr[1],
            qB_arr[2]);

    tret -= tstep;

    if (tret < t0) { break; }

    flag = DDSolveB(dd_mem, tret, IDA_NORMAL);

    TEST_ASSERT(flag >= 0);

    DDGetB(dd_mem, indexB, &tret, yyB, ypB);
    DDGetQuadB(dd_mem, indexB, &tret, qB);
  }

  /* Cleanup */
  DDQuadFree(dd_mem);
  DDAdjFree(dd_mem);
  DDFree(&dd_mem);
  DDSPDestroy(sp);
  DDStagedPivotMatDestroy(pJ0);
  N_VDestroy(Y);
  N_VDestroy(Q);
  N_VDestroy(ypB);
  N_VDestroy(yyB);
  N_VDestroy(qB);
  DDStaticInfoDestroy(si);
  SUNContext_Free(&sunctx);
  SUNLinSolFree(LS);
  SUNLinSolFree(LSB);
  SUNMatDestroy(AB);
  SUNMatDestroy(J);
  SUNMatDestroy(J0);
  fclose(fileb);
  fclose(filef);
  free(data);

  return EXIT_SUCCESS;
}
