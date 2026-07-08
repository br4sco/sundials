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

  /* Allocate state and Jacobian data. */
  SUNMatrix J0 = SUNDenseMatrix(si->N, si->N, sunctx);
  TEST_ASSERT(J0);
  PIVMatrix pJ0 = PIVMatWrapDense(J0);
  TEST_ASSERT(pJ0);
  N_Vector Y = N_VNew_Serial(si->N_all_orders, sunctx);
  TEST_ASSERT(Y);

  /* Set DAE parameters. */
  const sunrealtype m = SUN_RCONST(1.1), l = SUN_RCONST(1.2), g = SUN_RCONST(1.3);
  PendulumData* data = malloc(sizeof(*data));
  data->m            = m;
  data->param[0]     = l;
  data->param[1]     = g;

  /* Set initial values. */
  sunrealtype theta0 = SUN_RCONST(PI) / FIVE + SUN_RCONST(PI) / TWO;
  PendulumY0(data, theta0, Y);

  /* Create pivot memory and compute initial spec. */
  PivMem pm = PIVCreate(sunctx, si, pJ0, PendulumJacf0);
  TEST_ASSERT(pm);
  TEST_ASSERT(PIVSetUserData(pm, data) == SUN_SUCCESS);
  sunbooleantype spec_changed;
  TEST_ASSERT(PIVPivot(pm, ZERO, t0, Y, &spec_changed) == SUN_SUCCESS);

  /* Create solver session. */
  DDMem dd_mem = DDCreate(sunctx);
  TEST_ASSERT(dd_mem);

  TEST_ASSERT(DDInit(dd_mem, si, PendulumRes, PIVGetSpec(pm), t0, Y) ==
              IDA_SUCCESS);

  TEST_ASSERT(DDAdjInit(dd_mem, Nd, IDA_POLYNOMIAL) == IDA_SUCCESS);

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
  fprintf(filef, "t,x,y,λ,ΔL,p\n"); /* print header */

  int flag         = IDA_SUCCESS;
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

    const sunrealtype x = P_Ith(Y, 0, 0), y = P_Ith(Y, 1, 0),
                      lam = P_Ith(Y, 2, 0);

    fprintf(filef,
            "%.20f,%.20f,%.20f,%.20f,%.20f,%d\n",
            t,
            x,
            y,
            lam,
            x * x + y * y - l * l,
            spec_changed ? 1 : 0);

    t += tstep;

    if (t >= tout) { break; }

    TEST_ASSERT(DDSolveF(dd_mem, t, &tret, Y, IDA_NORMAL, &ncheck) == IDA_SUCCESS);
  }

  /* ------------------------------------------------------------------------
   * Setup Backwards Problem
   * ------------------------------------------------------------------------ */

  /* Allocate state and set initial values */
  N_Vector yyB = N_VNew_Serial(si->N_backwards, sunctx);
  TEST_ASSERT(yyB);

  N_Vector ypB = N_VClone(yyB);
  TEST_ASSERT(ypB);

  PendulumYyBT(data, Y, yyB, ypB);

  /* Initialize backwards problem. */
  int indexB;
  TEST_ASSERT(DDCreateB(dd_mem, &indexB) == IDA_SUCCESS);

  TEST_ASSERT(DDInitB(dd_mem, indexB, PendulumResB, tret, yyB, ypB) ==
              IDA_SUCCESS);

  TEST_ASSERT(
    DDSStolerancesB(dd_mem, indexB, SUN_RCONST(1.e-6), SUN_RCONST(1.e-6)) ==
    IDA_SUCCESS);

  TEST_ASSERT(DDSetUserDataB(dd_mem, indexB, data) == IDA_SUCCESS);

  SUNMatrix AB = SUNDenseMatrix(si->N_backwards, si->N_backwards, sunctx);
  TEST_ASSERT(AB);
  SUNLinearSolver LSB = SUNLinSol_Dense(yyB, AB, sunctx);
  TEST_ASSERT(LSB);

  TEST_ASSERT(DDSetLinearSolverB(dd_mem, indexB, LSB, AB) == IDA_SUCCESS);
  TEST_ASSERT(DDSetJacFnB(dd_mem, indexB, PendulumJacFnB) == IDA_SUCCESS);

  /* Set up result file */
  FILE* fileb = fopen("pendulum_ASA_backwards.csv", "w");
  TEST_ASSERT(fileb);

  /* ------------------------------------------------------------------------
   * Integrate Backwards
   * ------------------------------------------------------------------------ */

  /* Print datafile header. */
  fprintf(fileb, "t,xB,yB,λB,dxB,dyB,dλB\n");

  double* yyB_arr = N_VGetArrayPointer(yyB);
  double* ypB_arr = N_VGetArrayPointer(ypB);

  while (SUNTRUE)
  {
    fprintf(fileb,
            "%f,%f,%f,%f,%f,%f,%f\n",
            tret,
            yyB_arr[0],
            yyB_arr[2],
            yyB_arr[4],
            ypB_arr[0],
            ypB_arr[2],
            ypB_arr[4]);

    tret -= tstep;

    if (tret < t0) { break; }

    flag = DDSolveB(dd_mem, tret, IDA_NORMAL);

    TEST_ASSERT(flag >= 0);

    DDGetB(dd_mem, indexB, &tret, yyB, ypB);
  }

  /* Cleanup */
  DDFree(&dd_mem);
  PIVDestroy(&pm);
  PIVMatDestroy(pJ0);
  N_VDestroy(Y);
  N_VDestroy(ypB);
  N_VDestroy(yyB);
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
