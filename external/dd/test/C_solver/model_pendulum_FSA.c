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

int main(int argc, char* argv[])
{
  sunbooleantype analytic_sens_residual = argc > 1;
  const sunrealtype t0                  = ZERO;
  const sunrealtype tstep               = SUN_RCONST(0.1);
  const sunrealtype tout                = SUN_RCONST(100.0);

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

  /* Allocate state and Jacobian data. */
  SUNMatrix J0 = SUNDenseMatrix(si->N, si->N, sunctx);
  TEST_ASSERT(J0);
  PIVMatrix pJ0 = PIVMatWrapDense(J0);
  TEST_ASSERT(pJ0);
  N_Vector Y = N_VNew_Serial(si->N_all_orders, sunctx);
  TEST_ASSERT(Y);

  /* Allocate forward forward-sensitivity arrays. */
  N_Vector* YS = N_VCloneVectorArray(PENDULUM_NP, Y);
  TEST_ASSERT(YS);

  /* Set DAE parameters. */
  const sunrealtype m = SUN_RCONST(1.1), l = SUN_RCONST(1.2), g = SUN_RCONST(1.3);
  PendulumData* data = malloc(sizeof(*data));
  data->m            = m;
  data->param[0]     = l;
  data->param[1]     = g;

  /* Set initial values. */
  sunrealtype theta0 = SUN_RCONST(PI) / FIVE + SUN_RCONST(PI) / TWO;
  PendulumY0(data, theta0, Y);
  PendulumYS0(data, theta0, YS);

  /* Create pivot memory and compute initial spec. */
  PIVMem pm = PIVCreate(sunctx, si, pJ0, PendulumJacf0);
  TEST_ASSERT(pm);
  TEST_ASSERT(PIVSetUserData(pm, data) == SUN_SUCCESS);
  sunbooleantype spec_changed;
  TEST_ASSERT(PIVPivot(pm, ZERO, t0, Y, &spec_changed) == SUN_SUCCESS);

  /* Create solver session. */
  DDMem dd_mem = DDCreate(sunctx);
  TEST_ASSERT(dd_mem);

  TEST_ASSERT(DDInit(dd_mem, si, PendulumRes, pm->spec, t0, Y) == IDA_SUCCESS);

  TEST_ASSERT(DDSensInit(dd_mem,
                         PENDULUM_NP,
                         IDA_STAGGERED,
                         analytic_sens_residual ? PendulumResS : NULL,
                         YS) == IDA_SUCCESS);

  TEST_ASSERT(DDSetUserData(dd_mem, data) == IDA_SUCCESS);

  TEST_ASSERT(DDSSTolerances(dd_mem, SUN_RCONST(1.0e-6), SUN_RCONST(1.0e-7)) ==
              IDA_SUCCESS);

  TEST_ASSERT(DDSensEEtolerances(dd_mem) == IDA_SUCCESS);
  sunrealtype pbar[] = {data->param[0], data->param[1]};
  TEST_ASSERT(DDSetSensParams(dd_mem, data->param, pbar, NULL) == IDA_SUCCESS)

  /* Setup and set linear solver. */
  SUNMatrix J = SUNDenseMatrix(si->N_all_orders, si->N_all_orders, sunctx);
  TEST_ASSERT(J);
  SUNLinearSolver LS = SUNLinSol_Dense(Y, J, sunctx);
  TEST_ASSERT(LS);

  TEST_ASSERT(DDSetLinearSolver(dd_mem, LS, J) == IDA_SUCCESS);

  /* Set stop time */
  TEST_ASSERT(DDSetStopTime(dd_mem, tout) == IDA_SUCCESS);

  /* Set up result file */
  FILE* file = fopen(analytic_sens_residual
                       ? "pendulum_analytic_sens_residual_FSA.csv"
                       : "pendulum_FSA.csv",
                     "w");
  TEST_ASSERT(file);

  /* Solve and output solution. */
  fprintf(file, "t,x,y,λ,xl,yl,λl,xg,yg,λg,ΔL,p\n");

  int flag      = IDA_SUCCESS;
  sunrealtype t = t0;
  sunrealtype tret;

  while (flag != IDA_TSTOP_RETURN)
  {
    TEST_ASSERT(PIVPivot(pm, ZERO, t, Y, &spec_changed) == SUN_SUCCESS);
    if (spec_changed)
    {
      TEST_ASSERT(DDSetSpec(dd_mem, pm->spec) == SUN_SUCCESS);
    }

    const sunrealtype x = P_Ith(Y, 0, 0), y = P_Ith(Y, 1, 0),
                      lam = P_Ith(Y, 2, 0),

                      xl = P_Ith(YS[0], 0, 0), yl = P_Ith(YS[0], 1, 0),
                      laml = P_Ith(YS[0], 2, 0),

                      xg = P_Ith(YS[1], 0, 0), yg = P_Ith(YS[1], 1, 0),
                      lamg = P_Ith(YS[1], 2, 0);

    fprintf(file,
            "%.20f,%.20f,%.20f,%.20f,%.20f,%.20f,%.20f,%.20f,%.20f,%.20f,%."
            "20f,%d\n",
            t,
            x,
            y,
            lam,
            xl,
            yl,
            laml,
            xg,
            yg,
            lamg,
            x * x + y * y - l * l,
            spec_changed ? 1 : 0);

    t += tstep;
    flag = DDSolve(dd_mem, t, &tret, Y, IDA_NORMAL);

    TEST_ASSERT((flag == IDA_SUCCESS) || (flag == IDA_TSTOP_RETURN));
    TEST_ASSERT(DDGetSens(dd_mem, &tret, YS) == IDA_SUCCESS);
  }

  /* Cleanup */
  DDFree(&dd_mem);
  PIVDestroy(&pm);
  PIVMatDestroy(pJ0);
  N_VDestroy(Y);
  N_VDestroyVectorArray(YS, PENDULUM_NP);
  DDStaticInfoDestroy(si);
  SUNContext_Free(&sunctx);
  SUNLinSolFree(LS);
  SUNMatDestroy(J);
  SUNMatDestroy(J0);
  fclose(file);
  free(data);

  return EXIT_SUCCESS;
}
