#include <idas/idas.h>
#include <math.h>
#include <nvector/nvector_serial.h>
#include <stddef.h>
#include <stdio.h>
#include <sundials/sundials_core.h>
#include <sundials/sundials_matrix.h>
#include <sundials/sundials_nvector.h>
#include <sunlinsol/sunlinsol_dense.h>
#include <sunmatrix/sunmatrix_dense.h>

#include "dd.h"
#include "matrix.h"
#include "models.h"
#include "structure.h"
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

  /* Compute DAE structure. */
  Structure* st = STCreate(PENDULUM_N, PENDULUM_C, PENDULUM_D,
                           PENDULUM_EQN_NAMES, PENDULUM_VAR_NAMES);
  TEST_ASSERT(st);

  /* Setup Sundials context. */
  SUNContext ctx;
  TEST_ASSERT(SUNContext_Create(SUN_COMM_NULL, &ctx) == SUN_SUCCESS);

  /* Allocate state and Jacobian data. */
  SUNMatrix J0 = SUNDenseMatrix(st->st_DAE_N, st->st_DAE_N, ctx);
  TEST_ASSERT(J0);
  ExtSUNMatrix* jac0 = ExtSUNMatWrapDense(J0);
  TEST_ASSERT(jac0);
  N_Vector Y = N_VNew_Serial(st->st_N, ctx);
  TEST_ASSERT(Y);

  /* Set DAE parameters. */
  const sunrealtype m = SUN_RCONST(1.1), l = SUN_RCONST(1.2), g = SUN_RCONST(1.3);
  PendulumData* data = malloc(sizeof(*data));
  data->m            = m;
  data->param[0]     = l;
  data->param[1]     = g;

  /* Set initial values. */
  sunrealtype theta0 = SUN_RCONST(M_PI) / FIVE + SUN_RCONST(M_PI) / TWO;
  PendulumY0(data, theta0, Y);

  /* Compute J0 at the initial time. */
  TEST_ASSERT(SUNMatZero(J0) == SUN_SUCCESS);
  TEST_ASSERT(PendulumJacf0(t0, Y, jac0, data) == 0);

  /* Create solver session. */
  DDMem dd_mem = DDCreate(ctx);
  TEST_ASSERT(dd_mem);

  TEST_ASSERT(DDInit(dd_mem, st, SUN_RCONST(0.0), PendulumJacf0, jac0,
                     PendulumRes, t0, Y) == IDA_SUCCESS);

  TEST_ASSERT(DDSetUserData(dd_mem, data) == IDA_SUCCESS);

  TEST_ASSERT(DDSSTolerances(dd_mem, SUN_RCONST(1.0e-9), SUN_RCONST(1.0e-9)) ==
              IDA_SUCCESS);

  /* Setup and set linear solver. */
  SUNMatrix A = SUNDenseMatrix(st->st_N, st->st_N, ctx);
  TEST_ASSERT(A);
  SUNLinearSolver LS = SUNLinSol_Dense(Y, A, ctx);
  TEST_ASSERT(LS);

  TEST_ASSERT(DDSetLinearSolver(dd_mem, LS, A) == IDA_SUCCESS);

  /* Set stop time */
  TEST_ASSERT(DDSetStopTime(dd_mem, tout) == IDA_SUCCESS);

  /* Set up result file */
  FILE* file = fopen("pendulum.csv", "w");
  TEST_ASSERT(file);

  /* Solve and output solution. */
  fprintf(file, "t,x,y,λ,ΔL,p\n"); /* print header */

  int flag      = IDA_SUCCESS;
  sunrealtype t = t0;
  sunrealtype tret;

  while (flag != IDA_TSTOP_RETURN)
  {
    PivotResult pr = DDPivot(dd_mem);
    TEST_ASSERT(pr >= 0);

    const sunrealtype x = P_X(Y, 0), y = P_Y(Y, 0), lam = P_LAMBDA(Y);

    fprintf(file, "%.20f,%.20f,%.20f,%.20f,%.20f,%d\n", t, x, y, lam,
            x * x + y * y - l * l, pr == PIVOT_SUCCESS ? 1 : 0);

    t += tstep;

    flag = DDSolve(dd_mem, t, &tret, Y, IDA_NORMAL);
    TEST_ASSERT(flag >= 0);
  }

  /* Cleanup */
  DDFree(&dd_mem);
  ExtSUNMatDestroy(jac0);
  N_VDestroy(Y);
  STDestroy(st);
  SUNContext_Free(&ctx);
  SUNLinSolFree(LS);
  SUNMatDestroy(J0);
  SUNMatDestroy(A);
  fclose(file);
  free(data);

  return EXIT_SUCCESS;
}
