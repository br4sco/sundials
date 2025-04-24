#include <idas/idas.h>
#include <math.h>
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

int main(int argc, char* argv[])
{
  enum
  {
    NONE  = 'n',
    DENSE = 'd',
    CSR   = 'r',
    CSC   = 'c'
  } mat_type;

  TEST_ASSERT(argc > 1);
  switch (argv[1][0])
  {
  case NONE: mat_type = NONE; break;
  case DENSE: mat_type = DENSE; break;
  case CSR: mat_type = CSR; break;
  case CSC: mat_type = CSC; break;
  default: TEST_ASSERT(0);
  }

  const sunrealtype t0    = ZERO;
  const sunrealtype tstep = SUN_RCONST(0.1);
  const sunrealtype tout  = SUN_RCONST(100.0);

  /* Compute DAE structure. */
  Structure* st = STCreate(PENDULUM_N, PENDULUM_C, PENDULUM_D,
                           PENDULUM_EQN_NAMES, PENDULUM_VAR_NAMES);
  TEST_ASSERT(st);

  const sunindextype N = st->st_N;

  /* Setup Sundials context. */
  SUNContext sunctx;
  TEST_ASSERT(SUNContext_Create(SUN_COMM_NULL, &sunctx) == SUN_SUCCESS);

  /* Allocate state and Jacobian data. */
  SUNMatrix J0 = SUNDenseMatrix(st->st_DAE_N, st->st_DAE_N, sunctx);
  TEST_ASSERT(J0);
  DDMatrix* dd_J0 = DDMatWrapDense(J0);
  TEST_ASSERT(dd_J0);
  N_Vector Y = N_VNew_Serial(N, sunctx);
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
  TEST_ASSERT(PendulumJacf0(t0, Y, dd_J0, data) == 0);

  /* Create solver session. */
  DDMem dd_mem = DDCreate(sunctx);
  TEST_ASSERT(dd_mem);

  TEST_ASSERT(DDInit(dd_mem, st, SUN_RCONST(0.0), PendulumJacf0, dd_J0,
                     PendulumRes, t0, Y) == IDA_SUCCESS);

  TEST_ASSERT(DDSetUserData(dd_mem, data) == IDA_SUCCESS);

  TEST_ASSERT(DDSSTolerances(dd_mem, SUN_RCONST(1.0e-9), SUN_RCONST(1.0e-9)) ==
              IDA_SUCCESS);

  /* Setup and set linear solver. */
  SUNMatrix J        = NULL;
  SUNLinearSolver LS = NULL;

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
    J = SUNSparseMatrix(N, N, PENDULUM_JAC_NNZ + 4,
                        mat_type == CSR ? CSR_MAT : CSC_MAT, sunctx);
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
    TEST_ASSERT(DDSetJacFn(dd_mem, (DDLsJacFn){.id = DD_JAC_1,
                                               .fn.jacfn1 = PendulumJacfn_Dense}) ==
                IDA_SUCCESS);
    break;
  case CSR:
    TEST_ASSERT(DDSetJacFn(dd_mem, (DDLsJacFn){.id = DD_JAC_1,
                                               .fn.jacfn1 = PendulumJacfn_CSR}) ==
                IDA_SUCCESS);
    break;
  case CSC:
    TEST_ASSERT(DDSetJacFn(dd_mem, (DDLsJacFn){.id = DD_JAC_2,
                                               .fn.jacfn2 = PendulumJacfn_CSC}) ==
                IDA_SUCCESS);
    break;
  }

  /* Set stop time */
  TEST_ASSERT(DDSetStopTime(dd_mem, tout) == IDA_SUCCESS);

  /* Set up result file */
  char filename[25];
  sprintf(filename, "pendulum_%c.csv", mat_type);
  FILE* file = fopen(filename, "w");
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
  DDMatDestroy(dd_J0);
  N_VDestroy(Y);
  STDestroy(st);
  SUNContext_Free(&sunctx);
  SUNLinSolFree(LS);
  SUNMatDestroy(J);
  SUNMatDestroy(J0);
  fclose(file);
  free(data);

  return EXIT_SUCCESS;
}
