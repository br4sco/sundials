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

#include "dd.h"
#include "matrix.h"
#include "models.h"
#include "structure.h"
#include "sundials/sundials_types.h"
#include "sunmatrix/sunmatrix_sparse.h"
#include "test.h"

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

  /* Setup Sundials context. */
  SUNContext sunctx;
  TEST_ASSERT(SUNContext_Create(SUN_COMM_NULL, &sunctx) == SUN_SUCCESS);

  /* Compute DAE structure. */
  const char* eqn_names[] = {"f₁", "f₂"};
  const char* var_names[] = {"x", "y"};

  DAEStruct st = STCreate(sunctx,
                          LOTKA_VOLTERRA_N,
                          LOTKA_VOLTERRA_C,
                          LOTKA_VOLTERRA_D,
                          LOTKA_VOLTERRA_VAR_IDX_MAP,
                          eqn_names,
                          var_names);

  TEST_ASSERT(st);

  const sunindextype N = st->N_all_orders;

  /* Allocate state and Jacobian data. */
  SUNMatrix J0 = SUNDenseMatrix(st->N, st->N, sunctx);
  TEST_ASSERT(J0);

  DDMatrix dd_J0 = DDMatWrapDense(J0);
  TEST_ASSERT(dd_J0);
  N_Vector Y = N_VNew_Serial(N, sunctx);
  TEST_ASSERT(Y);

  /* Allocate forward forward-sensitivity arrays. */
  N_Vector* YS = N_VCloneVectorArray(LOTKA_VOLTERRA_NP, Y);
  TEST_ASSERT(YS);

  /* Set parameters */
  LotkaVolterraParams p = {.a = SUN_RCONST(1.5),
                           .b = SUN_RCONST(1.0),
                           .c = SUN_RCONST(1.0),
                           .d = SUN_RCONST(3.0)};

  /* Set initial values. */
  sunindextype t0 = SUN_RCONST(0.0);
  N_VConst(SUN_RCONST(0.0), Y);
  LV_Ith(Y, 0, 0) = SUN_RCONST(1.0);
  LV_Ith(Y, 0, 1) = p.a - p.b;
  LV_Ith(Y, 1, 0) = SUN_RCONST(1.0);
  LV_Ith(Y, 1, 1) = p.d - p.c;
  for (int i = 0; i < LOTKA_VOLTERRA_NP; ++i)
  {
    N_VConst(SUN_RCONST(0.0), YS[i]);
  }

  /* Compute J0 at the initial time. */
  TEST_ASSERT(SUNMatZero(J0) == SUN_SUCCESS);
  TEST_ASSERT(LotkaVolterraJacf0(t0, Y, J0, &p) == 0);

  /* Create solver session. */
  DDMem dd_mem = DDCreate(sunctx);
  TEST_ASSERT(dd_mem);

  TEST_ASSERT(DDInit(dd_mem,
                     st,
                     SUN_RCONST(0.0),
                     LotkaVolterraJacf0,
                     dd_J0,
                     LotkaVolterraRes,
                     t0,
                     Y) == IDA_SUCCESS);

  TEST_ASSERT(
    DDSensInit(dd_mem, LOTKA_VOLTERRA_NP, IDA_STAGGERED, LotkaVolterraResS, YS) ==
    IDA_SUCCESS);

  /* Set parameters as user data. */
  TEST_ASSERT(DDSetUserData(dd_mem, &p) == IDA_SUCCESS);

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
    J = SUNSparseMatrix(N,
                        N,
                        LOTKA_VOLTERRA_JAC_NNZ + 4,
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
    TEST_ASSERT(DDSetJacFn(dd_mem,
                           (DDLsJacFn){.id = DD_JAC_1,
                                       .fn.jacfn1 = LotkaVolterraJacfn_Dense}) ==
                IDA_SUCCESS);
    break;
  case CSR:
    TEST_ASSERT(DDSetJacFn(dd_mem,
                           (DDLsJacFn){.id        = DD_JAC_1,
                                       .fn.jacfn1 = LotkaVolterraJacfn_CSR}) ==
                IDA_SUCCESS);
    break;
  case CSC:
    TEST_ASSERT(DDSetJacFn(dd_mem,
                           (DDLsJacFn){.id        = DD_JAC_2,
                                       .fn.jacfn2 = LotkaVolterraJacfn_CSC}) ==
                IDA_SUCCESS);
    break;
  }

  /* Set stop time */
  TEST_ASSERT(DDSetStopTime(dd_mem, SUN_RCONST(100.0)) == IDA_SUCCESS);

  /* Set up result file */
  char filename[25];
  sprintf(filename, "lotka-volterra_%c.csv", mat_type);
  FILE* res = fopen(filename, "w");
  TEST_ASSERT(res);

  /* Solve and output solution. */
  fprintf(res, "t,x,y,x(a),y(a),x(b),y(b),x(c),y(c),x(d),y(d),p\n");

  int sr           = IDA_SUCCESS;
  sunrealtype tout = t0;
  sunrealtype tret;

  while (sr != IDA_TSTOP_RETURN)
  {
    PivotResult pr = DDPivot(dd_mem);
    TEST_ASSERT(pr >= 0);

    const sunrealtype x = LV_Ith(Y, 0, 0), y = LV_Ith(Y, 1, 0);
    const sunrealtype xS[] = {LV_Ith(YS[0], 0, 0),
                              LV_Ith(YS[1], 0, 0),
                              LV_Ith(YS[2], 0, 0),
                              LV_Ith(YS[3], 0, 0)};
    const sunrealtype yS[] = {LV_Ith(YS[0], 1, 0),
                              LV_Ith(YS[1], 1, 0),
                              LV_Ith(YS[2], 1, 0),
                              LV_Ith(YS[3], 1, 0)};

    fprintf(res,
            "%.20f,%.20f,%.20f,%.20f,%.20f,%.20f,%.20f,%.20f,%.20f,%.20f,%."
            "20f%d\n",
            tout,
            x,
            y,
            xS[0],
            yS[0],
            xS[1],
            yS[1],
            xS[2],
            yS[2],
            xS[3],
            yS[3],
            pr == PIVOT_SUCCESS ? 1 : 0);

    tout += SUN_RCONST(0.1);
    sr = DDSolve(dd_mem, tout, &tret, Y, IDA_NORMAL);

    TEST_ASSERT((sr == IDA_SUCCESS) || (sr == IDA_TSTOP_RETURN));
    TEST_ASSERT(DDGetSens(dd_mem, &tret, YS) == IDA_SUCCESS);
  }

  /* Cleanup */
  DDFree(&dd_mem);
  DDMatDestroy(dd_J0);
  N_VDestroy(Y);
  N_VDestroyVectorArray(YS, LOTKA_VOLTERRA_NP);
  STDestroy(st);
  SUNContext_Free(&sunctx);
  SUNLinSolFree(LS);
  SUNMatDestroy(J);
  SUNMatDestroy(J0);
  fclose(res);

  return EXIT_SUCCESS;
}
