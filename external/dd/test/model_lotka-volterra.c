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
#include "matrix.h"
#include "models.h"
#include "structure.h"
#include "sundials/sundials_types.h"
#include "test.h"

int main(void)
{
  /* Compute DAE structure. */
  char* eqn_names[] = {"f₁", "f₂"};
  char* var_names[] = {"x", "y"};
  Structure* st = STCreate(LOTKA_VOLTERRA_N, LOTKA_VOLTERRA_C, LOTKA_VOLTERRA_D,
                           eqn_names, var_names);
  TEST_ASSERT(st);

  /* Setup Sundials context. */
  SUNContext ctx;
  TEST_ASSERT(SUNContext_Create(SUN_COMM_NULL, &ctx) == SUN_SUCCESS);

  /* Allocate state and Jacobian data. */
  SUNMatrix J0 = SUNDenseMatrix(st->st_DAE_N, st->st_DAE_N, ctx);
  TEST_ASSERT(J0);

  DDMatrix* jac0 = DDMatWrapDense(J0);
  TEST_ASSERT(jac0);
  N_Vector yy = N_VNew_Serial(st->st_N, ctx);
  TEST_ASSERT(yy);

  /* Allocate forward forward-sensitivity arrays. */
  N_Vector* yyS = N_VCloneVectorArray(LOTKA_VOLTERRA_NP, yy);
  TEST_ASSERT(yyS);

  /* Set parameters */
  LotkaVolterraParams p = {.a = SUN_RCONST(1.5),
                           .b = SUN_RCONST(1.0),
                           .c = SUN_RCONST(1.0),
                           .d = SUN_RCONST(3.0)};

  /* Set initial values. */
  sunindextype t0 = SUN_RCONST(0.0);
  N_VConst(SUN_RCONST(0.0), yy);
  LV_X(yy, 0) = SUN_RCONST(1.0);
  LV_X(yy, 1) = p.a - p.b;
  LV_Y(yy, 0) = SUN_RCONST(1.0);
  LV_Y(yy, 1) = p.d - p.c;
  for (int i = 0; i < LOTKA_VOLTERRA_NP; ++i)
  {
    N_VConst(SUN_RCONST(0.0), yyS[i]);
  }

  /* Compute J0 at the initial time. */
  TEST_ASSERT(SUNMatZero(J0) == SUN_SUCCESS);
  TEST_ASSERT(LotkaVolterraJacf0(t0, yy, jac0, &p) == 0);

  /* Create solver session. */
  DDMem dd_mem = DDCreate(ctx);
  TEST_ASSERT(dd_mem);

  TEST_ASSERT(DDInit(dd_mem, st, SUN_RCONST(0.0), LotkaVolterraJacf0, jac0,
                     LotkaVolterraRes, t0, yy) == IDA_SUCCESS);

  TEST_ASSERT(DDSensInit(dd_mem, LOTKA_VOLTERRA_NP, IDA_STAGGERED,
                         LotkaVolterraResS, yyS) == IDA_SUCCESS);

  /* Set parameters as user data. */
  TEST_ASSERT(DDSetUserData(dd_mem, &p) == IDA_SUCCESS);

  TEST_ASSERT(DDSSTolerances(dd_mem, SUN_RCONST(1.0e-9), SUN_RCONST(1.0e-9)) ==
              IDA_SUCCESS);

  /* Setup and set linear solver. */
  SUNMatrix jac = SUNDenseMatrix(st->st_N, st->st_N, ctx);
  TEST_ASSERT(jac);
  SUNLinearSolver ls = SUNLinSol_Dense(yy, jac, ctx);
  TEST_ASSERT(ls);

  TEST_ASSERT(DDSetLinearSolver(dd_mem, ls, jac) == IDA_SUCCESS);

  /* Set stop time */
  TEST_ASSERT(DDSetStopTime(dd_mem, SUN_RCONST(100.0)) == IDA_SUCCESS);

  /* Set up result file */
  FILE* res = fopen("lotka-volterra.csv", "w");
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

    const sunrealtype x = LV_X(yy, 0), y = LV_Y(yy, 0);
    const sunrealtype xS[] = {LV_X(yyS[0], 0), LV_X(yyS[1], 0), LV_X(yyS[2], 0),
                              LV_X(yyS[3], 0)},
                      yS[] = {LV_Y(yyS[0], 0), LV_Y(yyS[1], 0), LV_Y(yyS[2], 0),
                              LV_Y(yyS[3], 0)};

    fprintf(res,
            "%.20f,%.20f,%.20f,%.20f,%.20f,%.20f,%.20f,%.20f,%.20f,%.20f,%."
            "20f%d\n",
            tout, x, y, xS[0], yS[0], xS[1], yS[1], xS[2], yS[2], xS[3], yS[3],
            pr == PIVOT_SUCCESS ? 1 : 0);

    tout += SUN_RCONST(0.1);
    sr = DDSolve(dd_mem, tout, &tret, yy, IDA_NORMAL);

    TEST_ASSERT((sr == IDA_SUCCESS) || (sr == IDA_TSTOP_RETURN));
    TEST_ASSERT(DDGetSens(dd_mem, &tret, yyS) == IDA_SUCCESS);
  }

  /* Cleanup */
  fclose(res);
  SUNLinSolFree(ls);
  SUNMatDestroy(jac);
  DDFree(&dd_mem);
  N_VDestroy(yy);
  N_VDestroyVectorArray(yyS, LOTKA_VOLTERRA_NP);
  DDMatDestroy(jac0);
  SUNMatDestroy(J0);
  STDestroy(st);
  SUNContext_Free(&ctx);

  return EXIT_SUCCESS;
}
