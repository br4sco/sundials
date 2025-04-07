/* -----------------------------------------------------------------------------
 * Programmer(s): Oscar Eriksson @ KTH
 * -----------------------------------------------------------------------------
 * -----------------------------------------------------------------------------
 * Unit test for manually adding checkpoints
 * ---------------------------------------------------------------------------*/

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include "idas/idas.h"
#include "nvector/nvector_serial.h"
#include "sundials/sundials_matrix.h"
#include "sundials/sundials_nvector.h"
#include "sunlinsol/sunlinsol_dense.h"
#include "sunmatrix/sunmatrix_dense.h"

#define ZERO SUN_RCONST(0.0)
#define ONE  SUN_RCONST(1.0)

typedef struct
{
  sunrealtype params[2];
} Data;

static int dae_res(sunrealtype t, N_Vector yy, N_Vector yp, N_Vector rr,
                   void* user_data)
{
  Data* data          = (Data*)user_data;
  const sunrealtype a = data->params[0];
  const sunrealtype b = data->params[1];

  sunrealtype* yy_arr = N_VGetArrayPointer(yy);
  sunrealtype* yp_arr = N_VGetArrayPointer(yp);

  sunrealtype* rr_arr = N_VGetArrayPointer(rr);

  rr_arr[0] = yp_arr[0] + a * yy_arr[0] - b;

  return 0;
}

static int dae_jac(sunrealtype t, sunrealtype cj, N_Vector yy, N_Vector yp,
                   N_Vector rr, SUNMatrix J, void* user_data, N_Vector tmp1,
                   N_Vector tmp2, N_Vector tmp3)
{
  Data* data          = (Data*)user_data;
  const sunrealtype a = data->params[0];

  sunrealtype* J_arr = SUNDenseMatrix_Data(J);

  J_arr[0] = a + cj;

  return 0;
}

/* g = y */

static int dae_resB(sunrealtype t, N_Vector yy, N_Vector yp, N_Vector yyB,
                    N_Vector ypB, N_Vector rr, void* user_data)
{
  Data* data          = (Data*)user_data;
  const sunrealtype a = data->params[0];

  sunrealtype* yyB_arr = N_VGetArrayPointer(yyB);
  sunrealtype* ypB_arr = N_VGetArrayPointer(ypB);
  sunrealtype* rr_arr  = N_VGetArrayPointer(rr);

  rr_arr[0] = ypB_arr[0] - a * yyB_arr[0] + ONE;

  return 0;
}

static int dae_jacB(sunrealtype t, sunrealtype cj, N_Vector yy, N_Vector yp,
                    N_Vector yyB, N_Vector ypB, N_Vector rr, SUNMatrix J,
                    void* user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3)
{
  Data* data          = (Data*)user_data;
  const sunrealtype a = data->params[0];

  sunrealtype* J_arr = SUNDenseMatrix_Data(J);

  J_arr[0] = -a + cj;

  return 0;
}

int main(void)
{
  const int neq            = 1;
  const int Nd             = 10;
  const int Nt             = 1000;
  const sunrealtype eps    = SUN_RCONST(1.0e-14);
  const sunrealtype dt     = SUN_RCONST(0.10);
  const sunrealtype reltol = SUN_RCONST(1.0e-9);
  const sunrealtype abstol = SUN_RCONST(1.0e-9);
  const sunrealtype a      = SUN_RCONST(1.1);
  const sunrealtype b      = SUN_RCONST(1.2);
  const sunrealtype y0     = ZERO;
  const sunrealtype yp0    = b / a;
  const sunrealtype yBT    = ZERO;
  const sunrealtype ypBT   = -ONE / a;
  Data data                = {.params = {a, b}};

  /* --------------
   * Create context
   * -------------- */

  SUNContext sunctx = NULL;

  int flag = SUNContext_Create(SUN_COMM_NULL, &sunctx);
  if (flag != SUN_SUCCESS)
  {
    fprintf(stderr, "SUNContext_Create returned %i\n", flag);
    return EXIT_FAILURE;
  }

  /* -----------------------
   * Setup initial condition
   * ----------------------- */

  N_Vector yy = N_VNew_Serial(neq, sunctx);
  if (!yy) { return EXIT_FAILURE; }
  N_VConst(y0, yy);

  N_Vector yp = N_VClone(yy);
  if (!yp) { return EXIT_FAILURE; }
  N_VConst(yp0, yp);

  /* --------------------------
   * Setup IDAS Forward Problem
   * -------------------------- */

  void* ida_mem = IDACreate(sunctx);
  if (!ida_mem) { return EXIT_FAILURE; }

  if (IDAInit(ida_mem, dae_res, ZERO, yy, yp) != IDA_SUCCESS)
  {
    return EXIT_FAILURE;
  }

  if (IDASetUserData(ida_mem, &data) != IDA_SUCCESS) { return EXIT_FAILURE; }

  if (IDASStolerances(ida_mem, reltol, abstol) != IDA_SUCCESS)
  {
    return EXIT_FAILURE;
  }

  SUNMatrix A = SUNDenseMatrix(neq, neq, sunctx);
  if (!A) { return EXIT_FAILURE; }

  SUNLinearSolver LS = SUNLinSol_Dense(yy, A, sunctx);
  if (!LS) { return EXIT_FAILURE; }

  if (IDASetLinearSolver(ida_mem, LS, A) != IDA_SUCCESS)
  {
    return EXIT_FAILURE;
  }

  if (IDASetJacFn(ida_mem, dae_jac) != IDA_SUCCESS) { return EXIT_FAILURE; }

  if (IDAAdjInit(ida_mem, Nd, IDA_POLYNOMIAL) != IDA_SUCCESS)
  {
    return EXIT_FAILURE;
  }

  /* ---------------
   * Advance in time
   * --------------- */

  sunrealtype tret = ZERO;
  int nckpnts1     = 0;

  if (IDASolveF(ida_mem, Nt * dt, &tret, yy, yp, IDA_NORMAL, &nckpnts1) !=
      IDA_SUCCESS)
  {
    return EXIT_FAILURE;
  }

  if (nckpnts1 <= 0)
  {
    fprintf(stderr, "Number of checkpoints are %d\n", nckpnts1);
    return EXIT_FAILURE;
  }

  const sunrealtype yT = NV_Ith_S(yy, 0);

  fprintf(stderr, "--- Start saved checkpoints ---\n");
  IDAadjCheckPointRec* chkpnts = malloc((nckpnts1 + 1) * sizeof(*chkpnts));
  if (IDAGetAdjCheckPointsInfo(ida_mem, chkpnts) != IDA_SUCCESS)
  {
    return EXIT_FAILURE;
  }

  for (int i = 0; i < nckpnts1; ++i)
  {
    fprintf(stderr, "t1 = %.50f\nt0 = %.50f\n\n", chkpnts[i].t1, chkpnts[i].t0);
  }

  free(chkpnts);
  fprintf(stderr, "--- End saved checkpoints ---\n");

  /* ------------------------------------------------
   * Setup initial condition for the backward problem
   * ------------------------------------------------ */

  N_Vector yyB = N_VClone(yy);
  if (!yyB) { return EXIT_FAILURE; }
  N_VConst(yBT, yyB);

  N_Vector ypB = N_VClone(yy);
  if (!ypB) { return EXIT_FAILURE; }
  N_VConst(ypBT, ypB);

  /* ---------------------------
   * Setup IDAS Backward Problem
   * --------------------------- */

  int which;
  if (IDACreateB(ida_mem, &which) != IDA_SUCCESS) { return EXIT_FAILURE; }

  if (IDAInitB(ida_mem, which, dae_resB, tret, yyB, ypB) != IDA_SUCCESS)
  {
    return EXIT_FAILURE;
  }

  if (IDASetUserDataB(ida_mem, which, &data) != IDA_SUCCESS)
  {
    return EXIT_FAILURE;
  }

  if (IDASStolerancesB(ida_mem, which, reltol, abstol) != IDA_SUCCESS)
  {
    return EXIT_FAILURE;
  }

  SUNMatrix AB = SUNDenseMatrix(neq, neq, sunctx);
  if (!AB) { return EXIT_FAILURE; }

  SUNLinearSolver LSB = SUNLinSol_Dense(yyB, AB, sunctx);
  if (!LSB) { return EXIT_FAILURE; }

  if (IDASetLinearSolverB(ida_mem, which, LSB, AB) != IDA_SUCCESS)
  {
    return EXIT_FAILURE;
  }

  if (IDASetJacFnB(ida_mem, which, dae_jacB) != IDA_SUCCESS)
  {
    return EXIT_FAILURE;
  }

  /* -------------------------
   * Advance backwards in time
   * ------------------------- */

  flag = IDASolveB(ida_mem, ZERO, IDA_NORMAL);
  if (flag < 0)
  {
    fprintf(stderr, "IDASolveB returned %i\n", flag);
    return EXIT_FAILURE;
  }

  sunrealtype t0 = ZERO;
  if (IDAGetB(ida_mem, which, &t0, yyB, ypB) != IDA_SUCCESS)
  {
    return EXIT_FAILURE;
  }

  if (t0 > ZERO)
  {
    fprintf(stderr, "IDASolveB solved backwards until t = %f\n", t0);
    return EXIT_FAILURE;
  }

  const sunrealtype yB0 = NV_Ith_S(yyB, 0);

  /* -----------------------
   * Setup initial condition
   * ----------------------- */

  N_VConst(y0, yy);
  N_VConst(yp0, yp);

  /* -----------------------------------
   * Setup Re-initialize Forward Problem
   * ----------------------------------- */

  if (IDAReInit(ida_mem, ZERO, yy, yp) != IDA_SUCCESS) { return EXIT_FAILURE; }

  if (IDAAdjReInit(ida_mem) != IDA_SUCCESS) { return EXIT_FAILURE; }

  /* ---------------
   * Advance in time
   * --------------- */
  int nckpnts2 = 0;

  fprintf(stderr, "--- Start forward integration ---\n");
  for (int i = 1; i <= Nt; ++i)
  {
    if (IDASolveF(ida_mem, i * dt, &tret, yy, yp, IDA_NORMAL, &nckpnts2) !=
        IDA_SUCCESS)
    {
      return EXIT_FAILURE;
    }

    sunrealtype tcur;
    int nckpnts;
    if (IDAStoreCheckPoint(ida_mem, &tcur, &nckpnts) != IDA_SUCCESS)
    {
      return EXIT_FAILURE;
    }

    if (nckpnts > nckpnts2)
    {
      fprintf(stderr, "Stored checkpoint at t = %f\n", tcur);
    }
    nckpnts2 = nckpnts;
  }
  fprintf(stderr, "--- End forward integration ---\n");

  if (nckpnts2 <= nckpnts1)
  {
    fprintf(stderr, "Number of checkpoints are %d and %d\n", nckpnts1, nckpnts2);
    return EXIT_FAILURE;
  }

  sunrealtype yTerr = SUNRabs(yT - NV_Ith_S(yy, 0));
  if (yTerr > eps)
  {
    fprintf(stderr, "yT error is %.50f\n", yTerr);
    return EXIT_FAILURE;
  }

  fprintf(stderr, "--- Start saved checkpoints ---\n");
  chkpnts = malloc((nckpnts2 + 1) * sizeof(*chkpnts));
  if (IDAGetAdjCheckPointsInfo(ida_mem, chkpnts) != IDA_SUCCESS)
  {
    return EXIT_FAILURE;
  }

  for (int i = 0; i < nckpnts2; ++i)
  {
    fprintf(stderr, "t1 = %.50f\nt0 = %.50f\n\n", chkpnts[i].t1, chkpnts[i].t0);
  }

  free(chkpnts);
  fprintf(stderr, "--- End saved checkpoints ---\n");

  /* ------------------------------------------------
   * Setup initial condition for the backward problem
   * ------------------------------------------------ */

  N_VConst(yBT, yyB);
  N_VConst(ypBT, ypB);

  /* ------------------------------
   * Re-initialize Backward Problem
   * ------------------------------ */

  if (IDAReInitB(ida_mem, which, tret, yyB, ypB) != IDA_SUCCESS)
  {
    return EXIT_FAILURE;
  }

  flag = IDASolveB(ida_mem, ZERO, IDA_NORMAL);
  if (flag < 0)
  {
    fprintf(stderr, "IDASolveB returned %i\n", flag);
    return EXIT_FAILURE;
  }

  t0 = ZERO;
  if (IDAGetB(ida_mem, which, &t0, yyB, ypB) != IDA_SUCCESS)
  {
    return EXIT_FAILURE;
  }

  if (t0 > ZERO)
  {
    fprintf(stderr, "IDASolveB solved backwards until t = %f\n", t0);
    return EXIT_FAILURE;
  }

  sunrealtype yB0err = SUNRabs(yB0 - NV_Ith_S(yyB, 0));
  if (yB0err > eps)
  {
    fprintf(stderr, "yB0 error is %.50f\n", yB0err);
    return EXIT_FAILURE;
  }

  /* --------
   * Clean up
   * -------- */

  IDAFree(&ida_mem);
  N_VDestroy(yy);
  N_VDestroy(yp);
  SUNMatDestroy(A);
  SUNLinSolFree(LS);
  N_VDestroy(yyB);
  N_VDestroy(ypB);
  SUNMatDestroy(AB);
  SUNLinSolFree(LSB);
  SUNContext_Free(&sunctx);

  printf("SUCCESS\n");

  return EXIT_SUCCESS;
}
