#include <stdbool.h>
#include <stdlib.h>

#include <sunmatrix/sunmatrix_sparse.h>

#include "matrix.h"
#include "models.h"
#include "test.h"

int main(void)
{
  SUNContext ctx;
  SUNContext_Create(SUN_COMM_NULL, &ctx);

  Structure* st;
  SUNMatrix A, B;

  /* ------------------------------------------------------------------------
   * Lotka-Volterra
   * ------------------------------------------------------------------------ */

  st = STCreate(LOTKA_VOLTERRA_N, LOTKA_VOLTERRA_C, LOTKA_VOLTERRA_D, NULL, NULL);
  TEST_ASSERT(st != NULL);

  A = DDSparseSUNMatFromStructure(st, LOTKA_VOLTERRA_JAC_NNZ, CSC_MAT, ctx);
  TEST_ASSERT(A != NULL);
  TEST_ASSERT(SM_NNZ_S(A) == LOTKA_VOLTERRA_JAC_NNZ + 4);

  B = DDSparseSUNMatFromStructure(st, LOTKA_VOLTERRA_JAC_NNZ, CSR_MAT, ctx);
  TEST_ASSERT(B != NULL);
  TEST_ASSERT(SM_NNZ_S(B) == LOTKA_VOLTERRA_JAC_NNZ + 4);

  STDestroy(st);
  SUNMatDestroy(A);
  SUNMatDestroy(B);

  /* ------------------------------------------------------------------------
   * Pendulum
   * ------------------------------------------------------------------------ */

  st = STCreate(PENDULUM_N, PENDULUM_C, PENDULUM_D, NULL, NULL);
  TEST_ASSERT(st != NULL);

  A = DDSparseSUNMatFromStructure(st, PENDULUM_JAC_NNZ, CSC_MAT, ctx);
  TEST_ASSERT(A != NULL);
  TEST_ASSERT(SM_NNZ_S(A) == PENDULUM_JAC_NNZ + 8);

  B = DDSparseSUNMatFromStructure(st, PENDULUM_JAC_NNZ, CSR_MAT, ctx);
  TEST_ASSERT(B != NULL);
  TEST_ASSERT(SM_NNZ_S(B) == PENDULUM_JAC_NNZ + 4);

  STDestroy(st);
  SUNMatDestroy(A);
  SUNMatDestroy(B);

  /* ------------------------------------------------------------------------
   * Linear system
   * ------------------------------------------------------------------------ */

  st = STCreate(LINSYS_N, LINSYS_C, LINSYS_D, NULL, NULL);
  TEST_ASSERT(st != NULL);

  A = DDSparseSUNMatFromStructure(st, LINSYS_JAC_NNZ, CSC_MAT, ctx);
  TEST_ASSERT(A != NULL);
  TEST_ASSERT(SM_NNZ_S(A) == LINSYS_JAC_NNZ + 14);

  B = DDSparseSUNMatFromStructure(st, LINSYS_JAC_NNZ, CSR_MAT, ctx);
  TEST_ASSERT(B != NULL);
  TEST_ASSERT(SM_NNZ_S(B) == LINSYS_JAC_NNZ + 4);

  STDestroy(st);
  SUNMatDestroy(A);
  SUNMatDestroy(B);

  /* ------------------------------------------------------------------------
   * Non-linear system
   * ------------------------------------------------------------------------ */

  st = STCreate(NONLINSYS_N, NONLINSYS_C, NONLINSYS_D, NULL, NULL);
  TEST_ASSERT(st != NULL);

  A = DDSparseSUNMatFromStructure(st, NONLINSYS_JAC_NNZ, CSC_MAT, ctx);
  TEST_ASSERT(A != NULL);
  TEST_ASSERT(SM_NNZ_S(A) == NONLINSYS_JAC_NNZ + 22);

  B = DDSparseSUNMatFromStructure(st, NONLINSYS_JAC_NNZ, CSR_MAT, ctx);
  TEST_ASSERT(B != NULL);
  TEST_ASSERT(SM_NNZ_S(B) == NONLINSYS_JAC_NNZ + 10);

  STDestroy(st);
  SUNMatDestroy(A);
  SUNMatDestroy(B);

  return EXIT_SUCCESS;
}
