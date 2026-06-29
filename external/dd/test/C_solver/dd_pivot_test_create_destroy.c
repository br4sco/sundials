#include <stdlib.h>
#include <sundials/sundials_core.h>
#include <sunmatrix/sunmatrix_dense.h>

#include "matrix.h"
#include "models.h"
#include "pivot.h"
#include "structure.h"
#include "test.h"
#include "test_structure.h"

int main(void)
{
  SUNContext_Create(SUN_COMM_NULL, &CTX);

  DAEStruct st = STCreate(CTX,
                          PENDULUM_N,
                          PENDULUM_C,
                          PENDULUM_D,
                          PENDULUM_VAR_IDX_MAP,
                          NULL,
                          NULL);

  TEST_ASSERT(st != NULL);
  SUNMatrix J_0 = SUNDenseMatrix(st->N, st->N, CTX);
  TEST_ASSERT(J_0 != NULL);
  DDMatrix dd_J_0 = DDMatWrapDense(J_0);
  TEST_ASSERT(dd_J_0 != NULL);

  PivMem pm = PIVCreate(CTX, st, dd_J_0, PendulumJacf0);
  TEST_ASSERT(pm != NULL);
  PIVDestroy(&pm);
  pm = NULL;
  PIVDestroy(&pm);

  STDestroy(st);
  SUNMatDestroy(J_0);
  DDMatDestroy(dd_J_0);

  return EXIT_SUCCESS;
}
