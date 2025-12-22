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

  DAEStruct st =
    STCreate(PENDULUM_N, PENDULUM_C, PENDULUM_D, PENDULUM_VAR_IDX_MAP, NULL, NULL);

  TEST_ASSERT(st != NULL);
  DDMatrix jac = DDMatWrapDense(jac_pendulum_create(0));
  TEST_ASSERT(jac != NULL);

  PivMem pm = PIVCreate(CTX, st, jac);
  TEST_ASSERT(pm != NULL);
  PIVDestroy(pm);
  pm = NULL;
  PIVDestroy(pm);

  STDestroy(st);
  SUNMatDestroy(DDMatGetSUNMat(jac));
  DDMatDestroy(jac);

  return EXIT_SUCCESS;
}
