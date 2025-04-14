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

  Structure* st = STCreate(PENDULUM_N, PENDULUM_C, PENDULUM_D, NULL, NULL);
  TEST_ASSERT(st != NULL);
  ExtSUNMatrix* jac = ExtSUNMatWrapDense(jac_pendulum_create(0));
  TEST_ASSERT(jac != NULL);

  PivMem* ps = PMCreate(st, jac);
  TEST_ASSERT(ps != NULL);
  PMDestroy(ps);
  ps = NULL;
  PMDestroy(ps);

  STDestroy(st);
  SUNMatDestroy(ExtSUNMatGetMat(jac));
  ExtSUNMatDestroy(jac);

  return EXIT_SUCCESS;
}
