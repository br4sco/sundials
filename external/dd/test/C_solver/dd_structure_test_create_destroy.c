#include <stdlib.h>
#include <sundials/sundials_core.h>
#include <sunmatrix/sunmatrix_dense.h>

#include "models.h"
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
  STDestroy(st);
  st = NULL;
  STDestroy(st);

  return EXIT_SUCCESS;
}
