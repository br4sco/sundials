#include <stdlib.h>
#include <sundials/sundials_core.h>
#include <sunmatrix/sunmatrix_dense.h>

#include "matrix.h"
#include "models.h"
#include "structure.h"
#include "test.h"
#include "test_structure.h"

int main(void)
{
  SUNContext_Create(SUN_COMM_NULL, &CTX);

  Structure* st = STCreate(PENDULUM_N, PENDULUM_C, PENDULUM_D, NULL, NULL);
  TEST_ASSERT(st != NULL);
  STDestroy(st);
  st = NULL;
  STDestroy(st);

  return EXIT_SUCCESS;
}
