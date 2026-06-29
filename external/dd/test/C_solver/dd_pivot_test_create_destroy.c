#include <stdlib.h>
#include <sundials/sundials_core.h>
#include <sunmatrix/sunmatrix_dense.h>

#include "matrix.h"
#include "models.h"
#include "pivot.h"
#include "static_info.h"
#include "test.h"
#include "test_structure.h"

int main(void)
{
  SUNContext_Create(SUN_COMM_NULL, &CTX);

  DDStaticInfo si = DDstaticInfoCreate(CTX,
                          PENDULUM_N,
                          PENDULUM_C,
                          PENDULUM_D,
                          PENDULUM_VAR_IDX_MAP,
                          NULL,
                          NULL);

  TEST_ASSERT(si != NULL);
  SUNMatrix J_0 = SUNDenseMatrix(si->N, si->N, CTX);
  TEST_ASSERT(J_0 != NULL);
  DDMatrix dd_J_0 = DDMatWrapDense(J_0);
  TEST_ASSERT(dd_J_0 != NULL);

  PivMem pm = PIVCreate(CTX, si, dd_J_0, PendulumJacf0);
  TEST_ASSERT(pm != NULL);
  PIVDestroy(&pm);
  pm = NULL;
  PIVDestroy(&pm);

  DDstaticInfoDestroy(si);
  SUNMatDestroy(J_0);
  DDMatDestroy(dd_J_0);

  return EXIT_SUCCESS;
}
