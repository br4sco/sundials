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

  DDStaticInfo si = DDStaticInfoCreate(CTX,
                                       PENDULUM_N,
                                       PENDULUM_C,
                                       PENDULUM_D,
                                       PENDULUM_VAR_IDX_MAP,
                                       NULL,
                                       NULL);

  TEST_ASSERT(si != NULL);
  SUNMatrix J0 = SUNDenseMatrix(si->N, si->N, CTX);
  TEST_ASSERT(J0 != NULL);
  PIVMatrix pJ0 = PIVMatWrapDense(J0);
  TEST_ASSERT(pJ0 != NULL);

  PivMem pm = PIVCreate(CTX, si, pJ0, PendulumJacf0);
  TEST_ASSERT(pm != NULL);
  PIVDestroy(&pm);
  pm = NULL;
  PIVDestroy(&pm);

  DDStaticInfoDestroy(si);
  SUNMatDestroy(J0);
  PIVMatDestroy(pJ0);

  return EXIT_SUCCESS;
}
