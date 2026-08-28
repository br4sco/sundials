#include <stdlib.h>
#include <sundials/sundials_core.h>
#include <sunmatrix/sunmatrix_dense.h>

#include "dd_staged_pivot_impl.h"
#include "dd_staged_pivot_matrix.h"
#include "models.h"
#include "static_info.h"
#include "test.h"
#include "test_structure.h"

int main(void)
{
  SUNContext_Create(SUN_COMM_NULL, &CTX);

  DDStaticInfo si = DDStaticInfoCreate(PENDULUM_N,
                                       PENDULUM_C,
                                       PENDULUM_D,
                                       PENDULUM_VAR_IDX_MAP,
                                       NULL,
                                       NULL);

  TEST_ASSERT(si != NULL);
  SUNMatrix J0 = SUNDenseMatrix(si->N, si->N, CTX);
  TEST_ASSERT(J0 != NULL);
  DDStagedPivotMatrix pJ0 = DDStagedPivotMatWrapDense(J0);
  TEST_ASSERT(pJ0 != NULL);

  DDStagedPivot pm = DDSPCreateStaged(CTX, si, pJ0, PendulumJacf0);
  TEST_ASSERT(pm != NULL);
  DDSPDestroyStaged(&pm);
  pm = NULL;
  DDSPDestroyStaged(&pm);

  DDStaticInfoDestroy(si);
  SUNMatDestroy(J0);
  DDStagedPivotMatDestroy(pJ0);

  return EXIT_SUCCESS;
}
