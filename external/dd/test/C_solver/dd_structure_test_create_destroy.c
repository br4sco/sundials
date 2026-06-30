#include <stdlib.h>
#include <sundials/sundials_core.h>
#include <sunmatrix/sunmatrix_dense.h>

#include "models.h"
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
  DDStaticInfoDestroy(si);
  si = NULL;
  DDStaticInfoDestroy(si);

  return EXIT_SUCCESS;
}
