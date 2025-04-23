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
  DDMatrix* jac = DDMatWrapDense(jac_pendulum_create(M_PI / 2));
  TEST_ASSERT(jac != NULL);
  PivMem* pm = PMCreate(CTX, st, jac);
  TEST_ASSERT(pm != NULL);

  TEST_ASSERT(PPivot(st, jac, 0, pm) == SUN_SUCCESS);
  TEST_ASSERT(PPComputeDDSpec(st, pm) == SUN_SUCCESS);

  size_t k              = 0;
  sunbooleantype* known = pm->pm_known[k];
  sunindextype* vars    = pm->pm_vars[k];
  TEST_ASSERT(known[0] == SUNFALSE);
  TEST_ASSERT(known[1]);
  TEST_ASSERT(known[2] == SUNFALSE);
  TEST_ASSERT(vars[0] == 1);
  TEST_ASSERT(vars[1] == 0);

  k     = 1;
  known = pm->pm_known[k];
  vars  = pm->pm_vars[k];
  TEST_ASSERT(known[0] == SUNFALSE);
  TEST_ASSERT(known[1]);
  TEST_ASSERT(known[2] == SUNFALSE);
  TEST_ASSERT(vars[0] == 1);
  TEST_ASSERT(vars[1] == 0);

  k     = 2;
  known = pm->pm_known[k];
  vars  = pm->pm_vars[k];
  TEST_ASSERT(known[0]);
  TEST_ASSERT(known[1]);
  TEST_ASSERT(known[2]);
  TEST_ASSERT(vars[0] == 0);
  TEST_ASSERT(vars[1] == 1);
  TEST_ASSERT(vars[2] == 2);

  uint8_t* spec = pm->pm_spec;
  TEST_ASSERT(spec[0] == 2);
  TEST_ASSERT(spec[1] == 0);
  TEST_ASSERT(spec[2] == 0);
  TEST_ASSERT(pm->pm_NNZ_spec == 1);
  TEST_ASSERT(pm->pm_NZ_spec[0] == 0);

  PMDestroy(pm);
  STDestroy(st);
  DDMatDestroy(jac);

  return EXIT_SUCCESS;
}
