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

  Struc st = STCreate(PENDULUM_N, PENDULUM_C, PENDULUM_D, NULL, NULL);
  TEST_ASSERT(st != NULL);
  DDMatrix jac = DDMatWrapDense(jac_pendulum_create(0));
  TEST_ASSERT(jac != NULL);
  PivMem pm = PMCreate(CTX, st, jac);
  TEST_ASSERT(pm != NULL);

  TEST_ASSERT(PPivot(st, jac, 0, pm) == SUN_SUCCESS);
  TEST_ASSERT(PPComputeDDSpec(st, pm) == SUN_SUCCESS);

  size_t k              = 0;
  sunbooleantype* known = pm->known_k[k];
  sunindextype* vars    = pm->vars_k[k];
  TEST_ASSERT(known[0]);
  TEST_ASSERT(known[1] == SUNFALSE);
  TEST_ASSERT(known[2] == SUNFALSE);
  TEST_ASSERT(vars[0] == 0);
  TEST_ASSERT(vars[1] == 1);

  k     = 1;
  known = pm->known_k[k];
  vars  = pm->vars_k[k];
  TEST_ASSERT(known[0]);
  TEST_ASSERT(known[1] == SUNFALSE);
  TEST_ASSERT(known[2] == SUNFALSE);
  TEST_ASSERT(vars[0] == 0);
  TEST_ASSERT(vars[1] == 1);

  k     = 2;
  known = pm->known_k[k];
  vars  = pm->vars_k[k];
  TEST_ASSERT(known[0]);
  TEST_ASSERT(known[1]);
  TEST_ASSERT(known[2]);
  TEST_ASSERT(vars[0] == 0);
  TEST_ASSERT(vars[1] == 1);
  TEST_ASSERT(vars[2] == 2);

  uint8_t* spec = pm->spec;
  TEST_ASSERT(spec[0] == 0);
  TEST_ASSERT(spec[1] == 2);
  TEST_ASSERT(spec[2] == 0);
  TEST_ASSERT(pm->NNZ_spec == 1);
  TEST_ASSERT(pm->NZ_spec[0] == 1);

  PMDestroy(pm);
  STDestroy(st);
  DDMatDestroy(jac);

  return EXIT_SUCCESS;
}
