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

  Structure* st = STCreate(NONLINSYS_N, NONLINSYS_C, NONLINSYS_D, NULL, NULL);
  TEST_ASSERT(st != NULL);
  DDMatrix* jac = DDMatWrapDense(jac_nonlinsys_create(1, 1, 1, 1, 1, 1, 1, 1));
  TEST_ASSERT(jac != NULL);
  PivMem* pm = PMCreate(CTX, st, jac);
  TEST_ASSERT(pm != NULL);

  TEST_ASSERT(PPivot(st, jac, 0, pm) == SUN_SUCCESS);
  TEST_ASSERT(PPComputeDDSpec(st, pm) == SUN_SUCCESS);

  size_t k = 0;
  TEST_ASSERT(st->st_Mk[k] == 0) /* emtpy stage */

  k                     = 1;
  sunbooleantype* known = pm->pm_known[k];
  sunindextype* eqns    = st->st_eqns[k];
  sunindextype* vars    = pm->pm_vars[k];
  TEST_ASSERT(st->st_Mk[k] == 2)
  TEST_ASSERT(eqns[0] == 2);
  TEST_ASSERT(eqns[1] == 3);
  TEST_ASSERT(known[0] == SUNFALSE);
  TEST_ASSERT(known[1] == SUNFALSE);
  TEST_ASSERT(known[2]);
  TEST_ASSERT(known[3]);
  TEST_ASSERT(known[4] == SUNFALSE);
  TEST_ASSERT(vars[0] == 3);
  TEST_ASSERT(vars[1] == 2);

  k     = 2;
  known = pm->pm_known[k];
  eqns  = st->st_eqns[k];
  vars  = pm->pm_vars[k];
  TEST_ASSERT(st->st_Mk[k] == 4)
  TEST_ASSERT(eqns[0] == 0);
  TEST_ASSERT(eqns[1] == 2);
  TEST_ASSERT(eqns[2] == 3);
  TEST_ASSERT(eqns[3] == 4);
  TEST_ASSERT(known[0] == SUNFALSE);
  TEST_ASSERT(known[1]);
  TEST_ASSERT(known[2]);
  TEST_ASSERT(known[3]);
  TEST_ASSERT(known[4]);
  TEST_ASSERT(vars[0] == 4);
  TEST_ASSERT(vars[1] == 3);
  TEST_ASSERT(vars[2] == 2);
  TEST_ASSERT(vars[3] == 1);

  k     = 3;
  known = pm->pm_known[k];
  eqns  = st->st_eqns[k];
  vars  = pm->pm_vars[k];
  TEST_ASSERT(st->st_Mk[k] == 5)
  TEST_ASSERT(eqns[0] == 0);
  TEST_ASSERT(eqns[1] == 1);
  TEST_ASSERT(eqns[2] == 2);
  TEST_ASSERT(eqns[3] == 3);
  TEST_ASSERT(eqns[4] == 4);
  TEST_ASSERT(known[0]);
  TEST_ASSERT(known[1]);
  TEST_ASSERT(known[2]);
  TEST_ASSERT(known[3]);
  TEST_ASSERT(known[4]);
  TEST_ASSERT(vars[0] == 0);
  TEST_ASSERT(vars[1] == 1);
  TEST_ASSERT(vars[2] == 2);
  TEST_ASSERT(vars[3] == 3);
  TEST_ASSERT(vars[4] == 4);

  uint8_t* spec = pm->pm_spec;
  TEST_ASSERT(spec[0] == 3);
  TEST_ASSERT(spec[1] == 1);
  TEST_ASSERT(spec[2] == 0);
  TEST_ASSERT(spec[3] == 0);
  TEST_ASSERT(spec[4] == 1);
  TEST_ASSERT(pm->pm_NNZ_spec == 3);
  sunindextype* NZ_spec = pm->pm_NZ_spec;
  TEST_ASSERT(NZ_spec[0] == 0);
  TEST_ASSERT(NZ_spec[1] == 1);
  TEST_ASSERT(NZ_spec[2] == 4);

  PMDestroy(pm);
  STDestroy(st);
  SUNMatDestroy(DDMatGetSUNMat(jac));
  DDMatDestroy(jac);

  return EXIT_SUCCESS;
}
