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

  DAEStruct st = STCreate(CTX,
                          NONLINSYS_N,
                          NONLINSYS_C,
                          NONLINSYS_D,
                          NONLINSYS_VAR_IDX_MAP,
                          NULL,
                          NULL);

  TEST_ASSERT(st != NULL);
  DDMatrix jac = DDMatWrapDense(jac_nonlinsys_create(1, 1, 1, 1, 1, 1, 1, 1));
  TEST_ASSERT(jac != NULL);
  PivMem pm = PIVCreate(CTX, st, jac);
  TEST_ASSERT(pm != NULL);

  TEST_ASSERT(PIVPivot(st, jac, 0, pm) == SUN_SUCCESS);
  TEST_ASSERT(PIVComputeDDSpec(st, pm) == SUN_SUCCESS);

  size_t k = 0;
  TEST_ASSERT(st->M_k[k] == 0) /* emtpy stage */

  k                     = 1;
  sunbooleantype* known = pm->known_k[k];
  sunindextype* eqns    = st->eqns_k[k];
  TEST_ASSERT(st->M_k[k] == 2)
  TEST_ASSERT(eqns[0] == 2);
  TEST_ASSERT(eqns[1] == 3);
  TEST_ASSERT(known[0] == SUNFALSE);
  TEST_ASSERT(known[1] == SUNFALSE);
  TEST_ASSERT(known[2]);
  TEST_ASSERT(known[3]);
  TEST_ASSERT(known[4] == SUNFALSE);

  k     = 2;
  known = pm->known_k[k];
  eqns  = st->eqns_k[k];
  TEST_ASSERT(st->M_k[k] == 4)
  TEST_ASSERT(eqns[0] == 0);
  TEST_ASSERT(eqns[1] == 2);
  TEST_ASSERT(eqns[2] == 3);
  TEST_ASSERT(eqns[3] == 4);
  TEST_ASSERT(known[0] == SUNFALSE);
  TEST_ASSERT(known[1]);
  TEST_ASSERT(known[2]);
  TEST_ASSERT(known[3]);
  TEST_ASSERT(known[4]);

  k     = 3;
  known = pm->known_k[k];
  eqns  = st->eqns_k[k];
  TEST_ASSERT(st->M_k[k] == 5)
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

  uint8_t* spec = pm->spec;
  TEST_ASSERT(spec[0] == 3);
  TEST_ASSERT(spec[1] == 1);
  TEST_ASSERT(spec[2] == 0);
  TEST_ASSERT(spec[3] == 0);
  TEST_ASSERT(spec[4] == 1);

  PIVDestroy(&pm);
  STDestroy(st);
  SUNMatDestroy(DDMatGetSUNMat(jac));
  DDMatDestroy(jac);

  return EXIT_SUCCESS;
}
