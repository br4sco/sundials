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

  DDStaticInfo si = DDStaticInfoCreate(NONLINSYS_N,
                                       NONLINSYS_C,
                                       NONLINSYS_D,
                                       NONLINSYS_VAR_IDX_MAP,
                                       NULL,
                                       NULL);

  TEST_ASSERT(si != NULL);
  SUNMatrix J0 = SUNDenseMatrix(NONLINSYS_N, NONLINSYS_N, CTX);
  TEST_ASSERT(J0 != NULL);
  PIVMatrix pJ0 = PIVMatWrapDense(J0);
  TEST_ASSERT(pJ0 != NULL);
  PIVMem pm = PIVCreate(CTX, si, pJ0, NonlinsysJacf0);
  TEST_ASSERT(pm != NULL);

  N_Vector Y = N_VNew_Serial(si->N_all_orders, CTX);
  TEST_ASSERT(Y != NULL);
  N_VConst(ONE, Y);

  sunbooleantype spec_changed;
  TEST_ASSERT(PIVPivot(pm, ZERO, ZERO, Y, &spec_changed) == SUN_SUCCESS);

  size_t k = 0;
  TEST_ASSERT(si->M_k[k] == 0) /* emtpy stage */

  k                     = 1;
  sunbooleantype* known = pm->known_k[k];
  sunindextype* eqns    = si->eqns_k[k];
  TEST_ASSERT(si->M_k[k] == 2)
  TEST_ASSERT(eqns[0] == 2);
  TEST_ASSERT(eqns[1] == 3);
  TEST_ASSERT(known[0] == SUNFALSE);
  TEST_ASSERT(known[1] == SUNFALSE);
  TEST_ASSERT(known[2]);
  TEST_ASSERT(known[3]);
  TEST_ASSERT(known[4] == SUNFALSE);

  k     = 2;
  known = pm->known_k[k];
  eqns  = si->eqns_k[k];
  TEST_ASSERT(si->M_k[k] == 4)
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
  eqns  = si->eqns_k[k];
  TEST_ASSERT(si->M_k[k] == 5)
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
  DDStaticInfoDestroy(si);
  N_VDestroy(Y);
  SUNMatDestroy(J0);
  PIVMatDestroy(pJ0);

  return EXIT_SUCCESS;
}
