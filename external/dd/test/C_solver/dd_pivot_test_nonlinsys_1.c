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
                                       NONLINSYS_N,
                                       NONLINSYS_C,
                                       NONLINSYS_D,
                                       NONLINSYS_VAR_IDX_MAP,
                                       NULL,
                                       NULL);

  TEST_ASSERT(si != NULL);
  SUNMatrix J_0 = SUNDenseMatrix(NONLINSYS_N, NONLINSYS_N, CTX);
  TEST_ASSERT(J_0 != NULL);
  DDMatrix jac = DDMatWrapDense(J_0);
  TEST_ASSERT(jac != NULL);
  PivMem pm = PIVCreate(CTX, si, jac, NonlinsysJacf0);
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
  DDstaticInfoDestroy(si);
  N_VDestroy(Y);
  SUNMatDestroy(J_0);
  DDMatDestroy(jac);

  return EXIT_SUCCESS;
}
