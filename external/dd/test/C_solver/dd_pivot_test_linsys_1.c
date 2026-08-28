#include <stdlib.h>
#include <sundials/sundials_core.h>
#include <sunmatrix/sunmatrix_dense.h>

#include "dd_staged_pivot_impl.h"
#include "dd_staged_pivot_matrix.h"
#include "dynamic_info.h"
#include "models.h"
#include "static_info.h"
#include "test.h"
#include "test_structure.h"

int main(void)
{
  SUNContext_Create(SUN_COMM_NULL, &CTX);

  DDStaticInfo si =
    DDStaticInfoCreate(LINSYS_N, LINSYS_C, LINSYS_D, LINSYS_VAR_IDX_MAP, NULL, NULL);

  TEST_ASSERT(si != NULL);
  SUNMatrix J0 = SUNDenseMatrix(LINSYS_N, LINSYS_N, CTX);
  TEST_ASSERT(J0 != NULL);
  DDStagedPivotMatrix pJ0 = DDStagedPivotMatWrapDense(J0);
  TEST_ASSERT(pJ0 != NULL);
  DDStagedPivot pm = DDSPCreateStaged(CTX, si, pJ0, LinsysJacf0);
  TEST_ASSERT(pm != NULL);

  N_Vector Y = N_VNew_Serial(si->N_all_orders, CTX);
  TEST_ASSERT(Y != NULL);
  N_VConst(ZERO, Y);

  sunbooleantype spec_changed;
  TEST_ASSERT(DDSPPivotStaged(pm, ZERO, ZERO, Y, NULL, &spec_changed) ==
              SUN_SUCCESS);

  size_t k              = 0;
  sunbooleantype* known = pm->known_k[k];
  TEST_ASSERT(si->M_k[k] == 2)
  TEST_ASSERT(known[0]);
  TEST_ASSERT(known[1] == SUNFALSE);
  TEST_ASSERT(known[2]);
  TEST_ASSERT(known[3] == SUNFALSE);

  k     = 1;
  known = pm->known_k[k];
  TEST_ASSERT(si->M_k[k] == 3)
  TEST_ASSERT(known[0]);
  TEST_ASSERT(known[1] == SUNFALSE);
  TEST_ASSERT(known[2]);
  TEST_ASSERT(known[3]);

  k     = 2;
  known = pm->known_k[k];
  TEST_ASSERT(si->M_k[k] == 4)
  TEST_ASSERT(known[0]);
  TEST_ASSERT(known[1]);
  TEST_ASSERT(known[2]);
  TEST_ASSERT(known[3]);

  uint8_t* spec = pm->spec;
  TEST_ASSERT(spec[0] == 0);
  TEST_ASSERT(spec[1] == 2);
  TEST_ASSERT(spec[2] == 0);
  TEST_ASSERT(spec[3] == 0);

  DDDAEState state = DDDAEStateCreate(CTX, si);
  TEST_ASSERT(state != NULL)

  TEST_ASSERT(DDDAEStateUpdate(state, pm->spec) == SUN_SUCCESS);

  Pair_sunindextype* aliases = state->diff_var_aliases;
  TEST_ASSERT(aliases[0].fst == 3);
  TEST_ASSERT(aliases[0].snd == 4);
  TEST_ASSERT(aliases[1].fst == 4);
  TEST_ASSERT(aliases[1].snd == 5);

  sunindextype* yy = state->yy_diff_alias_row;
  sunindextype* yp = state->yp_diff_alias_row;

  TEST_ASSERT(yy[0] < 0);
  TEST_ASSERT(yy[1] < 0);
  TEST_ASSERT(yy[2] < 0);
  TEST_ASSERT(yy[3] == 9);
  TEST_ASSERT(yy[4] == 10);
  TEST_ASSERT(yy[5] < 0);

  TEST_ASSERT(yp[0] < 0);
  TEST_ASSERT(yp[1] < 0);
  TEST_ASSERT(yp[2] < 0);
  TEST_ASSERT(yp[3] < 0);
  TEST_ASSERT(yp[4] == 9);
  TEST_ASSERT(yp[5] == 10);

  DDSPDestroyStaged(&pm);
  DDStaticInfoDestroy(si);
  N_VDestroy(Y);
  SUNMatDestroy(J0);
  DDStagedPivotMatDestroy(pJ0);
  DDDAEStateDestroy(&state);

  return EXIT_SUCCESS;
}
