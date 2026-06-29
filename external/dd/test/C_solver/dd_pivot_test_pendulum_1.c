#include <math.h>
#include <stdlib.h>
#include <sundials/sundials_core.h>
#include <sunmatrix/sunmatrix_dense.h>

#include "dynamic_info.h"
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
                                       PENDULUM_N,
                                       PENDULUM_C,
                                       PENDULUM_D,
                                       PENDULUM_VAR_IDX_MAP,
                                       NULL,
                                       NULL);

  TEST_ASSERT(si != NULL);
  SUNMatrix J_0 = SUNDenseMatrix(PENDULUM_N, PENDULUM_N, CTX);
  TEST_ASSERT(J_0 != NULL);
  DDMatrix jac = DDMatWrapDense(J_0);
  TEST_ASSERT(jac != NULL);
  PivMem pm = PIVCreate(CTX, si, jac, PendulumJacf0);
  TEST_ASSERT(pm != NULL);

  PendulumData data = {.m = ONE, .param = {ONE, ZERO}};
  PIVSetUserData(pm, &data);

  N_Vector Y = N_VNew_Serial(si->N_all_orders, CTX);
  TEST_ASSERT(Y != NULL);
  N_VConst(ZERO, Y);
  P_Ith(Y, 0, 0) = cos(0);
  P_Ith(Y, 1, 0) = sin(0);

  sunbooleantype spec_changed;
  TEST_ASSERT(PIVPivot(pm, ZERO, ZERO, Y, &spec_changed) == SUN_SUCCESS);

  size_t k              = 0;
  sunbooleantype* known = pm->known_k[k];
  TEST_ASSERT(known[0]);
  TEST_ASSERT(known[1] == SUNFALSE);
  TEST_ASSERT(known[2] == SUNFALSE);

  k     = 1;
  known = pm->known_k[k];
  TEST_ASSERT(known[0]);
  TEST_ASSERT(known[1] == SUNFALSE);
  TEST_ASSERT(known[2] == SUNFALSE);

  k     = 2;
  known = pm->known_k[k];
  TEST_ASSERT(known[0]);
  TEST_ASSERT(known[1]);
  TEST_ASSERT(known[2]);

  uint8_t* spec = pm->spec;
  TEST_ASSERT(spec[0] == 0);
  TEST_ASSERT(spec[1] == 2);
  TEST_ASSERT(spec[2] == 0);

  DDDAEState state = DDDAEStateCreate(si);
  TEST_ASSERT(state != NULL)

  TEST_ASSERT(DDDAEStateUpdate(si, pm->spec, state) == SUN_SUCCESS);

  Pair_sunindextype* aliases = state->diff_var_aliases;
  TEST_ASSERT(aliases[0].fst == 3);
  TEST_ASSERT(aliases[0].snd == 4);
  TEST_ASSERT(aliases[1].fst == 4);
  TEST_ASSERT(aliases[1].snd == 5);

  sunindextype* yy = state->yy_diff_alias_row;
  sunindextype* yp = state->yp_diff_alias_row;
  TEST_ASSERT(yy[0] < 0);
  TEST_ASSERT(yp[0] < 0);
  TEST_ASSERT(yy[1] < 0);
  TEST_ASSERT(yp[1] < 0);
  TEST_ASSERT(yy[2] < 0);
  TEST_ASSERT(yp[2] < 0);
  TEST_ASSERT(yy[3] == 5);
  TEST_ASSERT(yp[3] < 0);
  TEST_ASSERT(yp[4] == 5);
  TEST_ASSERT(yy[4] == 6);
  TEST_ASSERT(yy[5] < 0);
  TEST_ASSERT(yp[5] == 6);
  TEST_ASSERT(yy[6] < 0);
  TEST_ASSERT(yp[6] < 0);

  PIVDestroy(&pm);
  DDstaticInfoDestroy(si);
  N_VDestroy(Y);
  SUNMatDestroy(J_0);
  DDMatDestroy(jac);
  DDDAEStateDestroy(&state);

  return EXIT_SUCCESS;
}
