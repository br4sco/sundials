#include <stdlib.h>
#include <sundials/sundials_core.h>
#include <sunmatrix/sunmatrix_dense.h>

#include "dynamic_info.h"
#include "matrix.h"
#include "models.h"
#include "pivot.h"
#include "structure.h"
#include "sundials/sundials_types.h"
#include "test.h"
#include "test_structure.h"

int main(void)
{
  SUNContext_Create(SUN_COMM_NULL, &CTX);

  DAEStruct st = STCreate(CTX,
                          PENDULUM_N,
                          PENDULUM_C,
                          PENDULUM_D,
                          PENDULUM_VAR_IDX_MAP,
                          NULL,
                          NULL);

  TEST_ASSERT(st != NULL);
  DDMatrix jac = DDMatWrapDense(jac_pendulum_create(0, -1));
  TEST_ASSERT(jac != NULL);
  PivMem pm = PIVCreate(CTX, st, jac);
  TEST_ASSERT(pm != NULL);

  TEST_ASSERT(PIVPivot(st, jac, 0, pm) == SUN_SUCCESS);
  TEST_ASSERT(PIVComputeDDSpec(st, pm) == SUN_SUCCESS);

  size_t k              = 0;
  sunbooleantype* known = pm->known_k[k];
  TEST_ASSERT(known[0] == SUNFALSE);
  TEST_ASSERT(known[1]);
  TEST_ASSERT(known[2] == SUNFALSE);

  k     = 1;
  known = pm->known_k[k];
  TEST_ASSERT(known[0] == SUNFALSE);
  TEST_ASSERT(known[1]);
  TEST_ASSERT(known[2] == SUNFALSE);

  k     = 2;
  known = pm->known_k[k];
  TEST_ASSERT(known[0]);
  TEST_ASSERT(known[1]);
  TEST_ASSERT(known[2]);

  uint8_t* spec = pm->spec;
  TEST_ASSERT(spec[0] == 2);
  TEST_ASSERT(spec[1] == 0);
  TEST_ASSERT(spec[2] == 0);

  DDstateMem state = DDstateCreate(st);
  TEST_ASSERT(state != NULL)

  TEST_ASSERT(DDstateUpdate(st, pm->spec, state) == SUN_SUCCESS);

  Pair_sunindextype* aliases = state->diff_var_aliases;
  TEST_ASSERT(aliases[0].fst == 0);
  TEST_ASSERT(aliases[0].snd == 1);
  TEST_ASSERT(aliases[1].fst == 1);
  TEST_ASSERT(aliases[1].snd == 2);

  sunindextype* yy = state->yy_diff_alias_row;
  sunindextype* yp = state->yp_diff_alias_row;
  TEST_ASSERT(yy[0] == 5);
  TEST_ASSERT(yp[0] < 0);
  TEST_ASSERT(yp[1] == 5);
  TEST_ASSERT(yy[1] == 6);
  TEST_ASSERT(yy[2] < 0);
  TEST_ASSERT(yp[2] == 6);
  TEST_ASSERT(yy[3] < 0);
  TEST_ASSERT(yp[3] < 0);
  TEST_ASSERT(yy[4] < 0);
  TEST_ASSERT(yp[4] < 0);
  TEST_ASSERT(yy[5] < 0);
  TEST_ASSERT(yp[5] < 0);
  TEST_ASSERT(yy[6] < 0);
  TEST_ASSERT(yp[6] < 0);

  PIVDestroy(&pm);
  STDestroy(st);
  DDMatDestroy(jac);
  DDstateDestroy(&state);

  return EXIT_SUCCESS;
}
