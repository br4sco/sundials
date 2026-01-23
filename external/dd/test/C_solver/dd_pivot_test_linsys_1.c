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

  DAEStruct st =
    STCreate(LINSYS_N, LINSYS_C, LINSYS_D, LINSYS_VAR_IDX_MAP, NULL, NULL);

  TEST_ASSERT(st != NULL);
  DDMatrix jac = DDMatWrapDense(jac_linsys_create());
  TEST_ASSERT(jac != NULL);
  PivMem pm = PIVCreate(CTX, st, jac);
  TEST_ASSERT(pm != NULL);

  TEST_ASSERT(PIVPivot(st, jac, 0, pm) == SUN_SUCCESS);
  TEST_ASSERT(PIVComputeDDSpec(st, pm) == SUN_SUCCESS);

  size_t k              = 0;
  sunbooleantype* known = pm->known_k[k];
  TEST_ASSERT(st->M_k[k] == 2)
  TEST_ASSERT(known[0]);
  TEST_ASSERT(known[1] == SUNFALSE);
  TEST_ASSERT(known[2]);
  TEST_ASSERT(known[3] == SUNFALSE);

  k     = 1;
  known = pm->known_k[k];
  TEST_ASSERT(st->M_k[k] == 3)
  TEST_ASSERT(known[0]);
  TEST_ASSERT(known[1] == SUNFALSE);
  TEST_ASSERT(known[2]);
  TEST_ASSERT(known[3]);

  k     = 2;
  known = pm->known_k[k];
  TEST_ASSERT(st->M_k[k] == 4)
  TEST_ASSERT(known[0]);
  TEST_ASSERT(known[1]);
  TEST_ASSERT(known[2]);
  TEST_ASSERT(known[3]);

  uint8_t* spec = pm->spec;
  TEST_ASSERT(spec[0] == 0);
  TEST_ASSERT(spec[1] == 2);
  TEST_ASSERT(spec[2] == 0);
  TEST_ASSERT(spec[3] == 0);
  TEST_ASSERT(pm->NNZ_spec == 1);
  TEST_ASSERT(pm->NZ_spec[0] == 1);

  TEST_ASSERT(pm->N_diff_vars == 2);
  TEST_ASSERT(pm->diff_var_aliases[0].fst == 3);
  TEST_ASSERT(pm->diff_var_aliases[0].snd == 4);
  TEST_ASSERT(pm->diff_var_aliases[1].fst == 4);
  TEST_ASSERT(pm->diff_var_aliases[1].snd == 5);

  TEST_ASSERT(pm->yy_diff_alias_row[0] < 0);
  TEST_ASSERT(pm->yy_diff_alias_row[1] < 0);
  TEST_ASSERT(pm->yy_diff_alias_row[2] < 0);
  TEST_ASSERT(pm->yy_diff_alias_row[3] == 9);
  TEST_ASSERT(pm->yy_diff_alias_row[4] == 10);
  TEST_ASSERT(pm->yy_diff_alias_row[5] < 0);

  TEST_ASSERT(pm->yp_diff_alias_row[0] < 0);
  TEST_ASSERT(pm->yp_diff_alias_row[1] < 0);
  TEST_ASSERT(pm->yp_diff_alias_row[2] < 0);
  TEST_ASSERT(pm->yp_diff_alias_row[3] < 0);
  TEST_ASSERT(pm->yp_diff_alias_row[4] == 9);
  TEST_ASSERT(pm->yp_diff_alias_row[5] == 10);

  PIVDestroy(pm);
  STDestroy(st);
  SUNMatDestroy(DDMatGetSUNMat(jac));
  DDMatDestroy(jac);

  return EXIT_SUCCESS;
}
