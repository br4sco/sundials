#include <stdlib.h>
#include <sundials/sundials_core.h>
#include <sunmatrix/sunmatrix_dense.h>

#include "models.h"
#include "structure.h"
#include "test.h"
#include "test_structure.h"

int main(void)
{
  SUNContext_Create(SUN_COMM_NULL, &CTX);

  sunindextype i, j, k;
  DAEStruct st = STCreate(LOTKA_VOLTERRA_N, LOTKA_VOLTERRA_C, LOTKA_VOLTERRA_D,
                      NULL, NULL);
  TEST_ASSERT(st != NULL);

  TEST_ASSERT(st->DAE_size == 2);
  TEST_ASSERT(st->DAE_1ord_size == 2);

  TEST_ASSERT(st->eqnofs[0] == 0); /* f₁(x', x, y) */
  TEST_ASSERT(st->eqnofs[1] == 0); /* f₂(y', x, y) */

  TEST_ASSERT(st->K == 2);

  TEST_ASSERT(st->M_k[0] == 0);
  TEST_ASSERT(st->M_k[1] == 2); /* f₁ f₂ */

  TEST_ASSERT(st->N_k[0] == 2); /* x' y' */
  TEST_ASSERT(st->N_k[1] == 2); /* x' y' */

  k                  = 0; /* stage k = -1 (empty) */
  sunindextype* vars = st->vars_k[k];
  j                  = 0; /* x */
  TEST_ASSERT(vars[0] == j);
  TEST_ASSERT(ST_VAR_ORDER(st, k, j) == 0);
  j = 1; /* y */
  TEST_ASSERT(vars[1] == j);
  TEST_ASSERT(ST_VAR_ORDER(st, k, j) == 0);

  k                  = 1; /* stage k = 0 */
  sunindextype* eqns = st->eqns_k[k];
  vars               = st->vars_k[k];
  i                  = 0; /* f₁ */
  TEST_ASSERT(eqns[0] == i);
  TEST_ASSERT(ST_EQN_ORDER(st, k, i) == 0);
  i = 1; /* f₂ */
  TEST_ASSERT(eqns[1] == i);
  TEST_ASSERT(ST_EQN_ORDER(st, k, i) == 0);
  j = 0; /* x' */
  TEST_ASSERT(vars[0] == j);
  TEST_ASSERT(ST_VAR_ORDER(st, k, j) == 1);
  j = 1; /* y' */
  TEST_ASSERT(vars[1] == j);
  TEST_ASSERT(ST_VAR_ORDER(st, k, j) == 1);

  STDestroy(st);

  return EXIT_SUCCESS;
}
