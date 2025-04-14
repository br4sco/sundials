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
  Structure* st = STCreate(PENDULUM_N, PENDULUM_C, PENDULUM_D, NULL, NULL);
  TEST_ASSERT(st != NULL);

  TEST_ASSERT(st->st_DAE_N == 3);
  TEST_ASSERT(st->st_DAE_N1 == 5);

  TEST_ASSERT(st->st_eqnofs[0] == 0); /* f₁(x'', x, λ) */
  TEST_ASSERT(st->st_eqnofs[1] == 0); /* f₂(y'', y, λ) */
  TEST_ASSERT(st->st_eqnofs[2] == 2); /* f₃(x, y) */

  TEST_ASSERT(st->st_varofs[0] == 2); /* x'' */
  TEST_ASSERT(st->st_varofs[1] == 2); /* y'' */
  TEST_ASSERT(st->st_varofs[2] == 0); /* λ */

  TEST_ASSERT(st->st_K == 3);
  TEST_ASSERT(ST_STAGE_FROM_INDEX(st, 0) == -2);
  TEST_ASSERT(ST_STAGE_FROM_INDEX(st, 1) == -1);
  TEST_ASSERT(ST_STAGE_FROM_INDEX(st, 2) == 0);

  TEST_ASSERT(st->st_Mk[0] == 1); /* f₃ */
  TEST_ASSERT(st->st_Mk[1] == 1); /* f₃' */
  TEST_ASSERT(st->st_Mk[2] == 3); /* f₁ f₂ f₃'' */

  TEST_ASSERT(st->st_Nk[0] == 2); /* x y */
  TEST_ASSERT(st->st_Nk[1] == 2); /* x' y' */
  TEST_ASSERT(st->st_Nk[2] == 3); /* x'' y'' λ */

  k                  = 0; /* stage k = -2 */
  sunindextype* eqns = st->st_eqns[k];
  sunindextype* vars = st->st_vars[k];
  i                  = 2; /* f₃ */
  TEST_ASSERT(eqns[0] == i);
  TEST_ASSERT(ST_EQN_ORDER(st, k, i) == 0);
  j = 0; /* x */
  TEST_ASSERT(vars[0] == j);
  TEST_ASSERT(ST_VAR_ORDER(st, k, j) == 0);
  j = 1; /* y */
  TEST_ASSERT(vars[1] == j);
  TEST_ASSERT(ST_VAR_ORDER(st, k, j) == 0);

  k    = 1; /* stage k = -1 */
  eqns = st->st_eqns[k];
  vars = st->st_vars[k];
  i    = 2; /* f₃' */
  TEST_ASSERT(eqns[0] == i);
  TEST_ASSERT(ST_EQN_ORDER(st, 1, i) == 1);
  j = 0; /* x' */
  TEST_ASSERT(vars[0] == j);
  TEST_ASSERT(ST_VAR_ORDER(st, k, j) == 1);
  j = 1; /* y' */
  TEST_ASSERT(vars[1] == j);
  TEST_ASSERT(ST_VAR_ORDER(st, k, j) == 1);

  k    = 2; /* stage k = 0 */
  eqns = st->st_eqns[k];
  vars = st->st_vars[k];
  i    = 0; /* f₁ */
  TEST_ASSERT(eqns[0] == i);
  TEST_ASSERT(ST_EQN_ORDER(st, k, i) == 0);
  i = 1; /* f₂ */
  TEST_ASSERT(eqns[1] == i);
  TEST_ASSERT(ST_EQN_ORDER(st, k, i) == 0);
  i = 2; /* f₃'' */
  TEST_ASSERT(eqns[2] == i);
  TEST_ASSERT(ST_EQN_ORDER(st, k, i) == 2);
  j = 0; /* x'' */
  TEST_ASSERT(vars[0] == j);
  TEST_ASSERT(ST_VAR_ORDER(st, k, j) == 2);
  j = 1; /* y'' */
  TEST_ASSERT(vars[1] == j);
  TEST_ASSERT(ST_VAR_ORDER(st, k, j) == 2);
  j = 2; /* λ */
  TEST_ASSERT(vars[2] == j);
  TEST_ASSERT(ST_VAR_ORDER(st, k, j) == 0);

  STDestroy(st);

  return EXIT_SUCCESS;
}
