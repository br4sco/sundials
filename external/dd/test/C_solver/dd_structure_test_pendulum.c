#include <stdlib.h>
#include <sundials/sundials_core.h>
#include <sunmatrix/sunmatrix_dense.h>

#include "models.h"
#include "static_info.h"
#include "test.h"
#include "test_structure.h"

int main(void)
{
  SUNContext_Create(SUN_COMM_NULL, &CTX);

  sunindextype i, j, k;
  DDStaticInfo si = DDStaticInfoCreate(PENDULUM_N,
                                       PENDULUM_C,
                                       PENDULUM_D,
                                       PENDULUM_VAR_IDX_MAP,
                                       NULL,
                                       NULL);

  TEST_ASSERT(si != NULL);

  TEST_ASSERT(si->N == 3);

  TEST_ASSERT(si->eqnofs[0] == 0); /* f₁(x'', x, λ) */
  TEST_ASSERT(si->eqnofs[1] == 0); /* f₂(y'', y, λ) */
  TEST_ASSERT(si->eqnofs[2] == 2); /* f₃(x, y) */

  TEST_ASSERT(si->varofs[0] == 2); /* x'' */
  TEST_ASSERT(si->varofs[1] == 2); /* y'' */
  TEST_ASSERT(si->varofs[2] == 0); /* λ */

  TEST_ASSERT(si->var_deriv_chains[0][0] == 0);
  TEST_ASSERT(si->var_deriv_chains[0][1] == 1);
  TEST_ASSERT(si->var_deriv_chains[0][2] == 2);
  TEST_ASSERT(si->var_deriv_chains[1][0] == 3);
  TEST_ASSERT(si->var_deriv_chains[1][1] == 4);
  TEST_ASSERT(si->var_deriv_chains[1][2] == 5);
  TEST_ASSERT(si->var_deriv_chains[2][0] == 6);

  TEST_ASSERT(si->K == 3);
  TEST_ASSERT(DDSI_STAGE_FROM_INDEX(si, 0) == -2);
  TEST_ASSERT(DDSI_STAGE_FROM_INDEX(si, 1) == -1);
  TEST_ASSERT(DDSI_STAGE_FROM_INDEX(si, 2) == 0);

  TEST_ASSERT(si->M_k[0] == 1); /* f₃ */
  TEST_ASSERT(si->M_k[1] == 1); /* f₃' */
  TEST_ASSERT(si->M_k[2] == 3); /* f₁ f₂ f₃'' */

  TEST_ASSERT(si->N_k[0] == 2); /* x y */
  TEST_ASSERT(si->N_k[1] == 2); /* x' y' */
  TEST_ASSERT(si->N_k[2] == 3); /* x'' y'' λ */

  k                  = 0; /* stage k = -2 */
  sunindextype* eqns = si->eqns_k[k];
  sunindextype* vars = si->vars_k[k];
  i                  = 2; /* f₃ */
  TEST_ASSERT(eqns[0] == i);
  TEST_ASSERT(DDSI_EQN_ORDER(si, k, i) == 0);
  j = 0; /* x */
  TEST_ASSERT(vars[0] == j);
  TEST_ASSERT(DDSI_VAR_ORDER(si, k, j) == 0);
  j = 1; /* y */
  TEST_ASSERT(vars[1] == j);
  TEST_ASSERT(DDSI_VAR_ORDER(si, k, j) == 0);

  k    = 1; /* stage k = -1 */
  eqns = si->eqns_k[k];
  vars = si->vars_k[k];
  i    = 2; /* f₃' */
  TEST_ASSERT(eqns[0] == i);
  TEST_ASSERT(DDSI_EQN_ORDER(si, 1, i) == 1);
  j = 0; /* x' */
  TEST_ASSERT(vars[0] == j);
  TEST_ASSERT(DDSI_VAR_ORDER(si, k, j) == 1);
  j = 1; /* y' */
  TEST_ASSERT(vars[1] == j);
  TEST_ASSERT(DDSI_VAR_ORDER(si, k, j) == 1);

  k    = 2; /* stage k = 0 */
  eqns = si->eqns_k[k];
  vars = si->vars_k[k];
  i    = 0; /* f₁ */
  TEST_ASSERT(eqns[0] == i);
  TEST_ASSERT(DDSI_EQN_ORDER(si, k, i) == 0);
  i = 1; /* f₂ */
  TEST_ASSERT(eqns[1] == i);
  TEST_ASSERT(DDSI_EQN_ORDER(si, k, i) == 0);
  i = 2; /* f₃'' */
  TEST_ASSERT(eqns[2] == i);
  TEST_ASSERT(DDSI_EQN_ORDER(si, k, i) == 2);
  j = 0; /* x'' */
  TEST_ASSERT(vars[0] == j);
  TEST_ASSERT(DDSI_VAR_ORDER(si, k, j) == 2);
  j = 1; /* y'' */
  TEST_ASSERT(vars[1] == j);
  TEST_ASSERT(DDSI_VAR_ORDER(si, k, j) == 2);
  j = 2; /* λ */
  TEST_ASSERT(vars[2] == j);
  TEST_ASSERT(DDSI_VAR_ORDER(si, k, j) == 0);

  DDStaticInfoDestroy(si);

  return EXIT_SUCCESS;
}
