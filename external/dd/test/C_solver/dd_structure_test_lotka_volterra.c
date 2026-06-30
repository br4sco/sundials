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
  DDStaticInfo si = DDStaticInfoCreate(CTX,
                                       LOTKA_VOLTERRA_N,
                                       LOTKA_VOLTERRA_C,
                                       LOTKA_VOLTERRA_D,
                                       LOTKA_VOLTERRA_VAR_IDX_MAP,
                                       NULL,
                                       NULL);

  TEST_ASSERT(si != NULL);

  TEST_ASSERT(si->N == 2);
  TEST_ASSERT(si->N_backwards == 2);

  TEST_ASSERT(si->eqnofs[0] == 0); /* f₁(x', x, y) */
  TEST_ASSERT(si->eqnofs[1] == 0); /* f₂(y', x, y) */

  TEST_ASSERT(si->K == 2);

  TEST_ASSERT(si->M_k[0] == 0);
  TEST_ASSERT(si->M_k[1] == 2); /* f₁ f₂ */

  TEST_ASSERT(si->N_k[0] == 2); /* x' y' */
  TEST_ASSERT(si->N_k[1] == 2); /* x' y' */

  k                  = 0; /* stage k = -1 (empty) */
  sunindextype* vars = si->vars_k[k];
  j                  = 0; /* x */
  TEST_ASSERT(vars[0] == j);
  TEST_ASSERT(DDSI_VAR_ORDER(si, k, j) == 0);
  j = 1; /* y */
  TEST_ASSERT(vars[1] == j);
  TEST_ASSERT(DDSI_VAR_ORDER(si, k, j) == 0);

  k                  = 1; /* stage k = 0 */
  sunindextype* eqns = si->eqns_k[k];
  vars               = si->vars_k[k];
  i                  = 0; /* f₁ */
  TEST_ASSERT(eqns[0] == i);
  TEST_ASSERT(DDSI_EQN_ORDER(si, k, i) == 0);
  i = 1; /* f₂ */
  TEST_ASSERT(eqns[1] == i);
  TEST_ASSERT(DDSI_EQN_ORDER(si, k, i) == 0);
  j = 0; /* x' */
  TEST_ASSERT(vars[0] == j);
  TEST_ASSERT(DDSI_VAR_ORDER(si, k, j) == 1);
  j = 1; /* y' */
  TEST_ASSERT(vars[1] == j);
  TEST_ASSERT(DDSI_VAR_ORDER(si, k, j) == 1);

  DDStaticInfoDestroy(si);

  return EXIT_SUCCESS;
}
