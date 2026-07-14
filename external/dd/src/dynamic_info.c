#include <stdlib.h>
#include <string.h>
#include <sundials/priv/sundials_errors_impl.h>
#include "sundials/sundials_errors.h"
#include "sundials/sundials_types.h"

#include "dynamic_info.h"

void DDDAEStateDestroy(DDDAEState* state_ptr)
{
  if (state_ptr == NULL || *state_ptr == NULL) { return; }

  DDDAEState state = *state_ptr;
  free(state->diff_var_aliases);
  free(state->yy_diff_alias_row);
  free(state->yp_diff_alias_row);
  free(state);

  *state_ptr = NULL;
}

DDDAEState DDDAEStateCreate(DDStaticInfo si)
{
  DDDAEState state = calloc(1, sizeof(*state));
  if (state == NULL) { return NULL; }

  state->diff_var_aliases = malloc(si->N_diff * sizeof(*state->diff_var_aliases));
  if (state->diff_var_aliases == NULL) { goto fail; }

  state->yy_diff_alias_row =
    malloc(si->N_all_orders * sizeof(*state->yy_diff_alias_row));
  if (state->yy_diff_alias_row == NULL) { goto fail; }

  state->yp_diff_alias_row =
    malloc(si->N_all_orders * sizeof(*state->yp_diff_alias_row));
  if (state->yp_diff_alias_row == NULL) { goto fail; }

  return state;

fail:
  DDDAEStateDestroy(&state);
  return NULL;
}

DDDAEState DDDAEStateClone(DDStaticInfo si, DDDAEState state)
{
  DDDAEState new_state = DDDAEStateCreate(si);
  if (new_state == NULL) { return NULL; }

  DDDAEStateCopy(si, state, new_state);

  return new_state;
}

void DDDAEStateCopy(DDStaticInfo si, DDDAEState src, DDDAEState dst)
{
  SUNFunctionBegin(si->sunctx);

  memcpy(dst->diff_var_aliases,
         src->diff_var_aliases,
         si->N_diff * sizeof(*dst->diff_var_aliases));

  memcpy(dst->yy_diff_alias_row,
         src->yy_diff_alias_row,
         si->N_all_orders * sizeof(*src->yy_diff_alias_row));

  memcpy(dst->yp_diff_alias_row,
         src->yp_diff_alias_row,
         si->N_all_orders * sizeof(*src->yp_diff_alias_row));
}

SUNErrCode DDDAEStateUpdate(DDStaticInfo si, uint8_t* spec, DDDAEState state)

{
  SUNFunctionBegin(si->sunctx);

  memset(state->yy_diff_alias_row,
         -1,
         si->N_all_orders * sizeof(*state->yy_diff_alias_row));
  memset(state->yp_diff_alias_row,
         -1,
         si->N_all_orders * sizeof(*state->yp_diff_alias_row));

  sunindextype ofs = 0;
  for (sunindextype i = 0; i < si->N; ++i)
  {
    const uint8_t d = spec[i];

    for (uint8_t j = 0; j < d; ++j)
    {
      SUNAssert(ofs < si->N_diff, SUN_ERR_ARG_OUTOFRANGE);

      sunindextype yy_idx = si->var_deriv_chains[i][j],
                   yp_idx = si->var_deriv_chains[i][j + 1];

      SUNAssert(0 <= yy_idx && yy_idx < si->N_all_orders, SUN_ERR_ARG_OUTOFRANGE);
      SUNAssert(0 <= yp_idx && yp_idx < si->N_all_orders, SUN_ERR_ARG_OUTOFRANGE);

      state->diff_var_aliases[ofs] = (Pair_sunindextype){
        .fst = yy_idx,
        .snd = yp_idx,
      };

      sunindextype eqn                 = ofs + si->M_all_orders;
      state->yy_diff_alias_row[yy_idx] = eqn;
      state->yp_diff_alias_row[yp_idx] = eqn;

      ofs++;
    }
  }

  return SUN_SUCCESS;
}
