#include <string.h>
#include <sundials/priv/sundials_errors_impl.h>
#include "sundials/sundials_errors.h"
#include "sundials/sundials_types.h"

#include "dynamic_info.h"

void DDstateDestroy(DDstateMem* state_ptr)
{
  if (state_ptr == NULL || *state_ptr == NULL) { return; }

  DDstateMem state = *state_ptr;
  free(state->diff_var_aliases);
  free(state->yy_diff_alias_row);
  free(state->yp_diff_alias_row);
  free(state);

  *state_ptr = NULL;
}

DDstateMem DDstateCreate(DAEStruct st)
{
  DDstateMem state = calloc(1, sizeof(*state));
  if (state == NULL) { return NULL; }

  state->diff_var_aliases = malloc(st->N_diff * sizeof(*state->diff_var_aliases));
  if (state->diff_var_aliases == NULL) { goto fail; }

  state->yy_diff_alias_row =
    malloc(st->N_all_orders * sizeof(*state->yy_diff_alias_row));
  if (state->yy_diff_alias_row == NULL) { goto fail; }

  state->yp_diff_alias_row =
    malloc(st->N_all_orders * sizeof(*state->yp_diff_alias_row));
  if (state->yp_diff_alias_row == NULL) { goto fail; }

  return state;

fail:
  DDstateDestroy(&state);
  return NULL;
}

DDstateMem DDstateClone(DAEStruct st, DDstateMem state)
{
  DDstateMem new_state = DDstateCreate(st);
  if (new_state == NULL) { return NULL; }

  memcpy(new_state->diff_var_aliases,
         state->diff_var_aliases,
         st->N_diff * sizeof(*new_state->diff_var_aliases));

  memcpy(new_state->yy_diff_alias_row,
         state->yy_diff_alias_row,
         st->N_all_orders * sizeof(*state->yy_diff_alias_row));

  memcpy(new_state->yp_diff_alias_row,
         state->yy_diff_alias_row,
         st->N_all_orders * sizeof(*state->yp_diff_alias_row));

  return new_state;
}

SUNErrCode DDstateUpdate(DAEStruct st, uint8_t* spec, DDstateMem state)

{
  SUNFunctionBegin(st->sunctx);

  memset(state->yy_diff_alias_row,
         -1,
         st->N_all_orders * sizeof(*state->yy_diff_alias_row));
  memset(state->yp_diff_alias_row,
         -1,
         st->N_all_orders * sizeof(*state->yp_diff_alias_row));

  sunindextype ofs = 0;
  for (sunindextype i = 0; i < st->N; ++i)
  {
    const uint8_t d = spec[i];

    for (uint8_t j = 0; j < d; ++j)
    {
      SUNAssert(ofs < st->N_diff, SUN_ERR_ARG_OUTOFRANGE);

      sunindextype yy_idx = st->var_deriv_chains[i][j],
                   yp_idx = st->var_deriv_chains[i][j + 1];

      SUNAssert(0 <= yy_idx && yy_idx < st->N_all_orders, SUN_ERR_ARG_OUTOFRANGE);
      SUNAssert(0 <= yp_idx && yp_idx < st->N_all_orders, SUN_ERR_ARG_OUTOFRANGE);

      state->diff_var_aliases[ofs] = (Pair_sunindextype){
        .fst = yy_idx,
        .snd = yp_idx,
      };

      sunindextype eqn                 = ofs + st->M_all_orders;
      state->yy_diff_alias_row[yy_idx] = eqn;
      state->yp_diff_alias_row[yp_idx] = eqn;

      ofs++;
    }
  }

  return SUN_SUCCESS;
}
