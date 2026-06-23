#include <stdint.h>
#include <stdlib.h>
#include <sundials/sundials_math.h>

#include "structure.h"

void STDestroy(DAEStruct st)
{
  if (st)
  {
    sunindextype N = st->N;
    uint8_t K      = st->K;

    if (st->eqnofs != NULL)
    {
      free(st->eqnofs);
      st->eqnofs = NULL;
    }

    if (st->varofs != NULL)
    {
      free(st->varofs);
      st->varofs = NULL;
    }

    sunindextype** var_idx_map = st->var_deriv_chains;
    if (var_idx_map != NULL)
    {
      for (sunindextype j = 0; j < N; ++j) { var_idx_map[j] = NULL; }
      free(var_idx_map);
      st->var_deriv_chains_flat = NULL;
    }

    if (st->var_deriv_chains_flat != NULL)
    {
      free(st->var_deriv_chains_flat);
      st->var_deriv_chains_flat = NULL;
    }

    sunindextype** eqns = st->eqns_k;
    if (eqns != NULL)
    {
      for (size_t k = 0; k < K; ++k) { eqns[k] = NULL; }
      free(eqns);
      st->eqns_k = NULL;
    }

    sunindextype** vars = st->vars_k;
    if (vars != NULL)
    {
      for (size_t k = 0; k < K; ++k) { vars[k] = NULL; }
      free(vars);
      st->vars_k = NULL;
    }

    if (st->eqns_k_flat != NULL)
    {
      free(st->eqns_k_flat);
      st->eqns_k_flat = NULL;
    }

    if (st->vars_k_flat != NULL)
    {
      free(st->vars_k_flat);
      st->vars_k_flat = NULL;
    }

    free(st);
  }
}

DAEStruct STCreate(sunindextype N,
                   const uint8_t eqnofs[static N],
                   const uint8_t varofs[static N],
                   const sunindextype* var_idx_map[static N],
                   const char** eqn_names,
                   const char** var_names)
{
  DAEStruct st = malloc(sizeof(*st));
  if (st == NULL) { return NULL; }

  st->N = N;

  uint8_t K = 0;
  for (sunindextype i = 0; i < N; ++i)
  {
    if (varofs[i] > K) { K = varofs[i]; }
  }
  K++;
  st->K = K;

  st->M_all_orders = 0;
  st->N_all_orders = 0;
  st->N_backwards  = 0;
  for (sunindextype i = 0; i < N; ++i)
  {
    st->M_all_orders += eqnofs[i] + 1;
    st->N_all_orders += varofs[i] + 1;
    st->N_backwards += SUNMAX(varofs[i], 1);
  }

  st->eqnofs = malloc(N * sizeof(uint8_t));
  if (st->eqnofs == NULL)
  {
    STDestroy(st);
    return NULL;
  }

  st->varofs = malloc(N * sizeof(uint8_t));
  if (st->varofs == NULL)
  {
    STDestroy(st);
    return NULL;
  }

  st->var_deriv_chains = malloc(N * sizeof(*st->var_deriv_chains));
  if (st->var_deriv_chains == NULL)
  {
    STDestroy(st);
    return NULL;
  }
  st->var_deriv_chains_flat = malloc(st->N_all_orders * sizeof(sunindextype));
  if (st->var_deriv_chains_flat == NULL)
  {
    STDestroy(st);
    return NULL;
  }
  sunindextype ofs = 0;
  for (sunindextype j = 0; j < N; ++j)
  {
    st->var_deriv_chains[j] = st->var_deriv_chains_flat + ofs;
    for (uint8_t k = 0; k <= varofs[j]; ++k)
    {
      st->var_deriv_chains_flat[ofs] = var_idx_map[j][k];
      ++ofs;
    }
  }

  st->M_k = calloc(K, sizeof(sunindextype));
  if (st->M_k == NULL)
  {
    STDestroy(st);
    return NULL;
  }

  st->N_k = calloc(K, sizeof(sunindextype));
  if (st->N_k == NULL)
  {
    STDestroy(st);
    return NULL;
  }

  st->eqns_k = malloc(K * sizeof(sunindextype*));
  if (st->eqns_k == NULL)
  {
    STDestroy(st);
    return NULL;
  }

  st->vars_k = malloc(K * sizeof(sunindextype*));
  if (st->vars_k == NULL)
  {
    STDestroy(st);
    return NULL;
  }

  st->eqns_k_flat = malloc(st->M_all_orders * sizeof(sunindextype));
  if (st->eqns_k_flat == NULL)
  {
    STDestroy(st);
    return NULL;
  }

  st->vars_k_flat = malloc(st->N_all_orders * sizeof(sunindextype));
  if (st->vars_k_flat == NULL)
  {
    STDestroy(st);
    return NULL;
  }

  for (sunindextype i = 0; i < N; ++i)
  {
    st->eqnofs[i] = eqnofs[i];
    st->varofs[i] = varofs[i];
  }

  sunindextype i_ofs = 0;
  sunindextype j_ofs = 0;
  for (uint8_t k = 0; k < K; ++k)
  {
    for (sunindextype i = 0; i < N; ++i)
    {
      if (ST_STAGE_FROM_INDEX(st, k) + (int)eqnofs[i] >= 0)
      {
        st->eqns_k_flat[st->M_k[k] + i_ofs] = i;
        st->M_k[k]++;
      }

      if (ST_STAGE_FROM_INDEX(st, k) + (int)varofs[i] >= 0)
      {
        st->vars_k_flat[st->N_k[k] + j_ofs] = i;
        st->N_k[k]++;
      }
    }
    st->eqns_k[k] = st->eqns_k_flat + i_ofs;
    st->vars_k[k] = st->vars_k_flat + j_ofs;
    i_ofs += st->M_k[k];
    j_ofs += st->N_k[k];
  }

  st->eqn_names = eqn_names;
  st->var_names = var_names;

  return st;
}

void STPrint(DAEStruct st, FILE* file)
{
  fprintf(file, "--- START STRUCTURE ----\n");
  fprintf(file, "Equation offsets:\n");
  for (sunindextype i = 0; i < st->N; ++i)
  {
    fprintf(file, "\t[%ld]%s:\t%d\n", i, ST_EQN_NAME(st, i), st->eqnofs[i]);
  }
  fprintf(file, "Variable offsets:\n");
  for (sunindextype j = 0; j < st->N; ++j)
  {
    {
      fprintf(file, "\t[%ld]%s:\t%d\n", j, ST_VAR_NAME(st, j), st->varofs[j]);
    }
  }
  for (uint8_t k = 0; k < st->K; ++k)
  {
    fprintf(file, "Stage k = %d:\n", ST_STAGE_FROM_INDEX(st, k));
    fprintf(file, "Equations:\n");
    for (sunindextype l = 0; l < st->M_k[k]; ++l)
    {
      const sunindextype i = st->eqns_k[k][l];
      fprintf(file, "\t[%ld]%s^(%d)\n", i, ST_EQN_NAME(st, i),
              ST_EQN_ORDER(st, k, i));
    }
    fprintf(file, "Variables:\n");
    for (sunindextype l = 0; l < st->N_k[k]; ++l)
    {
      const sunindextype j = st->vars_k[k][l];
      fprintf(file, "\t[%ld]%s^(%d)\n", j, ST_VAR_NAME(st, j),
              ST_VAR_ORDER(st, k, j));
    }
  }
  fprintf(file, "--- END STRUCTURE ------\n");
}
