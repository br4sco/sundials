#include <stdint.h>
#include <stdlib.h>
#include <sundials/sundials_math.h>

#include "structure.h"

void STDestroy(DAEStruct st)
{
  if (st)
  {
    sunindextype size = st->DAE_size;
    uint8_t K         = st->K;

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

    sunindextype** var_idx_map = st->var_idx_map;
    if (var_idx_map != NULL)
    {
      for (sunindextype j = 0; j < size; ++j) { var_idx_map[j] = NULL; }
      free(var_idx_map);
      st->var_idx_map_data = NULL;
    }

    if (st->var_idx_map_data != NULL)
    {
      free(st->var_idx_map_data);
      st->var_idx_map_data = NULL;
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

    if (st->eqns != NULL)
    {
      free(st->eqns);
      st->eqns = NULL;
    }

    if (st->vars != NULL)
    {
      free(st->vars);
      st->vars = NULL;
    }

    free(st);
  }
}

DAEStruct STCreate(
  sunindextype size,
  const uint8_t eqnofs[static size],
  const uint8_t varofs[static size],
  const sunindextype* var_idx_map[static size],
  const char** eqn_names,
  const char** var_names
)
{
  DAEStruct st = malloc(sizeof(*st));
  if (st == NULL) { return NULL; }

  st->DAE_size = size;

  uint8_t K = 0;
  for (sunindextype i = 0; i < size; ++i)
  {
    if (varofs[i] > K) { K = varofs[i]; }
  }
  K++;
  st->K = K;

  st->M                 = 0;
  st->N                 = 0;
  st->DAE_backward_size = 0;
  for (sunindextype i = 0; i < size; ++i)
  {
    st->M += eqnofs[i] + 1;
    st->N += varofs[i] + 1;
    st->DAE_backward_size += SUNMAX(varofs[i], 1);
  }

  st->eqnofs = malloc(size * sizeof(uint8_t));
  if (st->eqnofs == NULL)
  {
    STDestroy(st);
    return NULL;
  }

  st->varofs = malloc(size * sizeof(uint8_t));
  if (st->varofs == NULL)
  {
    STDestroy(st);
    return NULL;
  }

  st->var_idx_map = malloc(size * sizeof(sunindextype));
  if (st->var_idx_map == NULL)
  {
    STDestroy(st);
    return NULL;
  }
  st->var_idx_map_data = malloc(st->N * sizeof(sunindextype));
  if (st->var_idx_map_data == NULL)
  {
    STDestroy(st);
    return NULL;
  }
  sunindextype ofs = 0;
  for (sunindextype j = 0; j < size; ++j)
  {
    st->var_idx_map[j] = st->var_idx_map_data + ofs;
    for (uint8_t k = 0; k <= varofs[j]; ++k)
    {
      st->var_idx_map_data[ofs] = var_idx_map[j][k];
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

  st->eqns = malloc(st->M * sizeof(sunindextype));
  if (st->eqns == NULL)
  {
    STDestroy(st);
    return NULL;
  }

  st->vars = malloc(st->N * sizeof(sunindextype));
  if (st->vars == NULL)
  {
    STDestroy(st);
    return NULL;
  }

  for (sunindextype i = 0; i < size; ++i)
  {
    st->eqnofs[i] = eqnofs[i];
    st->varofs[i] = varofs[i];
  }

  sunindextype i_ofs = 0;
  sunindextype j_ofs = 0;
  for (uint8_t k = 0; k < K; ++k)
  {
    for (sunindextype i = 0; i < size; ++i)
    {
      if (ST_STAGE_FROM_INDEX(st, k) + (int)eqnofs[i] >= 0)
      {
        st->eqns[st->M_k[k] + i_ofs] = i;
        st->M_k[k]++;
      }

      if (ST_STAGE_FROM_INDEX(st, k) + (int)varofs[i] >= 0)
      {
        st->vars[st->N_k[k] + j_ofs] = i;
        st->N_k[k]++;
      }
    }
    st->eqns_k[k] = st->eqns + i_ofs;
    st->vars_k[k] = st->vars + j_ofs;
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
  for (sunindextype i = 0; i < st->DAE_size; ++i)
  {
    fprintf(file, "\t[%ld]%s:\t%d\n", i, ST_EQN_NAME(st, i), st->eqnofs[i]);
  }
  fprintf(file, "Variable offsets:\n");
  for (sunindextype j = 0; j < st->DAE_size; ++j)
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
      fprintf(file, "\t[%ld]%s^(%d)\n", i, ST_EQN_NAME(st, i), ST_EQN_ORDER(st, k, i));
    }
    fprintf(file, "Variables:\n");
    for (sunindextype l = 0; l < st->N_k[k]; ++l)
    {
      const sunindextype j = st->vars_k[k][l];
      fprintf(file, "\t[%ld]%s^(%d)\n", j, ST_VAR_NAME(st, j), ST_VAR_ORDER(st, k, j));
    }
  }
  fprintf(file, "--- END STRUCTURE ------\n");
}
