#include <stdint.h>
#include <stdlib.h>
#include <sundials/sundials_math.h>

#include "structure.h"

void STDestroy(DAEStruct st)
{
  if (st)
  {
    uint8_t K = st->K;

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

    if (st->eqn_to_idx != NULL)
    {
      free(st->eqn_to_idx);
      st->eqn_to_idx = NULL;
    }

    if (st->var_to_idx != NULL)
    {
      free(st->var_to_idx);
      st->var_to_idx = NULL;
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

DAEStruct STCreate(sunindextype N,
               const uint8_t eqnofs[static N],
               const uint8_t varofs[static N],
               char** eqn_names,
               char** var_names)
{
  DAEStruct st = malloc(sizeof(*st));
  if (st == NULL) { return NULL; }

  st->DAE_size = N;

  uint8_t K = 0;
  for (sunindextype i = 0; i < N; ++i)
  {
    if (varofs[i] > K) { K = varofs[i]; }
  }
  K++;
  st->K = K;

  st->M             = 0;
  st->N             = 0;
  st->DAE_1ord_size = 0;
  for (sunindextype i = 0; i < N; ++i)
  {
    st->M += eqnofs[i] + 1;
    st->N += varofs[i] + 1;
    st->DAE_1ord_size += SUNMAX(varofs[i], 1);
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

  st->eqn_to_idx = malloc(N * sizeof(sunindextype));
  if (st->eqn_to_idx == NULL)
  {
    STDestroy(st);
    return NULL;
  }

  st->var_to_idx = malloc(N * sizeof(sunindextype));
  if (st->var_to_idx == NULL)
  {
    STDestroy(st);
    return NULL;
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

  sunindextype eacc = 0;
  sunindextype vacc = 0;
  for (sunindextype i = 0; i < N; ++i)
  {
    const uint8_t e   = eqnofs[i];
    const uint8_t v   = varofs[i];
    st->eqnofs[i]     = e;
    st->varofs[i]     = v;
    st->eqn_to_idx[i] = eacc;
    st->var_to_idx[i] = vacc;
    eacc += e + 1;
    vacc += v + 1;
  }

  sunindextype i_ofs = 0;
  sunindextype j_ofs = 0;
  for (uint8_t k = 0; k < K; ++k)
  {
    for (sunindextype i = 0; i < N; ++i)
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
