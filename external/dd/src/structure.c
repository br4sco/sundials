#include <stdint.h>
#include <stdlib.h>
#include <sundials/sundials_math.h>

#include "structure.h"

void STDestroy(Structure* st)
{
  if (st)
  {
    uint8_t K = st->st_K;

    if (st->st_eqnofs != NULL)
    {
      free(st->st_eqnofs);
      st->st_eqnofs = NULL;
    }

    if (st->st_varofs != NULL)
    {
      free(st->st_varofs);
      st->st_varofs = NULL;
    }

    if (st->st_acc_eqnofs != NULL)
    {
      free(st->st_acc_eqnofs);
      st->st_acc_eqnofs = NULL;
    }

    if (st->st_acc_varofs != NULL)
    {
      free(st->st_acc_varofs);
      st->st_acc_varofs = NULL;
    }

    sunindextype** eqns = st->st_eqns;
    if (eqns != NULL)
    {
      for (size_t k = 0; k < K; ++k) { eqns[k] = NULL; }
      free(eqns);
      st->st_eqns = NULL;
    }

    sunindextype** vars = st->st_vars;
    if (vars != NULL)
    {
      for (size_t k = 0; k < K; ++k) { vars[k] = NULL; }
      free(vars);
      st->st_vars = NULL;
    }

    if (st->st_eqnsdata != NULL)
    {
      free(st->st_eqnsdata);
      st->st_eqnsdata = NULL;
    }

    if (st->st_varsdata != NULL)
    {
      free(st->st_varsdata);
      st->st_varsdata = NULL;
    }

    free(st);
  }
}

Structure* STCreate(sunindextype N, const uint8_t eqnofs[static N],
                    const uint8_t varofs[static N], char** eqn_names,
                    char** var_names)
{
  Structure* st = malloc(sizeof(*st));
  if (st == NULL) { return NULL; }

  st->st_DAE_N = N;

  uint8_t K = 0;
  for (sunindextype i = 0; i < N; ++i)
  {
    if (varofs[i] > K) { K = varofs[i]; }
  }
  K++;
  st->st_K = K;

  st->st_M      = 0;
  st->st_N      = 0;
  st->st_DAE_N1 = 0;
  for (sunindextype i = 0; i < N; ++i)
  {
    st->st_M += eqnofs[i] + 1;
    st->st_N += varofs[i] + 1;
    st->st_DAE_N1 += SUNMAX(varofs[i], 1);
  }

  st->st_eqnofs = malloc(N * sizeof(uint8_t));
  if (st->st_eqnofs == NULL)
  {
    STDestroy(st);
    return NULL;
  }

  st->st_varofs = malloc(N * sizeof(uint8_t));
  if (st->st_varofs == NULL)
  {
    STDestroy(st);
    return NULL;
  }

  st->st_acc_eqnofs = malloc(N * sizeof(sunindextype));
  if (st->st_acc_eqnofs == NULL)
  {
    STDestroy(st);
    return NULL;
  }

  st->st_acc_varofs = malloc(N * sizeof(sunindextype));
  if (st->st_acc_varofs == NULL)
  {
    STDestroy(st);
    return NULL;
  }

  st->st_Mk = calloc(K, sizeof(sunindextype));
  if (st->st_Mk == NULL)
  {
    STDestroy(st);
    return NULL;
  }

  st->st_Nk = calloc(K, sizeof(sunindextype));
  if (st->st_Nk == NULL)
  {
    STDestroy(st);
    return NULL;
  }

  st->st_eqns = malloc(K * sizeof(sunindextype*));
  if (st->st_eqns == NULL)
  {
    STDestroy(st);
    return NULL;
  }

  st->st_vars = malloc(K * sizeof(sunindextype*));
  if (st->st_vars == NULL)
  {
    STDestroy(st);
    return NULL;
  }

  st->st_eqnsdata = malloc(st->st_M * sizeof(sunindextype));
  if (st->st_eqnsdata == NULL)
  {
    STDestroy(st);
    return NULL;
  }

  st->st_varsdata = malloc(st->st_N * sizeof(sunindextype));
  if (st->st_varsdata == NULL)
  {
    STDestroy(st);
    return NULL;
  }

  sunindextype eacc = 0;
  sunindextype vacc = 0;
  for (sunindextype i = 0; i < N; ++i)
  {
    const uint8_t e      = eqnofs[i];
    const uint8_t v      = varofs[i];
    st->st_eqnofs[i]     = e;
    st->st_varofs[i]     = v;
    st->st_acc_eqnofs[i] = eacc;
    st->st_acc_varofs[i] = vacc;
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
        st->st_eqnsdata[st->st_Mk[k] + i_ofs] = i;
        st->st_Mk[k]++;
      }

      if (ST_STAGE_FROM_INDEX(st, k) + (int)varofs[i] >= 0)
      {
        st->st_varsdata[st->st_Nk[k] + j_ofs] = i;
        st->st_Nk[k]++;
      }
    }
    st->st_eqns[k] = st->st_eqnsdata + i_ofs;
    st->st_vars[k] = st->st_varsdata + j_ofs;
    i_ofs += st->st_Mk[k];
    j_ofs += st->st_Nk[k];
  }

  st->st_eqnnames = eqn_names;
  st->st_varnames = var_names;

  return st;
}
