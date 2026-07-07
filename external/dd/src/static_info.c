#include <stdint.h>
#include <stdlib.h>
#include <sundials/sundials_math.h>

#include "static_info.h"

void DDStaticInfoDestroy(DDStaticInfo si)
{
  if (si)
  {
    sunindextype N = si->N;
    uint8_t K      = si->K;

    if (si->eqnofs != NULL)
    {
      free(si->eqnofs);
      si->eqnofs = NULL;
    }

    if (si->varofs != NULL)
    {
      free(si->varofs);
      si->varofs = NULL;
    }

    sunindextype** var_deriv_chains = si->var_deriv_chains;
    if (var_deriv_chains != NULL)
    {
      for (sunindextype j = 0; j < N; ++j) { var_deriv_chains[j] = NULL; }
      free(var_deriv_chains);
      si->var_deriv_chains_flat = NULL;
    }

    if (si->var_deriv_chains_flat != NULL)
    {
      free(si->var_deriv_chains_flat);
      si->var_deriv_chains_flat = NULL;
    }

    sunindextype** eqns = si->eqns_k;
    if (eqns != NULL)
    {
      for (size_t k = 0; k < K; ++k) { eqns[k] = NULL; }
      free(eqns);
      si->eqns_k = NULL;
    }

    sunindextype** vars = si->vars_k;
    if (vars != NULL)
    {
      for (size_t k = 0; k < K; ++k) { vars[k] = NULL; }
      free(vars);
      si->vars_k = NULL;
    }

    if (si->eqns_k_flat != NULL)
    {
      free(si->eqns_k_flat);
      si->eqns_k_flat = NULL;
    }

    if (si->vars_k_flat != NULL)
    {
      free(si->vars_k_flat);
      si->vars_k_flat = NULL;
    }

    free(si);
  }
}

DDStaticInfo DDStaticInfoCreate(SUNContext sunctx,
                                sunindextype N,
                                const uint8_t eqnofs[static N],
                                const uint8_t varofs[static N],
                                const sunindextype* var_deriv_chains[static N],
                                const char** eqn_names,
                                const char** var_names)
{
  DDStaticInfo si = malloc(sizeof(*si));
  if (si == NULL) { return NULL; }

  si->sunctx = sunctx;
  si->N      = N;

  uint8_t K = 0;
  for (sunindextype i = 0; i < N; ++i)
  {
    if (varofs[i] > K) { K = varofs[i]; }
  }
  K++;
  si->K = K;

  si->M_all_orders = 0;
  si->N_all_orders = 0;
  si->N_backwards  = 0;
  for (sunindextype i = 0; i < N; ++i)
  {
    si->M_all_orders += eqnofs[i] + 1;
    si->N_all_orders += varofs[i] + 1;
    si->N_backwards += SUNMAX(varofs[i], 1);
  }
  si->N_diff = si->N_all_orders - si->M_all_orders;

  assert(si->N_diff >= 0);

  si->eqnofs = malloc(N * sizeof(uint8_t));
  if (si->eqnofs == NULL)
  {
    DDStaticInfoDestroy(si);
    return NULL;
  }

  si->varofs = malloc(N * sizeof(uint8_t));
  if (si->varofs == NULL)
  {
    DDStaticInfoDestroy(si);
    return NULL;
  }

  si->var_deriv_chains = malloc(N * sizeof(*si->var_deriv_chains));
  if (si->var_deriv_chains == NULL)
  {
    DDStaticInfoDestroy(si);
    return NULL;
  }
  si->var_deriv_chains_flat = malloc(si->N_all_orders * sizeof(sunindextype));
  if (si->var_deriv_chains_flat == NULL)
  {
    DDStaticInfoDestroy(si);
    return NULL;
  }
  sunindextype ofs = 0;
  for (sunindextype j = 0; j < N; ++j)
  {
    si->var_deriv_chains[j] = si->var_deriv_chains_flat + ofs;
    for (uint8_t k = 0; k <= varofs[j]; ++k)
    {
      si->var_deriv_chains_flat[ofs] = var_deriv_chains[j][k];
      ++ofs;
    }
  }

  si->M_k = calloc(K, sizeof(sunindextype));
  if (si->M_k == NULL)
  {
    DDStaticInfoDestroy(si);
    return NULL;
  }

  si->N_k = calloc(K, sizeof(sunindextype));
  if (si->N_k == NULL)
  {
    DDStaticInfoDestroy(si);
    return NULL;
  }

  si->eqns_k = malloc(K * sizeof(sunindextype*));
  if (si->eqns_k == NULL)
  {
    DDStaticInfoDestroy(si);
    return NULL;
  }

  si->vars_k = malloc(K * sizeof(sunindextype*));
  if (si->vars_k == NULL)
  {
    DDStaticInfoDestroy(si);
    return NULL;
  }

  si->eqns_k_flat = malloc(si->M_all_orders * sizeof(sunindextype));
  if (si->eqns_k_flat == NULL)
  {
    DDStaticInfoDestroy(si);
    return NULL;
  }

  si->vars_k_flat = malloc(si->N_all_orders * sizeof(sunindextype));
  if (si->vars_k_flat == NULL)
  {
    DDStaticInfoDestroy(si);
    return NULL;
  }

  for (sunindextype i = 0; i < N; ++i)
  {
    si->eqnofs[i] = eqnofs[i];
    si->varofs[i] = varofs[i];
  }

  sunindextype i_ofs = 0;
  sunindextype j_ofs = 0;
  for (uint8_t k = 0; k < K; ++k)
  {
    for (sunindextype i = 0; i < N; ++i)
    {
      if (DDSI_STAGE_FROM_INDEX(si, k) + (int)eqnofs[i] >= 0)
      {
        si->eqns_k_flat[si->M_k[k] + i_ofs] = i;
        si->M_k[k]++;
      }

      if (DDSI_STAGE_FROM_INDEX(si, k) + (int)varofs[i] >= 0)
      {
        si->vars_k_flat[si->N_k[k] + j_ofs] = i;
        si->N_k[k]++;
      }
    }
    si->eqns_k[k] = si->eqns_k_flat + i_ofs;
    si->vars_k[k] = si->vars_k_flat + j_ofs;
    i_ofs += si->M_k[k];
    j_ofs += si->N_k[k];
  }

  si->eqn_names = eqn_names;
  si->var_names = var_names;

  return si;
}

void DDStaticInfoPrint(DDStaticInfo si, FILE* file)
{
  fprintf(file, "--- START STRUCTURE ----\n");
  fprintf(file, "Equation offsets:\n");
  for (sunindextype i = 0; i < si->N; ++i)
  {
    fprintf(file, "\t[%ld]%s:\t%d\n", i, DDSI_EQN_NAME(si, i), si->eqnofs[i]);
  }
  fprintf(file, "Variable offsets:\n");
  for (sunindextype j = 0; j < si->N; ++j)
  {
    {
      fprintf(file, "\t[%ld]%s:\t%d\n", j, DDSI_VAR_NAME(si, j), si->varofs[j]);
    }
  }
  for (uint8_t k = 0; k < si->K; ++k)
  {
    fprintf(file, "Stage k = %d:\n", DDSI_STAGE_FROM_INDEX(si, k));
    fprintf(file, "Equations:\n");
    for (sunindextype l = 0; l < si->M_k[k]; ++l)
    {
      const sunindextype i = si->eqns_k[k][l];
      fprintf(file,
              "\t[%ld]%s^(%d)\n",
              i,
              DDSI_EQN_NAME(si, i),
              DDSI_EQN_ORDER(si, k, i));
    }
    fprintf(file, "Variables:\n");
    for (sunindextype l = 0; l < si->N_k[k]; ++l)
    {
      const sunindextype j = si->vars_k[k][l];
      fprintf(file,
              "\t[%ld]%s^(%d)\n",
              j,
              DDSI_VAR_NAME(si, j),
              DDSI_VAR_ORDER(si, k, j));
    }
  }
  fprintf(file, "--- END STRUCTURE ------\n");
}
