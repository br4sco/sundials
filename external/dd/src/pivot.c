#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <sundials/priv/sundials_errors_impl.h>
#include <sundials/sundials_matrix.h>
#include <sunmatrix/sunmatrix_dense.h>

#include "matrix.h"
#include "pivot.h"
#include "structure.h"
#include "sundials/sundials_errors.h"

PivMem PIVCreate(SUNContext sunctx, DAEStruct st, DDMatrix A)
{
  SUNFunctionBegin(sunctx);

  PivMem pm = malloc(sizeof(*pm));
  SUNAssertNull(pm, SUN_ERR_MALLOC_FAIL);

  pm->sunctx = sunctx;

  const sunindextype dae_size = st->DAE_size;
  const uint8_t K             = st->K;

  pm->DAE_size = dae_size;
  pm->K        = K;

  pm->spec = calloc(dae_size, sizeof(*pm->spec));
  SUNAssertNull(pm->spec, SUN_ERR_MALLOC_FAIL);

  pm->NNZ_spec = 0;
  pm->NZ_spec  = malloc(dae_size * sizeof(*pm->NZ_spec));
  SUNAssertNull(pm->NZ_spec, SUN_ERR_MALLOC_FAIL);

  pm->N_diff_vars = st->N - st->M;
  pm->diff_vars   = malloc(pm->N_diff_vars * sizeof(*pm->diff_vars));
  SUNAssertNull(pm->diff_vars, SUN_ERR_MALLOC_FAIL);

  pm->known_k = malloc(K * sizeof(*pm->known_k));
  SUNAssertNull(pm->known_k, SUN_ERR_MALLOC_FAIL);

  pm->vars_k = malloc(K * sizeof(*pm->vars_k));
  SUNAssertNull(pm->vars_k, SUN_ERR_MALLOC_FAIL);

  pm->J_k = malloc(K * sizeof(DDMatrix*));
  SUNAssertNull(pm->J_k, SUN_ERR_MALLOC_FAIL);

  pm->wss = malloc(K * sizeof(DDMatrixWorkspace*));
  SUNAssertNull(pm->wss, SUN_ERR_MALLOC_FAIL);

  pm->known = calloc(K * dae_size, sizeof(*pm->known));
  SUNAssertNull(pm->known, SUN_ERR_MALLOC_FAIL);

  pm->vars = calloc(st->N, sizeof(*pm->vars));
  SUNAssertNull(pm->vars, SUN_ERR_MALLOC_FAIL);

  sunindextype ofs = 0;
  for (uint8_t k = 0; k < K; ++k)
  {
    const sunindextype M  = st->M_k[k];
    const sunindextype N  = st->N_k[k];
    const sunindextype* I = st->eqns_k[k];
    const sunindextype* J = st->vars_k[k];

    SUNAssertNull(N > 0, SUN_ERR_OP_FAIL);

    if (M == 0)
    {
      pm->J_k[k] = NULL;
      pm->wss[k] = NULL;
    }
    else
    {
      DDMatrix A_sub = DDMatCloneSub(A, M, I, N, J);
      SUNCheckLastErrNull();
      pm->J_k[k] = A_sub;

      DDMatrixWorkspace ws = DDMatCreateWS(A_sub);
      SUNCheckLastErrNull();
      pm->wss[k] = ws;
    }

    pm->known_k[k] = pm->known + k * dae_size;
    pm->vars_k[k]  = pm->vars + ofs;
    ofs += N;
  }

  return pm;
}

void PIVDestroy(PivMem pm)
{
  if (pm == NULL) { return; }

  uint8_t K = pm->K;

  if (pm->spec != NULL)
  {
    free(pm->spec);
    pm->spec = NULL;
  }

  if (pm->NZ_spec != NULL)
  {
    free(pm->NZ_spec);
    pm->NZ_spec = NULL;
  }

  if (pm->diff_vars != NULL)
  {
    free(pm->diff_vars);
    pm->diff_vars = NULL;
  }

  if (pm->known_k != NULL)
  {
    for (size_t k = 0; k < K; ++k) { pm->known_k[k] = NULL; }
    free(pm->known_k);
    pm->known_k = NULL;
  }

  if (pm->vars_k != NULL)
  {
    for (size_t k = 0; k < K; ++k) { pm->vars_k[k] = NULL; }
    free(pm->vars_k);
    pm->vars_k = NULL;
  }

  if (pm->J_k != NULL)
  {
    for (size_t k = 0; k < K; ++k)
    {
      DDMatrix submat = pm->J_k[k];
      if (submat)
      {
        SUNMatDestroy(DDMatGetSUNMat(submat));
        DDMatDestroy(submat);
      }
      pm->J_k[k] = NULL;
    }
    free(pm->J_k);
    pm->J_k = NULL;
  }

  if (pm->wss != NULL)
  {
    for (size_t k = 0; k < K; ++k)
    {
      DDMatWSDestroy(pm->wss[k]);
      pm->wss[k] = NULL;
    }
    free(pm->wss);
    pm->wss = NULL;
  }

  if (pm->known != NULL)
  {
    free(pm->known);
    pm->known = NULL;
  }

  if (pm->vars != NULL)
  {
    free(pm->vars);
    pm->vars = NULL;
  }

  free(pm);
}

static SUNErrCode PDReset(PivMem pm)
{
  const sunindextype N = pm->DAE_size;
  const uint8_t K      = pm->K;

  memset(pm->spec, 0, N * sizeof(*pm->spec));
  memset(pm->known, SUNFALSE, N * K * sizeof(*pm->known));

  return SUN_SUCCESS;
}

SUNErrCode PIVPivot(DAEStruct st, DDMatrix A, sunrealtype tol, PivMem pm)
{
  SUNFunctionBegin(pm->sunctx);

  SUNCheckCall(PDReset(pm));

  for (uint8_t k = 0; k < st->K; ++k)
  {
    const sunindextype M = st->M_k[k];
    const sunindextype N = st->N_k[k];

    SUNAssert(N > 0, SUN_ERR_OP_FAIL);

    const sunindextype* vars = st->vars_k[k];
    sunindextype* pm_vars    = pm->vars_k[k];
    memcpy(pm_vars, vars, N * sizeof(sunindextype));

    if (M == 0) { continue; }

    const sunindextype* eqns = st->eqns_k[k];

    const int dof = (int)N - (int)M;

    const uint8_t* varofs = st->varofs;

    sunbooleantype* pm_known = pm->known_k[k];
    if (dof == 0)
    {
      for (sunindextype l = 0; l < N; ++l) { pm_known[l] = SUNTRUE; }
    }
    else
    {
      for (sunindextype l = 0; l < N; ++l)
      {
        if (varofs[vars[l]] == 0) { pm_known[vars[l]] = SUNTRUE; }
      }

      DDMatrix submat = pm->J_k[k];
      SUNCheckCall(DDCopySub(A, submat, M, eqns, N, vars));

      const DDMatrixWorkspace ws = pm->wss[k];
      SUNCheckCall(DDMatPivot(submat, ws, tol, N, pm_vars));

      for (sunindextype l = 0; l < M; ++l) { pm_known[pm_vars[l]] = SUNTRUE; }
    }
  }

  return SUN_SUCCESS;
}

/* void PSPrintSubmat(const Structure st[static 1], const PivMem pm[static 1], */
/*                    uint8_t k, FILE* file) */
/* { */
/*   for (sunindextype i = 0; i < st->st_Nk[k]; ++i) */
/*   { */
/*     sunindextype j = pm->pm_vars[k][i]; */
/*     fprintf(file, "\td%d%s", ST_VAR_ORDER(st, k, j), ST_VAR_NAME(st, j)); */
/*   } */

/*   SUNMatrix mat = DDMatGetSUNMat(pm->pm_jacs[k]); */
/*   if (SUNMatGetID(mat) == SUNMATRIX_DENSE) { SUNDenseMatrix_Print(mat, file); } */
/* } */

static sunindextype ComputeNZSpec(sunindextype N,
                                  const uint8_t spec[N],
                                  sunindextype* NZ_spec)
{
  sunindextype len = 0;
  for (sunindextype i = 0; i < N; ++i)
  {
    if (spec[i] > 0)
    {
      NZ_spec[len] = i;
      len++;
    }
  }

  return len;
}

static SUNErrCode ComputeDiffVars(DAEStruct st, PivMem pm)
{
  SUNFunctionBegin(pm->sunctx);

  sunindextype ofs = 0;
  for (sunindextype i = 0; i < pm->NNZ_spec; ++i)
  {
    const sunindextype var = pm->NZ_spec[i], varofs = st->var_to_idx[var];
    const uint8_t dd = pm->spec[var];
    for (uint8_t j = 0; j < dd; ++j)
    {
      SUNAssert(ofs < pm->N_diff_vars, SUN_ERR_ARG_OUTOFRANGE);
      pm->diff_vars[ofs] = varofs + j;
      ofs++;
    }
  }

  return SUN_SUCCESS;
}

SUNErrCode PIVComputeDDSpec(DAEStruct st, PivMem pm)
{
  SUNFunctionBegin(pm->sunctx);

  for (uint8_t k = 0; k < st->K; ++k)
  {
    for (sunindextype l = 0; l < st->N_k[k]; ++l)
    {
      const sunindextype j = pm->vars_k[k][l];
      if (!pm->known_k[k][j])
      {
        SUNAssert(k <= 0 || !pm->known_k[k - 1][j], SUN_ERR_OP_FAIL);
        pm->spec[j]++;
      }
    }
  }

  pm->NNZ_spec = ComputeNZSpec(st->DAE_size, pm->spec, pm->NZ_spec);
  SUNCheckCall(ComputeDiffVars(st, pm));

  return SUN_SUCCESS;
}

SUNErrCode PIVUpdateDDSpec(DAEStruct st, const uint8_t spec[static 1], PivMem pm)
{
  SUNFunctionBegin(pm->sunctx);

  memcpy(pm->spec, spec, pm->DAE_size * sizeof(*spec));
  pm->NNZ_spec = ComputeNZSpec(pm->DAE_size, pm->spec, pm->NZ_spec);
  SUNCheckCall(ComputeDiffVars(st, pm));

  return SUN_SUCCESS;
}
