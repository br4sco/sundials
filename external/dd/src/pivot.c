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

PivMem* PMCreate(SUNContext sunctx, const Structure st[static 1],
                 const DDMatrix A[static 1])
{
  SUNFunctionBegin(sunctx);

  PivMem* pm = malloc(sizeof(*pm));
  SUNAssertNull(pm, SUN_ERR_MALLOC_FAIL);

  pm->sunctx = sunctx;

  const sunindextype N = st->st_DAE_N;
  const uint8_t K      = st->st_K;

  pm->pm_DAE_N = N;
  pm->pm_K     = K;

  pm->pm_spec = calloc(N, sizeof(*pm->pm_spec));
  SUNAssertNull(pm->pm_spec, SUN_ERR_MALLOC_FAIL);

  pm->pm_NNZ_spec = 0;
  pm->pm_NZ_spec  = malloc(N * sizeof(*pm->pm_NZ_spec));
  SUNAssertNull(pm->pm_NZ_spec, SUN_ERR_MALLOC_FAIL);

  pm->pm_N_diff = st->st_N - st->st_M;
  pm->pm_dvars  = malloc(pm->pm_N_diff * sizeof(*pm->pm_dvars));
  SUNAssertNull(pm->pm_dvars, SUN_ERR_MALLOC_FAIL);

  pm->pm_known = malloc(K * sizeof(*pm->pm_known));
  SUNAssertNull(pm->pm_known, SUN_ERR_MALLOC_FAIL);

  pm->pm_vars = malloc(K * sizeof(*pm->pm_vars));
  SUNAssertNull(pm->pm_vars, SUN_ERR_MALLOC_FAIL);

  pm->pm_Jk = malloc(K * sizeof(DDMatrix*));
  SUNAssertNull(pm->pm_Jk, SUN_ERR_MALLOC_FAIL);

  pm->pm_wss = malloc(K * sizeof(DDMatrixWorkspace*));
  SUNAssertNull(pm->pm_wss, SUN_ERR_MALLOC_FAIL);

  pm->pm_knowndata = calloc(K * N, sizeof(*pm->pm_knowndata));
  SUNAssertNull(pm->pm_knowndata, SUN_ERR_MALLOC_FAIL);

  pm->pm_varsdata = calloc(st->st_N, sizeof(*pm->pm_varsdata));
  SUNAssertNull(pm->pm_varsdata, SUN_ERR_MALLOC_FAIL);

  sunindextype ofs = 0;
  for (uint8_t k = 0; k < K; ++k)
  {
    const sunindextype Mk = st->st_Mk[k];
    const sunindextype Nk = st->st_Nk[k];
    const sunindextype* I = st->st_eqns[k];
    const sunindextype* J = st->st_vars[k];

    SUNAssertNull(Nk > 0, SUN_ERR_OP_FAIL);

    if (Mk == 0)
    {
      pm->pm_Jk[k] = NULL;
      pm->pm_wss[k]  = NULL;
    }
    else
    {
      DDMatrix* A_sub = DDMatCloneSub(A, Mk, I, Nk, J);
      SUNCheckLastErrNull();
      pm->pm_Jk[k] = A_sub;

      DDMatrixWorkspace* ws = DDMatCreateWS(A_sub);
      SUNCheckLastErrNull();
      pm->pm_wss[k] = ws;
    }

    pm->pm_known[k] = pm->pm_knowndata + k * N;
    pm->pm_vars[k]  = pm->pm_varsdata + ofs;
    ofs += Nk;
  }

  return pm;
}

void PMDestroy(PivMem* pm)
{
  if (pm == NULL) { return; }

  uint8_t K = pm->pm_K;

  if (pm->pm_spec != NULL)
  {
    free(pm->pm_spec);
    pm->pm_spec = NULL;
  }

  if (pm->pm_NZ_spec != NULL)
  {
    free(pm->pm_NZ_spec);
    pm->pm_NZ_spec = NULL;
  }

  if (pm->pm_dvars != NULL)
  {
    free(pm->pm_dvars);
    pm->pm_dvars = NULL;
  }

  if (pm->pm_known != NULL)
  {
    for (size_t k = 0; k < K; ++k) { pm->pm_known[k] = NULL; }
    free(pm->pm_known);
    pm->pm_known = NULL;
  }

  if (pm->pm_vars != NULL)
  {
    for (size_t k = 0; k < K; ++k) { pm->pm_vars[k] = NULL; }
    free(pm->pm_vars);
    pm->pm_vars = NULL;
  }

  if (pm->pm_Jk != NULL)
  {
    for (size_t k = 0; k < K; ++k)
    {
      DDMatrix* submat = pm->pm_Jk[k];
      if (submat)
      {
        SUNMatDestroy(DDMatGetSUNMat(submat));
        DDMatDestroy(submat);
      }
      pm->pm_Jk[k] = NULL;
    }
    free(pm->pm_Jk);
    pm->pm_Jk = NULL;
  }

  if (pm->pm_wss != NULL)
  {
    for (size_t k = 0; k < K; ++k)
    {
      DDMatWSDestroy(pm->pm_wss[k]);
      pm->pm_wss[k] = NULL;
    }
    free(pm->pm_wss);
    pm->pm_wss = NULL;
  }

  if (pm->pm_knowndata != NULL)
  {
    free(pm->pm_knowndata);
    pm->pm_knowndata = NULL;
  }

  if (pm->pm_varsdata != NULL)
  {
    free(pm->pm_varsdata);
    pm->pm_varsdata = NULL;
  }

  free(pm);
}

static SUNErrCode PDReset(const PivMem pm[static 1])
{
  const sunindextype N = pm->pm_DAE_N;
  const uint8_t K      = pm->pm_K;

  memset(pm->pm_spec, 0, N * sizeof(*pm->pm_spec));
  memset(pm->pm_knowndata, SUNFALSE, N * K * sizeof(*pm->pm_knowndata));

  return SUN_SUCCESS;
}

SUNErrCode PPivot(const Structure st[static 1], const DDMatrix A[static 1],
                  sunrealtype tol, const PivMem pm[static 1])
{
  SUNFunctionBegin(pm->sunctx);

  SUNCheckCall(PDReset(pm));

  for (uint8_t k = 0; k < st->st_K; ++k)
  {
    const sunindextype Mk = st->st_Mk[k];
    const sunindextype Nk = st->st_Nk[k];

    SUNAssert(Nk > 0, SUN_ERR_OP_FAIL);

    const sunindextype* vars = st->st_vars[k];
    sunindextype* pm_vars    = pm->pm_vars[k];
    memcpy(pm_vars, vars, Nk * sizeof(sunindextype));

    if (Mk == 0) { continue; }

    const sunindextype* eqns = st->st_eqns[k];

    const int dof = (int)Nk - (int)Mk;

    const uint8_t* varofs = st->st_varofs;

    sunbooleantype* pm_known = pm->pm_known[k];
    if (dof == 0)
    {
      for (sunindextype l = 0; l < Nk; ++l) { pm_known[l] = SUNTRUE; }
    }
    else
    {
      for (sunindextype l = 0; l < Nk; ++l)
      {
        if (varofs[vars[l]] == 0) { pm_known[vars[l]] = SUNTRUE; }
      }

      const DDMatrix* submat = pm->pm_Jk[k];
      SUNCheckCall(DDCopySub(A, submat, Mk, eqns, Nk, vars));

      const DDMatrixWorkspace* ws = pm->pm_wss[k];
      SUNCheckCall(DDMatPivot(submat, ws, tol, Nk, pm_vars));

      for (sunindextype l = 0; l < Mk; ++l) { pm_known[pm_vars[l]] = SUNTRUE; }
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

static sunindextype ComputeNZSpec(sunindextype N, const uint8_t spec[N],
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

static SUNErrCode ComputeDiffVars(const Structure st[static 1],
                                  PivMem pm[static 1])
{
  SUNFunctionBegin(pm->sunctx);

  sunindextype ofs = 0;
  for (sunindextype i = 0; i < pm->pm_NNZ_spec; ++i)
  {
    const sunindextype var = pm->pm_NZ_spec[i], varofs = st->st_acc_varofs[var];
    const uint8_t dd = pm->pm_spec[var];
    for (uint8_t j = 0; j < dd; ++j)
    {
      SUNAssert(ofs < pm->pm_N_diff, SUN_ERR_ARG_OUTOFRANGE);
      pm->pm_dvars[ofs] = varofs + j;
      ofs++;
    }
  }

  return SUN_SUCCESS;
}

SUNErrCode PPComputeDDSpec(const Structure st[static 1], PivMem pm[static 1])
{
  SUNFunctionBegin(pm->sunctx);

  for (uint8_t k = 0; k < st->st_K; ++k)
  {
    for (sunindextype l = 0; l < st->st_Nk[k]; ++l)
    {
      const sunindextype j = pm->pm_vars[k][l];
      if (!pm->pm_known[k][j])
      {
        SUNAssert(k <= 0 || !pm->pm_known[k - 1][j], SUN_ERR_OP_FAIL);
        pm->pm_spec[j]++;
      }
    }
  }

  pm->pm_NNZ_spec = ComputeNZSpec(st->st_DAE_N, pm->pm_spec, pm->pm_NZ_spec);
  SUNCheckCall(ComputeDiffVars(st, pm));

  return SUN_SUCCESS;
}

SUNErrCode PPUpdateDDSpec(const Structure st[static 1],
                          const uint8_t spec[static 1], PivMem pm[static 1])
{
  SUNFunctionBegin(pm->sunctx);

  memcpy(pm->pm_spec, spec, pm->pm_DAE_N * sizeof(*spec));
  pm->pm_NNZ_spec = ComputeNZSpec(pm->pm_DAE_N, pm->pm_spec, pm->pm_NZ_spec);
  SUNCheckCall(ComputeDiffVars(st, pm));

  return SUN_SUCCESS;
}
