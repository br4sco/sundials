#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <sundials/sundials_matrix.h>
#include <sunmatrix/sunmatrix_dense.h>

#include "matrix.h"
#include "pivot.h"
#include "structure.h"
#include "sundials/sundials_types.h"

PivMem* PMCreate(const Structure st[static 1], const ExtSUNMatrix mat[static 1])
{
  PivMem* pm = malloc(sizeof(*pm));
  if (pm == NULL) return NULL;

  const sunindextype N = st->st_DAE_N;
  const uint8_t K      = st->st_K;

  pm->pm_DAE_N = N;
  pm->pm_K     = K;

  pm->pm_spec = calloc(N, sizeof(uint8_t));
  if (pm->pm_spec == NULL)
  {
    PMDestroy(pm);
    return NULL;
  }

  pm->pm_NNZ_spec = 0;
  pm->pm_NZ_spec  = malloc(N * sizeof(sunindextype));
  if (pm->pm_NZ_spec == NULL)
  {
    PMDestroy(pm);
    return NULL;
  }

  pm->pm_known = malloc(K * sizeof(sunbooleantype*));
  if (pm->pm_known == NULL)
  {
    PMDestroy(pm);
    return NULL;
  }

  pm->pm_vars = malloc(K * sizeof(sunindextype*));
  if (pm->pm_vars == NULL)
  {
    PMDestroy(pm);
    return NULL;
  }

  pm->pm_jacs = malloc(K * sizeof(ExtSUNMatrix*));
  if (pm->pm_jacs == NULL)
  {
    PMDestroy(pm);
    return NULL;
  }

  pm->pm_wss = malloc(K * sizeof(ExtSUNMatrixWS*));
  if (pm->pm_wss == NULL)
  {
    PMDestroy(pm);
    return NULL;
  }

  pm->pm_knowndata = calloc(K * N, sizeof(sunbooleantype));
  if (pm->pm_knowndata == NULL)
  {
    PMDestroy(pm);
    return NULL;
  }

  pm->pm_varsdata = calloc(st->st_N, sizeof(sunindextype));
  if (pm->pm_varsdata == NULL)
  {
    PMDestroy(pm);
    return NULL;
  }

  sunindextype ofs = 0;
  for (uint8_t k = 0; k < K; ++k)
  {
    const sunindextype Mk = st->st_Mk[k];
    const sunindextype Nk = st->st_Nk[k];
    const sunindextype* I = st->st_eqns[k];
    const sunindextype* J = st->st_vars[k];

    assert(Nk > 0);
    if (Mk == 0)
    {
      pm->pm_jacs[k] = NULL;
      pm->pm_wss[k]  = NULL;
    }
    else
    {
      ExtSUNMatrix* submat = ExtSUNMatCloneSub(mat, Mk, I, Nk, J);
      if (submat == NULL)
      {
        PMDestroy(pm);
        return NULL;
      }
      pm->pm_jacs[k] = submat;

      ExtSUNMatrixWS* ws = ExtSUNMatCreateWS(submat);
      if (ws == NULL)
      {
        PMDestroy(pm);
        return NULL;
      }
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

  if (pm->pm_jacs != NULL)
  {
    for (size_t k = 0; k < K; ++k)
    {
      ExtSUNMatrix* submat = pm->pm_jacs[k];
      if (submat)
      {
        SUNMatDestroy(ExtSUNMatGetMat(submat));
        ExtSUNMatDestroy(submat);
      }
      pm->pm_jacs[k] = NULL;
    }
    free(pm->pm_jacs);
    pm->pm_jacs = NULL;
  }

  if (pm->pm_wss != NULL)
  {
    for (size_t k = 0; k < K; ++k)
    {
      ExtSUNMatWSDestroy(pm->pm_wss[k]);
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

static sunbooleantype PDReset(const PivMem pm[static 1])
{
  const sunindextype N = pm->pm_DAE_N;
  const uint8_t K      = pm->pm_K;

  memset(pm->pm_spec, 0, N * sizeof(*pm->pm_spec));
  memset(pm->pm_knowndata, SUNFALSE, N * K * sizeof(*pm->pm_knowndata));

  return SUNTRUE;
}

sunbooleantype PPivot(const Structure st[static 1],
                      const ExtSUNMatrix mat[static 1], sunrealtype tol,
                      const PivMem pm[static 1])
{
  PDReset(pm);

  for (uint8_t k = 0; k < st->st_K; ++k)
  {
    const sunindextype Mk = st->st_Mk[k];
    const sunindextype Nk = st->st_Nk[k];

    assert(Nk > 0);

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

      const ExtSUNMatrix* submat = pm->pm_jacs[k];
      if (!ExtSUNMatCopySub(mat, submat, Mk, eqns, Nk, vars))
      {
        return SUNFALSE;
      }

      const ExtSUNMatrixWS* ws = pm->pm_wss[k];
      if (!ExtSUNMatPivot(submat, ws, tol, Nk, pm_vars)) { return SUNFALSE; }

      for (sunindextype l = 0; l < Mk; ++l) { pm_known[pm_vars[l]] = SUNTRUE; }
    }
  }

  return SUNTRUE;
}

void PSPrintSubmat(const Structure st[static 1], const PivMem pm[static 1],
                   uint8_t k, FILE* file)
{
  for (sunindextype i = 0; i < st->st_Nk[k]; ++i)
  {
    sunindextype j = pm->pm_vars[k][i];
    fprintf(file, "\td%d%s", ST_VAR_ORDER(st, k, j), ST_VAR_NAME(st, j));
  }

  SUNMatrix mat = ExtSUNMatGetMat(pm->pm_jacs[k]);
  if (SUNMatGetID(mat) == SUNMATRIX_DENSE) { SUNDenseMatrix_Print(mat, file); }
}

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

sunbooleantype PPComputeDDSpec(const Structure st[static 1], PivMem pm[static 1])
{
  for (uint8_t k = 0; k < st->st_K; ++k)
  {
    for (sunindextype l = 0; l < st->st_Nk[k]; ++l)
    {
      const sunindextype j = pm->pm_vars[k][l];
      if (!pm->pm_known[k][j])
      {
        if (k > 0 && pm->pm_known[k - 1][j]) { return SUNFALSE; }

        pm->pm_spec[j]++;
      }
    }
  }

  pm->pm_NNZ_spec = ComputeNZSpec(st->st_DAE_N, pm->pm_spec, pm->pm_NZ_spec);

  return SUNTRUE;
}

sunbooleantype PPUpdateDDSpec(const uint8_t spec[static 1], PivMem pm[static 1])
{
  memcpy(pm->pm_spec, spec, pm->pm_DAE_N * sizeof(uint8_t));
  pm->pm_NNZ_spec = ComputeNZSpec(pm->pm_DAE_N, pm->pm_spec, pm->pm_NZ_spec);

  return SUNTRUE;
}
