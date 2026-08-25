#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <sundials/priv/sundials_errors_impl.h>
#include <sundials/sundials_matrix.h>
#include <sunmatrix/sunmatrix_dense.h>

#include "dd_err.h"
#include "matrix.h"
#include "pivot.h"
#include "static_info.h"
#include "sundials/sundials_errors.h"
#include "sundials/sundials_types.h"

PIVMem PIVCreate(SUNContext sunctx, DDStaticInfo si, PIVMatrix J_0, PIVJacFn0 jacfn0)
{
  SUNFunctionBegin(sunctx);

  SUNAssertNull(jacfn0, SUN_ERR_ARG_CORRUPT);

  PIVMem pm = calloc(1, sizeof(*pm));
  SUNAssertNull(pm, SUN_ERR_MALLOC_FAIL);

  pm->sunctx = sunctx;
  pm->si     = si;
  pm->jacfn0 = jacfn0;
  pm->J_0    = J_0;

  const sunindextype si_N = si->N;
  const uint8_t K         = si->K;

  pm->N = si_N;

  pm->K = K;

  pm->spec = calloc(si_N, sizeof(*pm->spec));
  SUNAssertNull(pm->spec, SUN_ERR_MALLOC_FAIL);

  pm->old_spec = calloc(si_N, sizeof(*pm->old_spec));
  SUNAssertNull(pm->old_spec, SUN_ERR_MALLOC_FAIL);

  pm->known_k = malloc(K * sizeof(*pm->known_k));
  SUNAssertNull(pm->known_k, SUN_ERR_MALLOC_FAIL);

  pm->vars_k = malloc(K * sizeof(*pm->vars_k));
  SUNAssertNull(pm->vars_k, SUN_ERR_MALLOC_FAIL);

  pm->J_k = malloc(K * sizeof(PIVMatrix*));
  SUNAssertNull(pm->J_k, SUN_ERR_MALLOC_FAIL);

  pm->wss = malloc(K * sizeof(PIVMatrixWorkspace*));
  SUNAssertNull(pm->wss, SUN_ERR_MALLOC_FAIL);

  pm->known_k_flat = calloc(K * si_N, sizeof(*pm->known_k_flat));
  SUNAssertNull(pm->known_k_flat, SUN_ERR_MALLOC_FAIL);

  pm->vars_k_flat = calloc(si->N_all_orders, sizeof(*pm->vars_k_flat));
  SUNAssertNull(pm->vars_k_flat, SUN_ERR_MALLOC_FAIL);

  sunindextype ofs = 0;
  for (uint8_t k = 0; k < K; ++k)
  {
    const sunindextype M  = si->M_k[k];
    const sunindextype N  = si->N_k[k];
    const sunindextype* I = si->eqns_k[k];
    const sunindextype* J = si->vars_k[k];

    SUNAssertNull(N > 0, SUN_ERR_OP_FAIL);

    if (M == 0)
    {
      pm->J_k[k] = NULL;
      pm->wss[k] = NULL;
    }
    else
    {
      PIVMatrix A_sub = PIVMatCloneSub(J_0, M, I, N, J);
      SUNCheckLastErrNull();
      pm->J_k[k] = A_sub;

      PIVMatrixWorkspace ws = PIVMatCreateWS(A_sub);
      SUNCheckLastErrNull();
      pm->wss[k] = ws;
    }

    pm->known_k[k] = pm->known_k_flat + k * si_N;
    pm->vars_k[k]  = pm->vars_k_flat + ofs;
    ofs += N;
  }

  return pm;
}

SUNErrCode PIVSetUserData(PIVMem pm, void* user_data)
{
  if (pm == NULL)
  {
    DDHandleErrWithCtx(SUN_ERR_CORRUPT, NULL);
    return SUN_ERR_CORRUPT;
  }

  pm->user_data = user_data;

  return SUN_SUCCESS;
}

void PIVDestroy(PIVMem* pm_ptr)
{
  if (pm_ptr == NULL || *pm_ptr == NULL) { return; }

  PIVMem pm = *pm_ptr;

  uint8_t K = pm->K;

  free(pm->spec);
  free(pm->old_spec);
  free(pm->known_k);
  free(pm->vars_k);

  if (pm->J_k != NULL)
  {
    for (size_t k = 0; k < K; ++k)
    {
      PIVMatrix submat = pm->J_k[k];
      if (submat)
      {
        SUNMatDestroy(PIVMatGetSUNMat(submat));
        PIVMatDestroy(submat);
      }
    }
    free(pm->J_k);
  }

  if (pm->wss != NULL)
  {
    for (size_t k = 0; k < K; ++k) { PIVMatWSDestroy(pm->wss[k]); }
    free(pm->wss);
  }

  free(pm->known_k_flat);
  free(pm->vars_k_flat);
  free(pm);

  *pm_ptr = NULL;
}

void PIVPrint(PIVMem pm, FILE* file)
{
  DDStaticInfo si = pm->si;

  fprintf(file, "--- START PIVOTDATA ----\n");
  for (uint8_t k = 0; k < si->K; ++k)
  {
    fprintf(file, "Stage k = %d:\n", DDSI_STAGE_FROM_INDEX(si, k));
    for (sunindextype n = 0; n < si->N_k[k]; ++n)
    {
      const sunindextype j = si->vars_k[k][n];
      fprintf(file,
              "\t[%ld]%s^(%d)\t%s\n",
              j,
              DDSI_VAR_NAME(si, j),
              DDSI_VAR_ORDER(si, k, j),
              pm->known_k[k][j] ? "true" : "false");
    }
  }
  fprintf(file, "--- END PIVOTDATA ------\n");
}

static void PDReset(PIVMem pm)
{
  const sunindextype N = pm->N;
  const uint8_t K      = pm->K;

  memset(pm->spec, 0, N * sizeof(*pm->spec));
  memset(pm->known_k_flat, SUNFALSE, N * K * sizeof(*pm->known_k_flat));
}

SUNErrCode PIVPivot(PIVMem pm,
                    sunrealtype tol,
                    sunrealtype t,
                    N_Vector Y,
                    sunbooleantype* spec_changed)
{
  SUNFunctionBegin(pm->sunctx);

  DDStaticInfo si = pm->si;

  memcpy(pm->old_spec, pm->spec, pm->N * sizeof(*pm->old_spec));

  PDReset(pm);

  SUNAssert(pm->jacfn0(t, Y, PIVMatGetSUNMat(pm->J_0), pm->user_data) >= 0,
            SUN_ERR_OP_FAIL);

  for (uint8_t k = 0; k < si->K; ++k)
  {
    const sunindextype M = si->M_k[k];
    const sunindextype N = si->N_k[k];

    SUNAssert(N > 0, SUN_ERR_OP_FAIL);

    if (M == 0) { continue; }

    const sunindextype* vars = si->vars_k[k];
    const sunindextype* eqns = si->eqns_k[k];
    sunbooleantype* pm_known = pm->known_k[k];

    sunindextype m = 0;
    if (N == M)
    {
      /* If the sub-matrix of the Jacobian is square all variables must be known
         under the assumption that the Jacobian is regular. */

      for (sunindextype n = 0; n < N; ++n)
      {
        const sunindextype j = vars[n];
        pm_known[j]          = SUNTRUE;
      }
      m = M;
    }
    else
    {
      SUNAssert(k != si->K - 1 && "J0 should always be square",
                SUN_ERR_OUTOFRANGE);

      /* Some variables may be marked as known from previous stages. */
      for (sunindextype n = 0; n < N; ++n)
      {
        const sunindextype j = vars[n];
        if (pm_known[j]) { m += 1; }
      }

      if (m != M)
      {
        /* If we don't already have the required number of known variables we
           need to select columns that forms a regular square sub-matrix. */

        sunindextype* pm_vars = pm->vars_k[k];
        memcpy(pm_vars, vars, N * sizeof(*pm_vars));

        PIVMatrix J_k = pm->J_k[k];
        SUNCheckCall(PIVCopySub(pm->J_0, J_k, eqns, pm_vars));

        const PIVMatrixWorkspace ws = pm->wss[k];
        SUNCheckCall(PIVMatPivot(J_k, ws, tol, N, pm_vars));

        for (sunindextype n = 0; n < N; ++n)
        {
          const sunindextype j = pm_vars[n];
          if (!pm_known[j])
          {
            pm_known[j] = SUNTRUE;
            m += 1;
            if (m == M) { break; }
          }
        }

        /* Because we have marked new variables as known and the derivative of a
           known variable must also be known, proceed to mark these before the
           next stage. */

        for (uint8_t kk = k + 1; kk < si->K; ++kk)
        {
          for (sunindextype n = 0; n < N; ++n)
          {
            const sunindextype j = vars[n];
            if (pm_known[j]) { pm->known_k[kk][j] = SUNTRUE; }
          }
        }
      }
    }

    SUNAssert(m == M && "Unable to find the required number of known variables "
                        "for this stage",
              SUN_ERR_OP_FAIL);
  }

  /* Compute new spec from known_k. */
  for (uint8_t k = 0; k < si->K; ++k)
  {
    for (sunindextype n = 0; n < si->N_k[k]; ++n)
    {
      const sunindextype* vars = si->vars_k[k];
      const sunindextype j     = vars[n];
      if (!pm->known_k[k][j])
      {
        SUNAssert(k <= 0 || !pm->known_k[k - 1][j], SUN_ERR_OP_FAIL);
        pm->spec[j]++;
      }
    }
  }

  *spec_changed =
    memcmp(pm->old_spec, pm->spec, pm->N * sizeof(*pm->old_spec)) != 0;

  return SUN_SUCCESS;
}

/* void PSPrintSubmat(const Structure si[static 1], const PivMem pm[static 1], */
/*                    uint8_t k, FILE* file) */
/* { */
/*   for (sunindextype i = 0; i < si->st_Nk[k]; ++i) */
/*   { */
/*     sunindextype j = pm->pm_vars[k][i]; */
/*     fprintf(file, "\td%d%s", DDSI_VAR_ORDER(si, k, j), DDSI_VAR_NAME(si, j)); */
/*   } */

/*   SUNMatrix mat = DDMatGetSUNMat(pm->pm_jacs[k]); */
/*   if (SUNMatGetID(mat) == SUNMATRIX_DENSE) { SUNDenseMatrix_Print(mat, file); } */
/* } */
