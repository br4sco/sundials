#include <assert.h>
#include <sundials/priv/sundials_errors_impl.h>
#include <sundials/sundials_errors.h>
#include <sundials/sundials_macros.h>
#include <sundials/sundials_matrix.h>
#include <sunmatrix/sunmatrix_dense.h>

#include "macros.h"
#include "matrix.h"
#include "sundials/sundials_types.h"

/* ==========================================================================
 * Generic DD Matrix
 * ========================================================================== */

void DDMatDestroy(DDMatrix self)
{
  if (self == NULL) { return; }

  self->A   = NULL;
  self->ops = NULL;
  free(self);
}

void DDMatWSDestroy(DDMatrixWorkspace self)
{
  if (self == NULL) { return; }

  if (self->destroy != NULL)
  {
    self->destroy(self);
    self->destroy = NULL;
  }

  if (self->content != NULL)
  {
    free(self->content);
    self->content = NULL;
  }

  free(self);
}

/* ==========================================================================
 * Dense DD Matrix
 * ========================================================================== */

static void DDMatWSContentDestroy_Dense(SUNDIALS_MAYBE_UNUSED DDMatrixWorkspace ws)
{
  return;
}

static DDMatrixWorkspace DDMatCreateWS_Dense(DDMatrix self)
{
  SUNMatrix A = DDMatGetSUNMat(self);
  SUNFunctionBegin(A->sunctx);

  SUNAssertNull(SUNMatGetID(A) == SUNMATRIX_DENSE, SUN_ERR_ARG_WRONGTYPE);

  DDMatrixWorkspace ws = malloc(sizeof(*ws));
  SUNAssertNull(ws, SUN_ERR_MALLOC_FAIL);

  ws->id      = DDMATRIXWS_ROWPIVOT;
  ws->destroy = DDMatWSContentDestroy_Dense;

  ws->content = malloc(SM_ROWS_D(A) * sizeof(sunindextype));
  SUNAssertNull(ws->content, SUN_ERR_MALLOC_FAIL);

  return ws;
}

SUNErrCode DDMatPivot_Dense(
  DDMatrix self,
  DDMatrixWorkspace ws,
  sunrealtype tol,
  sunindextype n,
  sunindextype colpivots[static n]
)
{
  SUNMatrix sm_self = DDMatGetSUNMat(self);
  SUNFunctionBegin(sm_self->sunctx);

  SUNAssert(SUNMatGetID(sm_self) == SUNMATRIX_DENSE, SUN_ERR_ARG_WRONGTYPE);

  SUNCheck(0 <= n && n <= SM_COLUMNS_D(sm_self), SUN_ERR_ARG_OUTOFRANGE);

  sunindextype* c      = colpivots;
  sunindextype* r      = (sunindextype*)ws->content;
  const sunindextype m = SM_ROWS_D(sm_self);

  for (sunindextype i = 0; i < m; ++i) { r[i] = i; }

  for (sunindextype j = 0; j < n; ++j) { c[j] = j; }

  for (sunindextype k = 0; k < SUNMIN(m, n); ++k)
  {
    /* Get current pivot value. */
    const sunrealtype pivotabsval = SUNRabs(SM_ELEMENT_D(sm_self, r[k], c[k]));

    /* Find the maximum value in the part of the matrix which we have
           not yet considered. */
    sunindextype max_i     = -1;
    sunindextype max_j     = -1;
    sunrealtype max_absval = SUNRabs(tol);
    for (sunindextype i = k; i < m; ++i)
    {
      for (sunindextype j = k; j < n; ++j)
      {
        sunrealtype tmp = SUNRabs(SM_ELEMENT_D(sm_self, r[i], c[j]));
        if (tmp > max_absval)
        {
          max_absval = tmp;
          max_i      = i;
          max_j      = j;
        }
      }
    }

    /* The matrix is singular. */
    SUNCheck(max_i != -1 && max_j != -1, SUN_ERR_OP_FAIL);

    /* Swap rows and columns so that the current maximum value (up to a
           factor) always appears in the pivot position. */
    if (max_absval > DD_PIVOT_SCALE * pivotabsval)
    {
      sunindextype tmp = r[k];
      r[k]             = r[max_i];
      r[max_i]         = tmp;
      tmp              = c[k];
      c[k]             = c[max_j];
      c[max_j]         = tmp;
    }

    /* Get the signed, non-zero, pivot value after any potential
         * swapping. */
    const sunrealtype pivotval = SM_ELEMENT_D(sm_self, r[k], c[k]);
    SUNAssert(SUNRabs(pivotval) >= tol, SUN_ERR_OP_FAIL);

    /* Perform one step of Gaussian elimination on the remaining rows.
         */
    for (sunindextype i = k + 1; i < m; ++i)
    {
      const sunrealtype leadval = SM_ELEMENT_D(sm_self, r[i], c[k]);
      if (leadval != SUN_RCONST(0.0))
      {
        const sunrealtype scaleval        = leadval / pivotval;
        SM_ELEMENT_D(sm_self, r[i], c[k]) = SUN_RCONST(0.0);
        for (sunindextype j = k + 1; j < n; ++j)
        {
          SM_ELEMENT_D(sm_self, r[i], c[j]) -= scaleval *
                                               SM_ELEMENT_D(sm_self, r[k], c[j]);
        }
      }
    }
  }

  return SUN_SUCCESS;
}

static DDMatrix DDMatCloneSub_Dense(
  DDMatrix self,
  sunindextype m,
  const sunindextype rows[static m],
  sunindextype n,
  const sunindextype cols[static n]
)
{
  SUNMatrix sm_self = DDMatGetSUNMat(self);
  SUNFunctionBegin(sm_self->sunctx);

  SUNAssertNull(SUNMatGetID(sm_self) == SUNMATRIX_DENSE, SUN_ERR_ARG_WRONGTYPE);

  SUNCheckNull(0 < m && m <= SM_ROWS_D(sm_self), SUN_ERR_ARG_OUTOFRANGE);
  SUNCheckNull(0 < n && n <= SM_COLUMNS_D(sm_self), SUN_ERR_ARG_OUTOFRANGE);

  SUNMatrix A_new = SUNDenseMatrix(m, n, sm_self->sunctx);
  if (A_new == NULL) { return NULL; }

  for (sunindextype j = 0; j < n; ++j)
  {
    SUNCheckNull(cols[j] >= 0 && cols[j] < SM_COLUMNS_D(sm_self), SUN_ERR_ARG_DIMSMISMATCH);
    for (sunindextype i = 0; i < m; ++i)
    {
      SUNCheckNull(rows[i] >= 0 && rows[i] < SM_ROWS_D(sm_self), SUN_ERR_ARG_DIMSMISMATCH);
      SM_ELEMENT_D(A_new, i, j) = SM_ELEMENT_D(sm_self, rows[i], cols[j]);
    }
  }

  return DDMatWrapDense(A_new);
}

static SUNErrCode DDMatCopySub_Dense(
  DDMatrix self,
  DDMatrix A,
  sunindextype m,
  const sunindextype rows[static m],
  sunindextype n,
  const sunindextype cols[static n]
)
{
  SUNMatrix sm_self = DDMatGetSUNMat(self);
  SUNMatrix sm_A    = DDMatGetSUNMat(A);
  SUNFunctionBegin(sm_self->sunctx);

  SUNAssert(SUNMatGetID(sm_self) == SUNMATRIX_DENSE, SUN_ERR_ARG_WRONGTYPE);
  SUNAssert(SUNMatGetID(sm_A) == SUNMATRIX_DENSE, SUN_ERR_ARG_WRONGTYPE);

  SUNCheck(0 <= m && m <= SM_ROWS_D(sm_A), SUN_ERR_ARG_OUTOFRANGE);
  SUNCheck(0 <= n && n <= SM_COLUMNS_D(sm_A), SUN_ERR_ARG_OUTOFRANGE);

  for (sunindextype j = 0; j < n; ++j)
  {
    SUNCheck(cols[j] >= 0 && cols[j] < SM_COLUMNS_D(sm_self), SUN_ERR_ARG_DIMSMISMATCH);
    for (sunindextype i = 0; i < m; ++i)
    {
      SUNCheck(rows[i] >= 0 && rows[i] < SM_ROWS_D(sm_self), SUN_ERR_ARG_DIMSMISMATCH);
      SM_ELEMENT_D(sm_A, i, j) = SM_ELEMENT_D(sm_self, rows[i], cols[j]);
    }
  }

  return SUN_SUCCESS;
}

static const DDMatrix_Ops DDMatrix_Ops_Dense = {
  .createworkspace = DDMatCreateWS_Dense,
  .pivot           = DDMatPivot_Dense,
  .clonesub        = DDMatCloneSub_Dense,
  .copysub         = DDMatCopySub_Dense,
};

DDMatrix DDMatWrapDense(SUNMatrix A)
{
  SUNFunctionBegin(A->sunctx);

  SUNAssertNull(SUNMatGetID(A) == SUNMATRIX_DENSE, SUN_ERR_ARG_WRONGTYPE);

  DDMatrix B = malloc(sizeof(*B));
  SUNCheckNull(B, SUN_ERR_MALLOC_FAIL);

  B->A   = A;
  B->ops = &DDMatrix_Ops_Dense;

  return B;
}

/* ==========================================================================
 * Sparse DD Matrix
 * ========================================================================== */

static const DDMatrix_Ops DDMatrix_Ops_Sparse = {0};

DDMatrix DDMatWrapSparse(SUNMatrix A)
{
  SUNFunctionBegin(A->sunctx);

  SUNAssertNull(SUNMatGetID(A) == SUNMATRIX_SPARSE, SUN_ERR_ARG_WRONGTYPE);

  DDMatrix B = malloc(sizeof(*B));
  SUNCheckNull(B, SUN_ERR_MALLOC_FAIL);

  B->A   = A;
  B->ops = &DDMatrix_Ops_Sparse;

  return B;
}
