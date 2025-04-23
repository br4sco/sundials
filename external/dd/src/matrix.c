#include <assert.h>
#include <sundials/priv/sundials_errors_impl.h>
#include <sundials/sundials_errors.h>
#include <sundials/sundials_macros.h>
#include <sundials/sundials_matrix.h>
#include <sunmatrix/sunmatrix_dense.h>

#include "macros.h"
#include "matrix.h"

/* ==========================================================================
 * Generic Extended Sundials Matrix
 * ========================================================================== */

void DDMatDestroy(DDMatrix* self)
{
  if (self == NULL) { return; }

  self->A   = NULL;
  self->ops = NULL;
  free(self);
}

void DDMatWSDestroy(DDMatrixWorkspace* self)
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
 * Dense Extended Sundials Matrix
 * ========================================================================== */

static void DDMatWSContentDestroy_Dense(SUNDIALS_MAYBE_UNUSED DDMatrixWorkspace* ws)
{
  return;
}

static DDMatrixWorkspace* DDMatCreateWS_Dense(const DDMatrix self[static 1])
{
  SUNMatrix A = DDMatGetSUNMat(self);
  SUNFunctionBegin(A->sunctx);

  SUNAssertNull(SUNMatGetID(A) == SUNMATRIX_DENSE, SUN_ERR_ARG_WRONGTYPE);

  DDMatrixWorkspace* ws = malloc(sizeof(*ws));
  SUNAssertNull(ws, SUN_ERR_MALLOC_FAIL);

  ws->id      = DDMATRIXWS_ROWPIVOT;
  ws->destroy = DDMatWSContentDestroy_Dense;

  ws->content = malloc(SM_ROWS_D(A) * sizeof(sunindextype));
  SUNAssertNull(ws->content, SUN_ERR_MALLOC_FAIL);

  return ws;
}

SUNErrCode DDMatPivot_Dense(const DDMatrix self[static 1],
                            const DDMatrixWorkspace ws[static 1], sunrealtype tol,
                            sunindextype n, sunindextype colpivots[static n])
{
  SUNMatrix A = DDMatGetSUNMat(self);
  SUNFunctionBegin(A->sunctx);

  SUNAssert(SUNMatGetID(A) == SUNMATRIX_DENSE, SUN_ERR_ARG_WRONGTYPE);
  SUNCheck(0 <= n && n <= SM_COLUMNS_D(A), SUN_ERR_ARG_OUTOFRANGE);

  sunindextype* c      = colpivots;
  sunindextype* r      = (sunindextype*)ws->content;
  const sunindextype m = SM_ROWS_D(A);

  for (sunindextype i = 0; i < m; ++i) { r[i] = i; }

  for (sunindextype j = 0; j < n; ++j) { c[j] = j; }

  for (sunindextype k = 0; k < SUNMIN(m, n); ++k)
  {
    /* Get current pivot value. */
    const sunrealtype pivotabsval = SUNRabs(SM_ELEMENT_D(A, r[k], c[k]));

    /* Find the maximum value in the part of the matrix which we have
           not yet considered. */
    sunindextype max_i     = -1;
    sunindextype max_j     = -1;
    sunrealtype max_absval = SUNRabs(tol);
    for (sunindextype i = k; i < m; ++i)
    {
      for (sunindextype j = k; j < n; ++j)
      {
        sunrealtype tmp = SUNRabs(SM_ELEMENT_D(A, r[i], c[j]));
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
    const sunrealtype pivotval = SM_ELEMENT_D(A, r[k], c[k]);
    SUNAssert(SUNRabs(pivotval) >= tol, SUN_ERR_OP_FAIL);

    /* Perform one step of Gaussian elimination on the remaining rows.
         */
    for (sunindextype i = k + 1; i < m; ++i)
    {
      const sunrealtype leadval = SM_ELEMENT_D(A, r[i], c[k]);
      if (leadval != SUN_RCONST(0.0))
      {
        const sunrealtype scaleval  = leadval / pivotval;
        SM_ELEMENT_D(A, r[i], c[k]) = SUN_RCONST(0.0);
        for (sunindextype j = k + 1; j < n; ++j)
        {
          SM_ELEMENT_D(A, r[i], c[j]) -= scaleval * SM_ELEMENT_D(A, r[k], c[j]);
        }
      }
    }
  }

  return SUN_SUCCESS;
}

static DDMatrix* DDMatCloneSub_Dense(const DDMatrix self[static 1],
                                     sunindextype m,
                                     const sunindextype rows[static m],
                                     sunindextype n,
                                     const sunindextype cols[static n])
{
  SUNMatrix A = DDMatGetSUNMat(self);
  assert(SUNMatGetID(A) == SUNMATRIX_DENSE);

  if (m <= 0 || m > SM_ROWS_D(A) || n <= 0 || n > SM_COLUMNS_D(A))
  {
    return NULL;
  }

  SUNMatrix A_new = SUNDenseMatrix(m, n, A->sunctx);
  if (A_new == NULL) { return NULL; }

  for (sunindextype j = 0; j < n; ++j)
  {
    assert(cols[j] >= 0 && cols[j] < SM_COLUMNS_D(A));
    for (sunindextype i = 0; i < m; ++i)
    {
      assert(rows[i] >= 0 && rows[i] < SM_ROWS_D(A));
      SM_ELEMENT_D(A_new, i, j) = SM_ELEMENT_D(A, rows[i], cols[j]);
    }
  }

  return DDMatWrapDense(A_new);
}

static sunbooleantype DDMatCopySub_Dense(const DDMatrix self[static 1],
                                         const DDMatrix A[static 1],
                                         sunindextype m,
                                         const sunindextype rows[static m],
                                         sunindextype n,
                                         const sunindextype cols[static n])
{
  SUNMatrix B = DDMatGetSUNMat(self);
  assert(SUNMatGetID(B) == SUNMATRIX_DENSE);
  SUNMatrix C = DDMatGetSUNMat(A);
  assert(SUNMatGetID(C) == SUNMATRIX_DENSE);

  if (m < 0 || m > SM_ROWS_D(C) || n < 0 || n > SM_COLUMNS_D(C))
  {
    return SUNFALSE;
  }

  for (sunindextype j = 0; j < n; ++j)
  {
    assert(cols[j] >= 0 && cols[j] < SM_COLUMNS_D(B));
    for (sunindextype i = 0; i < m; ++i)
    {
      assert(rows[i] >= 0 && rows[i] < SM_ROWS_D(B));
      SM_ELEMENT_D(C, i, j) = SM_ELEMENT_D(B, rows[i], cols[j]);
    }
  }

  return SUNTRUE;
}

static const DDMatrix_Ops extended_SUNMatrix_Ops_Dense =
  {.createworkspace = DDMatCreateWS_Dense,
   .pivot           = DDMatPivot_Dense,
   .clonesub        = DDMatCloneSub_Dense,
   .copysub         = DDMatCopySub_Dense};

DDMatrix* DDMatWrapDense(const SUNMatrix A)
{
  if (A == NULL || SUNMatGetID(A) != SUNMATRIX_DENSE) { return NULL; }

  DDMatrix* B = malloc(sizeof(*B));
  if (B == NULL) { return NULL; }

  B->A   = A;
  B->ops = &extended_SUNMatrix_Ops_Dense;

  return B;
}

/* ==========================================================================
 * Creation and Pivoting of Structured Sundials Matrices
 * ========================================================================== */

/* --------------------------------------------------------------------------
 * Sparse Matrices
 * -------------------------------------------------------------------------- */

SUNMatrix DDSparseSUNMatFromStructure(const Structure* st, sunindextype NNZ,
                                      int sparsetype, SUNContext sunctx)
{
  SUNFunctionBegin(sunctx);
  SUNAssertNull(st, SUN_ERR_ARG_CORRUPT);
  SUNAssertNull(NNZ >= 0, SUN_ERR_ARG_OUTOFRANGE);

  SUNAssertNull(st->st_N >= st->st_M, SUN_ERR_ARG_DIMSMISMATCH);
  SUNAssertNull(sparsetype == CSC_MAT || sparsetype == CSR_MAT,
                SUN_ERR_ARG_OUTOFRANGE);

  const sunindextype N = st->st_N;

  if (sparsetype == CSC_MAT)
  {
    sunindextype tmp = 0;
    for (sunindextype i = 0; i < st->st_DAE_N; ++i) { tmp += st->st_varofs[i]; }
    NNZ += 2 * tmp;
  }
  else { NNZ = NNZ + 2 * (N - st->st_M); }

  return SUNSparseMatrix(N, N, NNZ, sparsetype, sunctx);
}
