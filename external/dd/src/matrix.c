#include <assert.h>
#include <sundials/priv/sundials_errors_impl.h>
#include <sundials/sundials_errors.h>
#include <sundials/sundials_macros.h>
#include <sundials/sundials_matrix.h>
#include <sunmatrix/sunmatrix_dense.h>

#include "macros.h"
#include "matrix.h"
#include "sundials/sundials_types.h"
#include "sunmatrix/sunmatrix_sparse.h"

#define ONE SUN_RCONST(1.0);

/* ==========================================================================
 * Generic DD Matrix
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
 * Dense DD Matrix
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
                            const DDMatrixWorkspace ws[static 1],
                            sunrealtype tol,
                            sunindextype n,
                            sunindextype colpivots[static n])
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
          SM_ELEMENT_D(sm_self, r[i],
                       c[j]) -= scaleval * SM_ELEMENT_D(sm_self, r[k], c[j]);
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
  SUNMatrix sm_self = DDMatGetSUNMat(self);
  SUNFunctionBegin(sm_self->sunctx);

  SUNAssertNull(SUNMatGetID(sm_self) == SUNMATRIX_DENSE, SUN_ERR_ARG_WRONGTYPE);

  SUNCheckNull(0 < m && m <= SM_ROWS_D(sm_self), SUN_ERR_ARG_OUTOFRANGE);
  SUNCheckNull(0 < n && n <= SM_COLUMNS_D(sm_self), SUN_ERR_ARG_OUTOFRANGE);

  SUNMatrix A_new = SUNDenseMatrix(m, n, sm_self->sunctx);
  if (A_new == NULL) { return NULL; }

  for (sunindextype j = 0; j < n; ++j)
  {
    SUNCheckNull(cols[j] >= 0 && cols[j] < SM_COLUMNS_D(sm_self),
                 SUN_ERR_ARG_DIMSMISMATCH);
    for (sunindextype i = 0; i < m; ++i)
    {
      SUNCheckNull(rows[i] >= 0 && rows[i] < SM_ROWS_D(sm_self),
                   SUN_ERR_ARG_DIMSMISMATCH);
      SM_ELEMENT_D(A_new, i, j) = SM_ELEMENT_D(sm_self, rows[i], cols[j]);
    }
  }

  return DDMatWrapDense(A_new);
}

static SUNErrCode DDMatCopySub_Dense(const DDMatrix self[static 1],
                                     const DDMatrix A[static 1],
                                     sunindextype m,
                                     const sunindextype rows[static m],
                                     sunindextype n,
                                     const sunindextype cols[static n])
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
    SUNCheck(cols[j] >= 0 && cols[j] < SM_COLUMNS_D(sm_self),
             SUN_ERR_ARG_DIMSMISMATCH);
    for (sunindextype i = 0; i < m; ++i)
    {
      SUNCheck(rows[i] >= 0 && rows[i] < SM_ROWS_D(sm_self),
               SUN_ERR_ARG_DIMSMISMATCH);
      SM_ELEMENT_D(sm_A, i, j) = SM_ELEMENT_D(sm_self, rows[i], cols[j]);
    }
  }

  return SUN_SUCCESS;
}

static SUNErrCode DDMatEvalDDJac_Dense(const DDMatrix self[static 1],
                                       const Structure st[static 1],
                                       sunindextype NNZ_spec,
                                       const sunindextype NZ_spec[static NNZ_spec],
                                       const uint8_t spec[static 1],
                                       sunrealtype cj)
{
  SUNMatrix A = DDMatGetSUNMat(self);
  SUNFunctionBegin(A->sunctx);

  SUNAssert(SUNMatGetID(A) == SUNMATRIX_DENSE, SUN_ERR_ARG_WRONGTYPE);
  SUNCheck(SM_ROWS_D(A) == SM_COLUMNS_D(A), SUN_ERR_ARG_DIMSMISMATCH);
  SUNCheck(SM_ROWS_D(A) == st->st_N, SUN_ERR_ARG_DIMSMISMATCH);
  SUNCheck(st->st_M <= st->st_N, SUN_ERR_ARG_DIMSMISMATCH);

  sunindextype row = st->st_M;
  for (sunindextype i = 0; i < NNZ_spec; ++i)
  {
    const sunindextype var = NZ_spec[i], varofs = st->st_acc_varofs[var];
    const uint8_t dd = spec[var];
    for (int j = 0; j < dd; ++j)
    {
      const sunindextype col = varofs + j;

      SM_ELEMENT_D(A, row, col)     = -cj;
      SM_ELEMENT_D(A, row, col + 1) = ONE;

      row += 1;
    }
  }

  return SUN_SUCCESS;
}

static const DDMatrix_Ops DDMatrix_Ops_Dense = {.createworkspace =
                                                  DDMatCreateWS_Dense,
                                                .pivot    = DDMatPivot_Dense,
                                                .clonesub = DDMatCloneSub_Dense,
                                                .copysub  = DDMatCopySub_Dense,
                                                .evalddjac = DDMatEvalDDJac_Dense};

DDMatrix* DDMatWrapDense(const SUNMatrix A)
{
  SUNFunctionBegin(A->sunctx);

  SUNAssertNull(SUNMatGetID(A) == SUNMATRIX_DENSE, SUN_ERR_ARG_WRONGTYPE);

  DDMatrix* B = malloc(sizeof(*B));
  SUNCheckNull(B, SUN_ERR_MALLOC_FAIL);

  B->A   = A;
  B->ops = &DDMatrix_Ops_Dense;

  return B;
}

/* ==========================================================================
 * Sparse DD Matrix
 * ========================================================================== */

static SUNErrCode DDMatEvalDDJac_Sparse(const DDMatrix self[static 1],
                                        const Structure st[static 1],
                                        sunindextype NNZ_spec,
                                        const sunindextype NZ_spec[static NNZ_spec],
                                        const uint8_t spec[static 1],
                                        sunrealtype cj)
{
  SUNMatrix A = DDMatGetSUNMat(self);
  SUNFunctionBegin(A->sunctx);

  SUNAssert(SUNMatGetID(A) == SUNMATRIX_SPARSE, SUN_ERR_ARG_WRONGTYPE);
  SUNCheck(SM_ROWS_S(A) == SM_COLUMNS_S(A), SUN_ERR_ARG_DIMSMISMATCH);
  SUNCheck(SM_ROWS_S(A) == st->st_N, SUN_ERR_ARG_DIMSMISMATCH);
  SUNCheck(st->st_M <= st->st_N, SUN_ERR_ARG_DIMSMISMATCH);
  SUNCheck((SM_SPARSETYPE_S(A) == CSR_MAT) || (SM_SPARSETYPE_S(A) == CSC_MAT),
           SUN_ERR_ARG_OUTOFRANGE);

  if (SM_SPARSETYPE_S(A) == CSR_MAT)
  {
    sunindextype row = st->st_M, nnz = SM_NNZ_S(A) - 2 * (st->st_N - row);
    for (sunindextype i = 0; i < NNZ_spec; ++i)
    {
      const sunindextype var = NZ_spec[i];
      const uint8_t dd       = spec[var];
      for (uint8_t j = 0; j < dd; ++j)
      {
        const sunindextype col     = st->st_acc_varofs[var] + j;
        SM_DATA_S(A)[nnz]          = -cj;
        SM_DATA_S(A)[nnz + 1]      = ONE;
        SM_INDEXVALS_S(A)[nnz]     = col;
        SM_INDEXVALS_S(A)[nnz + 1] = col + 1;
        SM_INDEXPTRS_S(A)[row]     = nnz;
        row += 1;
        nnz += 2;
      }
    }

    SM_INDEXPTRS_S(A)[row] = nnz;
  }
  else
  {
    for (sunindextype i = 0; i < NNZ_spec; ++i)
    {
      const sunindextype var = NZ_spec[i];
      const uint8_t dd       = spec[var];
      for (uint8_t j = 0; j <= dd; ++j)
      {
        const sunindextype col     = st->st_acc_varofs[var] + j,
                           col_end = SM_INDEXPTRS_S(A)[col + 1];

        if (j != dd) { SM_DATA_S(A)[col_end - 1] = -cj; }
        if (0 < j)
        {
          const sunindextype ofs      = j < dd ? 2 : 1;
          SM_DATA_S(A)[col_end - ofs] = ONE;
        }
      }
    }
  }

  return SUN_SUCCESS;
}

static const DDMatrix_Ops DDMatrix_Ops_Sparse = {
  .evalddjac = DDMatEvalDDJac_Sparse};

DDMatrix* DDMatWrapSparse(const SUNMatrix A)
{
  SUNFunctionBegin(A->sunctx);

  SUNAssertNull(SUNMatGetID(A) == SUNMATRIX_SPARSE, SUN_ERR_ARG_WRONGTYPE);

  DDMatrix* B = malloc(sizeof(*B));
  SUNCheckNull(B, SUN_ERR_MALLOC_FAIL);

  B->A   = A;
  B->ops = &DDMatrix_Ops_Sparse;

  return B;
}

SUNMatrix DDSparseSUNMatFromStructure(const Structure* st,
                                      sunindextype NNZ,
                                      int sparsetype,
                                      SUNContext sunctx)
{
  SUNFunctionBegin(sunctx);

  SUNCheckNull(st, SUN_ERR_ARG_CORRUPT);
  SUNCheckNull(NNZ >= 0, SUN_ERR_ARG_OUTOFRANGE);

  SUNCheckNull(st->st_N >= st->st_M, SUN_ERR_ARG_DIMSMISMATCH);
  SUNCheckNull(sparsetype == CSC_MAT || sparsetype == CSR_MAT,
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
