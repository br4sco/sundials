#include <assert.h>
#include <sundials/sundials_macros.h>
#include <sundials/sundials_matrix.h>
#include <sunmatrix/sunmatrix_dense.h>

#include "macros.h"
#include "matrix.h"

/* ==========================================================================
 * Generic Extended Sundials Matrix
 * ========================================================================== */

void ExtSUNMatDestroy(ExtSUNMatrix* self)
{
  if (self != NULL)
  {
    self->mat = NULL;
    self->ops = NULL;
    free(self);
  }
}

void ExtSUNMatWSDestroy(ExtSUNMatrixWS* self)
{
  if (self != NULL)
  {
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
}

/* ==========================================================================
 * Dense Extended Sundials Matrix
 * ========================================================================== */

static void ExtSUNMatWSContentDestroy_Dense(SUNDIALS_MAYBE_UNUSED ExtSUNMatrixWS* ws)
{
  return;
}

static ExtSUNMatrixWS* ExtSUNMatCreateWS_Dense(const ExtSUNMatrix self[static 1])
{
  SUNMatrix mat = ExtSUNMatGetMat(self);
  assert(SUNMatGetID(mat) == SUNMATRIX_DENSE);

  ExtSUNMatrixWS* ws = malloc(sizeof(*ws));
  if (ws == NULL) { return NULL; }

  ws->id      = EXTSUNMATRIXWS_ROWPIVOT;
  ws->destroy = ExtSUNMatWSContentDestroy_Dense;

  ws->content = malloc(SM_ROWS_D(mat) * sizeof(sunindextype));
  if (ws->content == NULL)
  {
    ExtSUNMatWSDestroy(ws);
    return NULL;
  }

  return ws;
}

sunbooleantype ExtSUNMatPivot_Dense(const ExtSUNMatrix self[static 1],
                                    const ExtSUNMatrixWS ws[static 1],
                                    sunrealtype tol, sunindextype n,
                                    sunindextype colpivots[static n])
{
  SUNMatrix mat = ExtSUNMatGetMat(self);
  assert(SUNMatGetID(mat) == SUNMATRIX_DENSE);

  if (n < 0 || n > SM_COLUMNS_D(mat)) { return SUNFALSE; }

  sunindextype* c      = colpivots;
  sunindextype* r      = (sunindextype*)ws->content;
  const sunindextype m = SM_ROWS_D(mat);

  for (sunindextype i = 0; i < m; ++i) { r[i] = i; }

  for (sunindextype j = 0; j < n; ++j) { c[j] = j; }

  for (sunindextype k = 0; k < SUNMIN(m, n); ++k)
  {
    /* Get current pivot value. */
    const sunrealtype pivotabsval = SUNRabs(SM_ELEMENT_D(mat, r[k], c[k]));

    /* Find the maximum value in the part of the matrix which we have
           not yet considered. */
    sunindextype max_i     = -1;
    sunindextype max_j     = -1;
    sunrealtype max_absval = SUNRabs(tol);
    for (sunindextype i = k; i < m; ++i)
    {
      for (sunindextype j = k; j < n; ++j)
      {
        sunrealtype tmp = SUNRabs(SM_ELEMENT_D(mat, r[i], c[j]));
        if (tmp > max_absval)
        {
          max_absval = tmp;
          max_i      = i;
          max_j      = j;
        }
      }
    }

    if (max_i == -1 || max_j == -1) /* The matrix is singular. */
    {
      return SUNFALSE;
    }

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

    /* Get the signed, non-zero, pivot vale after any potential
         * swapping. */
    const sunrealtype pivotval = SM_ELEMENT_D(mat, r[k], c[k]);
    assert(SUNRabs(pivotval) > tol);

    /* Perform one step of Gaussian elimination on the remaining rows.
         */
    for (sunindextype i = k + 1; i < m; ++i)
    {
      const sunrealtype leadval = SM_ELEMENT_D(mat, r[i], c[k]);
      if (leadval != SUN_RCONST(0.0))
      {
        const sunrealtype scaleval    = leadval / pivotval;
        SM_ELEMENT_D(mat, r[i], c[k]) = SUN_RCONST(0.0);
        for (sunindextype j = k + 1; j < n; ++j)
        {
          SM_ELEMENT_D(mat, r[i], c[j]) -= scaleval *
                                           SM_ELEMENT_D(mat, r[k], c[j]);
        }
      }
    }
  }

  return SUNTRUE;
}

static ExtSUNMatrix* ExtSUNMatCloneSub_Dense(const ExtSUNMatrix self[static 1],
                                             sunindextype m,
                                             const sunindextype rows[static m],
                                             sunindextype n,
                                             const sunindextype cols[static n])
{
  SUNMatrix mat = ExtSUNMatGetMat(self);
  assert(SUNMatGetID(mat) == SUNMATRIX_DENSE);

  if (m <= 0 || m > SM_ROWS_D(mat) || n <= 0 || n > SM_COLUMNS_D(mat))
  {
    return NULL;
  }

  SUNMatrix newmat = SUNDenseMatrix(m, n, mat->sunctx);
  if (newmat == NULL) { return NULL; }

  for (sunindextype j = 0; j < n; ++j)
  {
    assert(cols[j] >= 0 && cols[j] < SM_COLUMNS_D(mat));
    for (sunindextype i = 0; i < m; ++i)
    {
      assert(rows[i] >= 0 && rows[i] < SM_ROWS_D(mat));
      SM_ELEMENT_D(newmat, i, j) = SM_ELEMENT_D(mat, rows[i], cols[j]);
    }
  }

  return ExtSUNMatWrapDense(newmat);
}

static sunbooleantype ExtSUNMatCopySub_Dense(const ExtSUNMatrix self[static 1],
                                             const ExtSUNMatrix extmat[static 1],
                                             sunindextype m,
                                             const sunindextype rows[static m],
                                             sunindextype n,
                                             const sunindextype cols[static n])
{
  SUNMatrix selfmat = ExtSUNMatGetMat(self);
  assert(SUNMatGetID(selfmat) == SUNMATRIX_DENSE);
  SUNMatrix mat = ExtSUNMatGetMat(extmat);
  assert(SUNMatGetID(mat) == SUNMATRIX_DENSE);

  if (m < 0 || m > SM_ROWS_D(mat) || n < 0 || n > SM_COLUMNS_D(mat))
  {
    return SUNFALSE;
  }

  for (sunindextype j = 0; j < n; ++j)
  {
    assert(cols[j] >= 0 && cols[j] < SM_COLUMNS_D(selfmat));
    for (sunindextype i = 0; i < m; ++i)
    {
      assert(rows[i] >= 0 && rows[i] < SM_ROWS_D(selfmat));
      SM_ELEMENT_D(mat, i, j) = SM_ELEMENT_D(selfmat, rows[i], cols[j]);
    }
  }

  return true;
}

static const ExtSUNMatrix_Ops extended_SUNMatrix_Ops_Dense =
  {.createworkspace = ExtSUNMatCreateWS_Dense,
   .pivot           = ExtSUNMatPivot_Dense,
   .clonesub        = ExtSUNMatCloneSub_Dense,
   .copysub         = ExtSUNMatCopySub_Dense};

ExtSUNMatrix* ExtSUNMatWrapDense(const SUNMatrix mat)
{
  if (mat == NULL || SUNMatGetID(mat) != SUNMATRIX_DENSE) { return NULL; }

  ExtSUNMatrix* extmat = malloc(sizeof(*extmat));
  if (extmat == NULL) { return NULL; }

  extmat->mat = mat;
  extmat->ops = &extended_SUNMatrix_Ops_Dense;

  return extmat;
}
