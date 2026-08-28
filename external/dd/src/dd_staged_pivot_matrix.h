#ifndef _DD_STAGED_PIVOT_MATRIX_H
#define _DD_STAGED_PIVOT_MATRIX_H

#include <assert.h>
#include <sundials/priv/sundials_errors_impl.h>
#include <sundials/sundials_matrix.h>
#include <sunmatrix/sunmatrix_sparse.h>

#include "static_info.h"
#include "sundials/sundials_errors.h"
#include "sundials/sundials_types.h"

/** @file
 * @brief Generic matrix abstraction for column-pivoting operations.
 */

/* ==========================================================================
 * Generic Staged Pivot Matrix
 * ========================================================================== */

/* --------------------------------------------------------------------------
 * Generic Staged Pivot Matrices and Operations
 * -------------------------------------------------------------------------- */

typedef struct _DDStagedPivotMatrix_Ops DDStagedPivotMatrix_Ops;

typedef struct
{
  SUNMatrix A;
  const DDStagedPivotMatrix_Ops* ops;
} _DDStagedPivotMatrix;

/** @brief Opaque handle to a staged pivot matrix. */
typedef _DDStagedPivotMatrix* DDStagedPivotMatrix;

/** @brief Opaque handle to a staged pivot matrix workspace. */
typedef struct _generic_DDStagedPivotMatrixWorkspace* DDStagedPivotMatrixWorkspace;

/** @brief Identifies the type of a DDStagedPivotMatrixWorkspace. */
typedef enum
{
  DDSTAGEDPIVOTMATRIXWS_ROWPIVOT /**< Workspace holding the subset of rows to pivot on */
} DDStagedPivotMatrixWorkspaceID;

struct _generic_DDStagedPivotMatrixWorkspace
{
  DDStagedPivotMatrixWorkspaceID id;
  void* content;
  void (*destroy)(DDStagedPivotMatrixWorkspace);
};

/** @brief API of extended generic matrices. */
struct _DDStagedPivotMatrix_Ops
{
  DDStagedPivotMatrixWorkspace (*const createworkspace)(DDStagedPivotMatrix);
  SUNErrCode (*const pivot)(DDStagedPivotMatrix,
                            DDStagedPivotMatrixWorkspace,
                            sunrealtype,
                            sunindextype n,
                            sunindextype[static n]);
  DDStagedPivotMatrix (*const clonesub)(DDStagedPivotMatrix,
                                        sunindextype m,
                                        const sunindextype[static m],
                                        sunindextype n,
                                        const sunindextype[static n]);
  SUNErrCode (*const copysub)(DDStagedPivotMatrix,
                              DDStagedPivotMatrix,
                              const sunindextype*,
                              const sunindextype*);
};

/* --------------------------------------------------------------------------
 * Generic Staged Pivot Matrix Interface
 * -------------------------------------------------------------------------- */

/** @brief Destroys extended matrix. */
void DDStagedPivotMatDestroy(DDStagedPivotMatrix);

/** @brief Returns underlying Sundials matrix. */
static inline SUNMatrix DDStagedPivotMatGetSUNMat(DDStagedPivotMatrix self)
{
  return self->A;
}

/** @brief Create extended matrix workspace. */
static inline DDStagedPivotMatrixWorkspace DDStagedPivotMatCreateWS(
  DDStagedPivotMatrix self)
{
  SUNFunctionBegin(DDStagedPivotMatGetSUNMat(self)->sunctx);
  SUNCheckNull(self->ops->createworkspace, SUN_ERR_NOT_IMPLEMENTED);
  return self->ops->createworkspace(self);
}

/**
 * @brief Reorders columns of the matrix so the most linearly independent
 *        ones come first.
 *
 * @param[in]    self       Staged pivot matrix.
 * @param[in]    ws         Workspace allocated by DDStagedPivotMatCreateWS().
 * @param[in]    tol        Pivot tolerance.
 * @param[in]    n          Number of columns.
 * @param[inout] colpivots  On entry, the column indices to consider; on
 *                          return, reordered so the leading columns form a
 *                          well-conditioned square sub-matrix.
 */
static inline SUNErrCode DDStagedPivotMatPivot(DDStagedPivotMatrix self,
                                               DDStagedPivotMatrixWorkspace ws,
                                               sunrealtype tol,
                                               sunindextype n,
                                               sunindextype colpivots[static n])
{
  SUNFunctionBegin(DDStagedPivotMatGetSUNMat(self)->sunctx);
  SUNCheck(self->ops->pivot, SUN_ERR_NOT_IMPLEMENTED);
  return self->ops->pivot(self, ws, tol, n, colpivots);
}

/**
 * @brief Allocates a new matrix containing the M×N sub-matrix of `self`
 *        at the given row and column indices.
 *
 * @param[in] self  Source matrix.
 * @param[in] m     Number of rows in the sub-matrix.
 * @param[in] rows  Row indices into `self` (length m).
 * @param[in] n     Number of columns in the sub-matrix.
 * @param[in] cols  Column indices into `self` (length n).
 *
 * @return A newly allocated @ref DDStagedPivotMatrix, or NULL on failure.
 */
static inline DDStagedPivotMatrix DDStagedPivotMatCloneSub(
  const DDStagedPivotMatrix self,
  sunindextype m,
  const sunindextype rows[static m],
  sunindextype n,
  const sunindextype cols[static n])
{
  SUNFunctionBegin(DDStagedPivotMatGetSUNMat(self)->sunctx);
  SUNCheckNull(self->ops->clonesub, SUN_ERR_NOT_IMPLEMENTED);
  return self->ops->clonesub(self, m, rows, n, cols);
}

/**
 * @brief Copies a sub-matrix of `self` into `A`.
 *
 * @param[in]  self  Source matrix.
 * @param[out] A     Destination sub-matrix (must already be allocated,
 *                   e.g. via DDStagedPivotMatCloneSub()).
 * @param[in]  rows  Row indices into `self`.
 * @param[in]  cols  Column indices into `self`.
 */
static inline SUNErrCode DDStagedPivotMatCopySub(DDStagedPivotMatrix self,
                                                 DDStagedPivotMatrix A,
                                                 const sunindextype* rows,
                                                 const sunindextype* cols)
{
  SUNFunctionBegin(DDStagedPivotMatGetSUNMat(self)->sunctx);
  SUNCheck(self->ops->copysub, SUN_ERR_NOT_IMPLEMENTED);
  return self->ops->copysub(self, A, rows, cols);
}

/* --------------------------------------------------------------------------
 * Generic DD Matrix Workspace Interface
 * -------------------------------------------------------------------------- */

/** @brief Destroys extended matrix workspace. */
void DDStagedPivotMatWSDestroy(DDStagedPivotMatrixWorkspace);

/* ==========================================================================
 * Dense DD Matrix
 * ========================================================================== */

/** @brief Wraps a dense Sundials matrix in a staged pivot matrix. */
DDStagedPivotMatrix DDStagedPivotMatWrapDense(SUNMatrix);

/* ==========================================================================
 * Sparse DD Matrix
 * ========================================================================== */

/** @brief Wraps a sparse Sundials matrix in a staged pivot matrix. */
DDStagedPivotMatrix DDStagedPivotMatWrapSparse(SUNMatrix);

/**
 * @brief Allocates a sparse SUNMatrix sized for the augmented DAE Jacobian.
 *
 * @param[in] si          Static DAE info.
 * @param[in] nnz         Number of non-zero entries.
 * @param[in] sparsetype  Storage format: CSR_MAT or CSC_MAT.
 * @param[in] sunctx      SUNDIALS context.
 *
 * @return A newly allocated sparse SUNMatrix, or NULL on failure.
 */
SUNMatrix DDStagedPivotSparseSUNMatFromStructure(const DDStaticInfo* si,
                                                 sunindextype nnz,
                                                 int sparsetype,
                                                 SUNContext sunctx);

#endif
