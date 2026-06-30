#ifndef _DD_MATRIX_H
#define _DD_MATRIX_H

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
 * Generic Pivot Matrix
 * ========================================================================== */

/* --------------------------------------------------------------------------
 * Generic Pivot Matrices and Operations
 * -------------------------------------------------------------------------- */

typedef struct _PIVMatrix_Ops PIVMatrix_Ops;

typedef struct
{
  SUNMatrix A;
  const PIVMatrix_Ops* ops;
} _PIVMatrix;

/** @brief Opaque handle to a pivot matrix. */
typedef _PIVMatrix* PIVMatrix;

/** @brief Opaque handle to a pivot matrix workspace. */
typedef struct _generic_PIVMatrixWorkspace* PIVMatrixWorkspace;

/** @brief Identifies the type of a PIVMatrixWorkspace. */
typedef enum
{
  PIVMATRIXWS_ROWPIVOT /**< Workspace holding the subset of rows to pivot on */
} PIVMatrixWorkspaceID;

struct _generic_PIVMatrixWorkspace
{
  PIVMatrixWorkspaceID id;
  void* content;
  void (*destroy)(PIVMatrixWorkspace);
};

/** @brief API of extended generic matrices. */
struct _PIVMatrix_Ops
{
  PIVMatrixWorkspace (*const createworkspace)(PIVMatrix);
  SUNErrCode (*const pivot)(PIVMatrix,
                            PIVMatrixWorkspace,
                            sunrealtype,
                            sunindextype n,
                            sunindextype[static n]);
  PIVMatrix (*const clonesub)(PIVMatrix,
                              sunindextype m,
                              const sunindextype[static m],
                              sunindextype n,
                              const sunindextype[static n]);
  SUNErrCode (*const copysub)(PIVMatrix,
                              PIVMatrix,
                              const sunindextype*,
                              const sunindextype*);
};

/* --------------------------------------------------------------------------
 * Generic Pivot Matrix Interface
 * -------------------------------------------------------------------------- */

/** @brief Destroys extended matrix. */
void PIVMatDestroy(PIVMatrix);

/** @brief Returns underlying Sundials matrix. */
static inline SUNMatrix PIVMatGetSUNMat(PIVMatrix self) { return self->A; }

/** @brief Create extended matrix workspace. */
static inline PIVMatrixWorkspace PIVMatCreateWS(PIVMatrix self)
{
  SUNFunctionBegin(PIVMatGetSUNMat(self)->sunctx);
  SUNCheckNull(self->ops->createworkspace, SUN_ERR_NOT_IMPLEMENTED);
  return self->ops->createworkspace(self);
}

/**
 * @brief Reorders columns of the matrix so the most linearly independent
 *        ones come first.
 *
 * @param[in]    self       Pivot matrix.
 * @param[in]    ws         Workspace allocated by PIVMatCreateWS().
 * @param[in]    tol        Pivot tolerance.
 * @param[in]    n          Number of columns.
 * @param[inout] colpivots  On entry, the column indices to consider; on
 *                          return, reordered so the leading columns form a
 *                          well-conditioned square sub-matrix.
 */
static inline SUNErrCode PIVMatPivot(PIVMatrix self,
                                     PIVMatrixWorkspace ws,
                                     sunrealtype tol,
                                     sunindextype n,
                                     sunindextype colpivots[static n])
{
  SUNFunctionBegin(PIVMatGetSUNMat(self)->sunctx);
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
 * @return A newly allocated @ref PIVMatrix, or NULL on failure.
 */
static inline PIVMatrix PIVMatCloneSub(const PIVMatrix self,
                                       sunindextype m,
                                       const sunindextype rows[static m],
                                       sunindextype n,
                                       const sunindextype cols[static n])
{
  SUNFunctionBegin(PIVMatGetSUNMat(self)->sunctx);
  SUNCheckNull(self->ops->clonesub, SUN_ERR_NOT_IMPLEMENTED);
  return self->ops->clonesub(self, m, rows, n, cols);
}

/**
 * @brief Copies a sub-matrix of `self` into `A`.
 *
 * @param[in]  self  Source matrix.
 * @param[out] A     Destination sub-matrix (must already be allocated,
 *                   e.g. via PIVMatCloneSub()).
 * @param[in]  rows  Row indices into `self`.
 * @param[in]  cols  Column indices into `self`.
 */
static inline SUNErrCode PIVCopySub(PIVMatrix self,
                                    PIVMatrix A,
                                    const sunindextype* rows,
                                    const sunindextype* cols)
{
  SUNFunctionBegin(PIVMatGetSUNMat(self)->sunctx);
  SUNCheck(self->ops->copysub, SUN_ERR_NOT_IMPLEMENTED);
  return self->ops->copysub(self, A, rows, cols);
}

/* --------------------------------------------------------------------------
 * Generic DD Matrix Workspace Interface
 * -------------------------------------------------------------------------- */

/** @brief Destroys extended matrix workspace. */
void PIVMatWSDestroy(PIVMatrixWorkspace);

/* ==========================================================================
 * Dense DD Matrix
 * ========================================================================== */

/** @brief Wraps a dense Sundials matrix in a pivot matrix. */
PIVMatrix PIVMatWrapDense(SUNMatrix);

/* ==========================================================================
 * Sparse DD Matrix
 * ========================================================================== */

/** @brief Wraps a sparse Sundials matrix in a pivot matrix. */
PIVMatrix PIVMatWrapSparse(SUNMatrix);

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
SUNMatrix PIVSparseSUNMatFromStructure(const DDStaticInfo* si,
                                       sunindextype nnz,
                                       int sparsetype,
                                       SUNContext sunctx);

#endif
