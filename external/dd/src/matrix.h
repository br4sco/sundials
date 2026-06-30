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
 * @brief pivot matrix definitions.
 */

/* ==========================================================================
 * Generic Pivot Matrix
 * ========================================================================== */

/* --------------------------------------------------------------------------
 * Generic Pivot Matrices and Operations
 * -------------------------------------------------------------------------- */

typedef struct _PIVMatrix_Ops PIVMatrix_Ops;

/** @brief The type of extended generic matrices. */
typedef struct
{
  SUNMatrix A;
  const PIVMatrix_Ops* ops;
} _PIVMatrix;

typedef _PIVMatrix* PIVMatrix;

/** @brief Condition number calculation workspace for generic matrices. */
typedef struct _generic_PIVMatrixWorkspace* PIVMatrixWorkspace;

/** @brief Extended matrix workspace id. */
typedef enum
{
  PIVMATRIXWS_ROWPIVOT
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

/** @brief Pivots underlying matrix columns to the left. */
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

/** @brief Clones a sub-matrix. */
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

/** @brief Copies a sub-matrix. */
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

/** @brief Creates a sparse matrix based on additional structural information. */
SUNMatrix PIVSparseSUNMatFromStructure(const DDStaticInfo*,
                                       sunindextype,
                                       int,
                                       SUNContext);

#endif
