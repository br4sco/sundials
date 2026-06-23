#ifndef _DD_MATRIX_H
#define _DD_MATRIX_H

#include <assert.h>
#include <sundials/priv/sundials_errors_impl.h>
#include <sundials/sundials_matrix.h>
#include <sunmatrix/sunmatrix_sparse.h>

#include "structure.h"
#include "sundials/sundials_errors.h"
#include "sundials/sundials_types.h"

/** @file
 * @brief DD matrix definitions.
 */

/* ==========================================================================
 * Generic DD Matrix
 * ========================================================================== */

/* --------------------------------------------------------------------------
 * Generic DD Matrices and Operations
 * -------------------------------------------------------------------------- */

typedef struct _DDMatrix_Ops DDMatrix_Ops;

/** @brief The type of extended generic matrices. */
typedef struct
{
  SUNMatrix A;
  const DDMatrix_Ops* ops;
} _DDMatrix;

typedef _DDMatrix* DDMatrix;

/** @brief Condition number calculation workspace for generic matrices. */
typedef struct _generic_DDMatrixWorkspace* DDMatrixWorkspace;

/** @brief Extended matrix workspace id. */
typedef enum
{
  DDMATRIXWS_ROWPIVOT
} DDMatrixWorkspaceID;

struct _generic_DDMatrixWorkspace
{
  DDMatrixWorkspaceID id;
  void* content;
  void (*destroy)(DDMatrixWorkspace);
};

/** @brief API of extended generic matrices. */
struct _DDMatrix_Ops
{
  DDMatrixWorkspace (*const createworkspace)(DDMatrix);
  SUNErrCode (*const pivot)(DDMatrix,
                            DDMatrixWorkspace,
                            sunrealtype,
                            sunindextype n,
                            sunindextype[static n]);
  DDMatrix (*const clonesub)(DDMatrix,
                             sunindextype m,
                             const sunindextype[static m],
                             sunindextype n,
                             const sunindextype[static n]);
  SUNErrCode (*const copysub)(DDMatrix,
                              DDMatrix,
                              const sunindextype*,
                              const sunindextype*);
};

/* --------------------------------------------------------------------------
 * Generic DD Matrix Interface
 * -------------------------------------------------------------------------- */

/** @brief Destroys extended matrix. */
void DDMatDestroy(DDMatrix);

/** @brief Returns underlying Sundials matrix. */
static inline SUNMatrix DDMatGetSUNMat(DDMatrix self) { return self->A; }

/** @brief Create extended matrix workspace. */
static inline DDMatrixWorkspace DDMatCreateWS(DDMatrix self)
{
  SUNFunctionBegin(DDMatGetSUNMat(self)->sunctx);
  SUNCheckNull(self->ops->createworkspace, SUN_ERR_NOT_IMPLEMENTED);
  return self->ops->createworkspace(self);
}

/** @brief Pivots underlying matrix columns to the left. */
static inline SUNErrCode DDMatPivot(DDMatrix self,
                                    DDMatrixWorkspace ws,
                                    sunrealtype tol,
                                    sunindextype n,
                                    sunindextype colpivots[static n])
{
  SUNFunctionBegin(DDMatGetSUNMat(self)->sunctx);
  SUNCheck(self->ops->pivot, SUN_ERR_NOT_IMPLEMENTED);
  return self->ops->pivot(self, ws, tol, n, colpivots);
}

/** @brief Clones a sub-matrix. */
static inline DDMatrix DDMatCloneSub(const DDMatrix self,
                                     sunindextype m,
                                     const sunindextype rows[static m],
                                     sunindextype n,
                                     const sunindextype cols[static n])
{
  SUNFunctionBegin(DDMatGetSUNMat(self)->sunctx);
  SUNCheckNull(self->ops->clonesub, SUN_ERR_NOT_IMPLEMENTED);
  return self->ops->clonesub(self, m, rows, n, cols);
}

/** @brief Copies a sub-matrix. */
static inline SUNErrCode DDCopySub(DDMatrix self,
                                   DDMatrix A,
                                   const sunindextype* rows,
                                   const sunindextype* cols)
{
  SUNFunctionBegin(DDMatGetSUNMat(self)->sunctx);
  SUNCheck(self->ops->copysub, SUN_ERR_NOT_IMPLEMENTED);
  return self->ops->copysub(self, A, rows, cols);
}

/* --------------------------------------------------------------------------
 * Generic DD Matrix Workspace Interface
 * -------------------------------------------------------------------------- */

/** @brief Destroys extended matrix workspace. */
void DDMatWSDestroy(DDMatrixWorkspace);

/* ==========================================================================
 * Dense DD Matrix
 * ========================================================================== */

/** @brief Wraps a dense Sundials matrix in a DD matrix. */
DDMatrix DDMatWrapDense(SUNMatrix);

/* ==========================================================================
 * Sparse DD Matrix
 * ========================================================================== */

/** @brief Wraps a sparse Sundials matrix in a DD matrix. */
DDMatrix DDMatWrapSparse(SUNMatrix);

/** @brief Creates a sparse matrix based on additional structural information. */
SUNMatrix DDSparseSUNMatFromStructure(const DAEStruct*,
                                      sunindextype,
                                      int,
                                      SUNContext);

#endif
