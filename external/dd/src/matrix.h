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
 * @brief DD matrix defintions.
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
} DDMatrix;

/** @brief Condition number calculation workspace for generic matrices. */
typedef struct _generic_DDMatrixWorkspace DDMatrixWorkspace;

/** @brief Extended matrix workspace id. */
typedef enum
{
  DDMATRIXWS_ROWPIVOT
} DDMatrixWorkspaceID;

struct _generic_DDMatrixWorkspace
{
  DDMatrixWorkspaceID id;
  void* content;
  void (*destroy)(DDMatrixWorkspace*);
};

/** @brief API of extended generic matrices. */
struct _DDMatrix_Ops
{
  DDMatrixWorkspace* (*const createworkspace)(const DDMatrix[static 1]);
  SUNErrCode (*const pivot)(const DDMatrix[static 1],
                            const DDMatrixWorkspace[static 1], sunrealtype,
                            sunindextype n, sunindextype[static n]);
  DDMatrix* (*const clonesub)(const DDMatrix[static 1], sunindextype m,
                              const sunindextype[static m], sunindextype n,
                              const sunindextype[static n]);
  SUNErrCode (*const copysub)(const DDMatrix[static 1], const DDMatrix[static 1],
                              sunindextype m, const sunindextype[static m],
                              sunindextype n, const sunindextype[static n]);
  SUNErrCode (*const evalddjac)(const DDMatrix[static 1],
                                const Structure[static 1], sunindextype NNZ_spec,
                                const sunindextype[static NNZ_spec],
                                const uint8_t[static 1], sunrealtype);
};

/* --------------------------------------------------------------------------
 * Generic DD Matrix Interface
 * -------------------------------------------------------------------------- */

/** @brief Destroys extended matrix. */
void DDMatDestroy(DDMatrix*);

/** @brief Returns underlying Sundials matrix. */
static inline SUNMatrix DDMatGetSUNMat(const DDMatrix self[static 1])
{
  return self->A;
}

/** @brief Create extended matrix workspace. */
static inline DDMatrixWorkspace* DDMatCreateWS(const DDMatrix self[static 1])
{
  SUNFunctionBegin(DDMatGetSUNMat(self)->sunctx);
  SUNCheckNull(self->ops->createworkspace, SUN_ERR_NOT_IMPLEMENTED);
  return self->ops->createworkspace(self);
}

/** @brief Pivots underlying matrix columns to the left. */
static inline SUNErrCode DDMatPivot(const DDMatrix self[static 1],
                                    const DDMatrixWorkspace ws[static 1],
                                    sunrealtype tol, sunindextype n,
                                    sunindextype colpivots[static n])
{
  SUNFunctionBegin(DDMatGetSUNMat(self)->sunctx);
  SUNCheck(self->ops->pivot, SUN_ERR_NOT_IMPLEMENTED);
  return self->ops->pivot(self, ws, tol, n, colpivots);
}

/** @brief Clones a submatrix. */
static inline DDMatrix* DDMatCloneSub(const DDMatrix self[static 1],
                                      sunindextype m,
                                      const sunindextype rows[static m],
                                      sunindextype n,
                                      const sunindextype cols[static n])
{
  SUNFunctionBegin(DDMatGetSUNMat(self)->sunctx);
  SUNCheckNull(self->ops->clonesub, SUN_ERR_NOT_IMPLEMENTED);
  return self->ops->clonesub(self, m, rows, n, cols);
}

/** @brief Copies a submatrix. */
static inline SUNErrCode DDCopySub(const DDMatrix self[static 1],
                                   const DDMatrix A[static 1], sunindextype m,
                                   const sunindextype rows[static m],
                                   sunindextype n,
                                   const sunindextype cols[static n])
{
  SUNFunctionBegin(DDMatGetSUNMat(self)->sunctx);
  SUNCheck(self->ops->copysub, SUN_ERR_NOT_IMPLEMENTED);
  return self->ops->copysub(self, A, m, rows, n, cols);
}

/** @brief Evaluates the part of the Jacobian related to pivoting DD
    variables. */
static inline SUNErrCode DDMatEvalDDJac(
  const DDMatrix self[static 1], const Structure st[static 1],
  sunindextype NNZ_spec, const sunindextype NZ_spec[static NNZ_spec],
  const uint8_t spec[static 1], sunrealtype cj)
{
  SUNFunctionBegin(DDMatGetSUNMat(self)->sunctx);
  SUNCheck(self->ops->evalddjac, SUN_ERR_NOT_IMPLEMENTED);
  return self->ops->evalddjac(self, st, NNZ_spec, NZ_spec, spec, cj);
}

/* --------------------------------------------------------------------------
 * Generic DD Matrix Workspace Interface
 * -------------------------------------------------------------------------- */

/** @brief Destroys extended matrix workspace. */
void DDMatWSDestroy(DDMatrixWorkspace*);

/* ==========================================================================
 * Dense DD Matrix
 * ========================================================================== */

/** @brief Wraps a dense Sundials matrix in a DD matrix. */
DDMatrix* DDMatWrapDense(const SUNMatrix);

/* ==========================================================================
 * Sparse DD Matrix
 * ========================================================================== */

/** @brief Wraps a sparse Sundials matrix in a DD matrix. */
DDMatrix* DDMatWrapSparse(const SUNMatrix);

/** @brief Creates a sparse matrix based on additional structural information. */
SUNMatrix DDSparseSUNMatFromStructure(const Structure*, sunindextype, int,
                                      SUNContext);

#endif
