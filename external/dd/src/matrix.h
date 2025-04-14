#ifndef _DD_MATRIX_H
#define _DD_MATRIX_H

#include <assert.h>
#include <sundials/sundials_matrix.h>
#include <sunmatrix/sunmatrix_sparse.h>

#include "structure.h"

/** @file
 * @brief Extended Sundials matrix defintions.
 */

/* ==========================================================================
 * Generic Extended Sundials Matrix
 * ========================================================================== */

/* --------------------------------------------------------------------------
 * Generic Extended Sundials Matrices and Operations
 * -------------------------------------------------------------------------- */

typedef struct _ExtSUNMatrix_Ops ExtSUNMatrix_Ops;

/** @brief The type of extended generic matrices. */
typedef struct
{
  SUNMatrix mat;
  const ExtSUNMatrix_Ops* ops;
} ExtSUNMatrix;

/** @brief Condition number calculation workspace for generic matrices. */
typedef struct _generic_ExtSUNMatrixWS ExtSUNMatrixWS;

/** @brief Extended matrix workspace id. */
typedef enum
{
  EXTSUNMATRIXWS_ROWPIVOT
} ExtSUNMatrixWS_ID;

struct _generic_ExtSUNMatrixWS
{
  ExtSUNMatrixWS_ID id;
  void* content;
  void (*destroy)(ExtSUNMatrixWS*);
};

/** @brief API of extended generic matrices. */
struct _ExtSUNMatrix_Ops
{
  ExtSUNMatrixWS* (*const createworkspace)(const ExtSUNMatrix[static 1]);
  sunbooleantype (*const pivot)(const ExtSUNMatrix[static 1],
                                const ExtSUNMatrixWS[static 1], sunrealtype,
                                sunindextype n, sunindextype[static n]);
  ExtSUNMatrix* (*const clonesub)(const ExtSUNMatrix[static 1], sunindextype m,
                                  const sunindextype[static m], sunindextype n,
                                  const sunindextype[static n]);
  sunbooleantype (*const copysub)(const ExtSUNMatrix[static 1],
                                  const ExtSUNMatrix[static 1], sunindextype m,
                                  const sunindextype[static m], sunindextype n,
                                  const sunindextype[static n]);
};

/* --------------------------------------------------------------------------
 * Generic Extended Sundials Matrix Interface
 * -------------------------------------------------------------------------- */

/** @brief Destroys extended matrix. */
void ExtSUNMatDestroy(ExtSUNMatrix*);

/** @brief Returns underlying Sundials matrix. */
static inline SUNMatrix ExtSUNMatGetMat(const ExtSUNMatrix self[static 1])
{
  return self->mat;
}

/** @brief Create extended matrix workspace. */
static inline ExtSUNMatrixWS* ExtSUNMatCreateWS(const ExtSUNMatrix self[static 1])
{
  return self->ops->createworkspace(self);
}

/** @brief Pivots underlying matrix columns to the left. */
static inline sunbooleantype ExtSUNMatPivot(const ExtSUNMatrix self[static 1],
                                            const ExtSUNMatrixWS ws[static 1],
                                            sunrealtype tol, sunindextype n,
                                            sunindextype colpivots[static n])
{
  return self->ops->pivot(self, ws, tol, n, colpivots);
}

/** @brief Clones a submatrix. */
static inline ExtSUNMatrix* ExtSUNMatCloneSub(const ExtSUNMatrix self[static 1],
                                              sunindextype m,
                                              const sunindextype rows[static m],
                                              sunindextype n,
                                              const sunindextype cols[static n])
{
  return self->ops->clonesub(self, m, rows, n, cols);
}

/** @brief Copies a submatrix. */
static inline sunbooleantype ExtSUNMatCopySub(const ExtSUNMatrix self[static 1],
                                              const ExtSUNMatrix extmat[static 1],
                                              sunindextype m,
                                              const sunindextype rows[static m],
                                              sunindextype n,
                                              const sunindextype cols[static n])
{
  return self->ops->copysub(self, extmat, m, rows, n, cols);
}

/* --------------------------------------------------------------------------
 * Generic Extended Sundials Matrix Workspace Interface
 * -------------------------------------------------------------------------- */

/** @brief Destroys extended matrix workspace. */
void ExtSUNMatWSDestroy(ExtSUNMatrixWS*);

/* ==========================================================================
 * Dense Extended Sundials Matrix
 * ========================================================================== */

/** @brief Wraps a dense Sundials matrix in an extended matrix. */
ExtSUNMatrix* ExtSUNMatWrapDense(const SUNMatrix);

/* ==========================================================================
 * Creation and Pivoting of Structured Sundials Matrices
 * ========================================================================== */

#endif
