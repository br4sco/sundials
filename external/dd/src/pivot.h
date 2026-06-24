#ifndef _DD_PIVOT_H
#define _DD_PIVOT_H

#include <assert.h>
#include <stddef.h>
#include <sundials/sundials_context.h>
#include <sundials/sundials_types.h>

#include "macros.h"
#include "matrix.h"
#include "structure.h"

/* ==========================================================================
 * Types and Interface
 * ========================================================================== */

/* --------------------------------------------------------------------------
 * Pivot Memory
 * -------------------------------------------------------------------------- */

DD_DEFINE_PAIR(sunindextype, sunindextype, sunindextype);

typedef struct
{
  SUNContext sunctx; /**< Sundials context  */

  sunindextype N; /**< Number of variables and equations of the zero'th derivative order */

  uint8_t K;                    /**< Number of stages */
  uint8_t* spec;                /**< Dummy derivative specification */
  sunbooleantype** known_k;     /**< Indicates known variables at each stage */
  sunindextype** vars_k;        /**< Pivoted variables in each stage */
  DDMatrix* J_k;                /**< Jacobian subset for each stage */
  DDMatrixWorkspace* wss;       /**< Workspace for each stage */
  sunindextype* vars_k_flat;    /**< `vars_k` as a flat array */
  sunbooleantype* known_k_flat; /**< `known_k` as a flat array */
} _PivMem;

/** @brief Holds pivoting state */
typedef _PivMem* PivMem;

/** @brief Creates pivot data based on DAE structure. */
PivMem PIVCreate(SUNContext, DAEStruct, DDMatrix);

/** @brief Destroys pivot data. */
void PIVDestroy(PivMem*);

/** @brief Print pvito data. */
void PIVPrint(DAEStruct, PivMem, FILE*);

/* /\** @brief Prints sub-matrix at the the given stage *\/ */
/* void PSPrintSubmat(const Structure[static 1], const PivMem[static 1], uint8_t, */
/*                    FILE*); */

/* --------------------------------------------------------------------------
 * Pivoting
 * -------------------------------------------------------------------------- */

/** @brief Pivots a DAE given its structure and Jacobian. */
SUNErrCode PIVPivot(DAEStruct, DDMatrix, sunrealtype, PivMem);

/** @brief Computes a DD spec from a pivoted DAE. */
SUNErrCode PIVComputeDDSpec(DAEStruct, PivMem);

#endif
