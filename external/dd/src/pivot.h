#ifndef _DD_PIVOT_H
#define _DD_PIVOT_H

#include <assert.h>
#include <stddef.h>
#include <sundials/sundials_context.h>
#include <sundials/sundials_types.h>

#include "matrix.h"
#include "structure.h"

/* ==========================================================================
 * Types and Interface
 * ========================================================================== */

/* --------------------------------------------------------------------------
 * Pivot Memory
 * -------------------------------------------------------------------------- */

typedef struct
{
  SUNContext sunctx;        /** Sundials context  */
  sunindextype DAE_size;    /** DAE size */
  uint8_t K;                /** Number of stages */
  uint8_t* spec;            /** Dummy derivative specification */
  sunindextype NNZ_spec;    /** Number of non-zero entries in `spec` */
  sunindextype* NZ_spec;    /** Non-zero entries in `spec` */
  sunindextype N_diff_vars; /** Number of diff equations in first-order DAE */
  sunindextype* diff_vars;  /** Diff variables in first-order DAE */
  sunbooleantype** known_k; /** Indicates known variables at each stage */
  sunindextype** vars_k;    /** Pivoted variables in each stage */
  DDMatrix* J_k;            /** Jacobian subset for each stage */
  DDMatrixWorkspace* wss;   /** Workspace for each stage */
  sunindextype* vars;       /** Varables in all stages  */
  sunbooleantype* known;    /** Known variables in all stages */
} _PivMem;

/** @brief Holds pivoting state */
typedef _PivMem* PivMem;

/** @brief Creates pivot data based on DAE structure. */
PivMem PMCreate(SUNContext, Struc, DDMatrix);

/** @brief Destroys pivot data. */
void PMDestroy(PivMem);

/* /\** @brief Prints sub-matrix at the the given stage *\/ */
/* void PSPrintSubmat(const Structure[static 1], const PivMem[static 1], uint8_t, */
/*                    FILE*); */

/* --------------------------------------------------------------------------
 * Pivoting
 * -------------------------------------------------------------------------- */

/** @brief Pivots a DAE given its structure and Jacobian. */
SUNErrCode PPivot(Struc, DDMatrix, sunrealtype, PivMem);

/** @brief Computes a DD spec from a pivoted DAE. */
SUNErrCode PPComputeDDSpec(Struc, PivMem);

/** @brief Update the DD spec. **/
SUNErrCode PPUpdateDDSpec(Struc, const uint8_t[static 1], PivMem);

#endif
