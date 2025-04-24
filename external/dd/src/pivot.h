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

/** @brief Holds pivoting state */
typedef struct
{
  SUNContext sunctx;          /** Sundials context  */
  sunindextype pm_DAE_N;      /** DAE size */
  uint8_t pm_K;               /** Number of stages */
  uint8_t* pm_spec;           /** Dummy derivative specification */
  sunindextype pm_NNZ_spec;   /** Number of non-zero entries in `spec` */
  sunindextype* pm_NZ_spec;   /** Non-zero entries in `spec` */
  sunindextype pm_N_diff;     /** Number of diff equations in first-order DAE */
  sunindextype* pm_dvars;     /** Diff variables in first-order DAE */
  sunbooleantype** pm_known;  /** Indicates known variables at each stage */
  sunindextype** pm_vars;     /** Pivoted variables in each stage */
  DDMatrix** pm_Jk;           /** Jacobian subset for each stage */
  DDMatrixWorkspace** pm_wss; /** Workspace for each stage */
  sunindextype* pm_varsdata;  /** Varables in all stages  */
  sunbooleantype* pm_knowndata; /** Known variables in all stages */
} PivMem;

/** @brief Creates pivot data based on DAE structure. */
PivMem* PMCreate(SUNContext, const Structure[static 1], const DDMatrix[static 1]);

/** @brief Destroys pivot data. */
void PMDestroy(PivMem*);

/* /\** @brief Prints submatrix at the the given stage *\/ */
/* void PSPrintSubmat(const Structure[static 1], const PivMem[static 1], uint8_t, */
/*                    FILE*); */

/* --------------------------------------------------------------------------
 * Pivoting
 * -------------------------------------------------------------------------- */

/** @brief Pivots a DAE given its structure and Jacobian. */
SUNErrCode PPivot(const Structure[static 1], const DDMatrix[static 1],
                  sunrealtype, const PivMem[static 1]);

/** @brief Computes a DD spec from a pivoted DAE. */
SUNErrCode PPComputeDDSpec(const Structure[static 1], PivMem[static 1]);

/** @brief Update the DD spec. **/
SUNErrCode PPUpdateDDSpec(const Structure[static 1], const uint8_t[static 1],
                          PivMem[static 1]);

#endif
