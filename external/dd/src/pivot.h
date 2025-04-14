#ifndef _DD_PIVOT_H
#define _DD_PIVOT_H

#include <assert.h>
#include <stddef.h>
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
  sunindextype pm_DAE_N;        /** DAE size */
  uint8_t pm_K;                 /** Number of stages */
  uint8_t* pm_spec;             /** Dummy derivative specification */
  sunindextype pm_NNZ_spec;     /** Number of non-zero entries in `spec` */
  sunindextype* pm_NZ_spec;     /** Non-zero entries in `spec` */
  sunbooleantype** pm_known;    /** Indicates known variables at each stage */
  sunindextype** pm_vars;       /** Pivoted variables in each stage */
  ExtSUNMatrix** pm_jacs;       /** Jacobian subset for each stage */
  ExtSUNMatrixWS** pm_wss;      /** Workspace for each stage */
  sunindextype* pm_varsdata;    /** Varables in all stages  */
  sunbooleantype* pm_knowndata; /** Known variables in all stages */
} PivMem;

/** @brief Creates pivot data based on DAE structure. */
PivMem* PMCreate(const Structure[static 1], const ExtSUNMatrix[static 1]);

/** @brief Destroys pivot data. */
void PMDestroy(PivMem*);

/** @brief Prints submatrix at the the given stage */
void PSPrintSubmat(const Structure[static 1], const PivMem[static 1], uint8_t,
                   FILE*);

/* --------------------------------------------------------------------------
 * Pivoting
 * -------------------------------------------------------------------------- */

/** @brief Pivots a DAE given its structure and Jacobian. */
sunbooleantype PPivot(const Structure[static 1], const ExtSUNMatrix[static 1],
                      sunrealtype, const PivMem[static 1]);

/** @brief Computes a DD spec from a pivoted DAE. */
sunbooleantype PPComputeDDSpec(const Structure[static 1], PivMem[static 1]);

/** @brief Update the DD spec. **/
sunbooleantype PPUpdateDDSpec(const uint8_t[static 1], PivMem[static 1]);

#endif
