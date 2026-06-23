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
  sunindextype N_diff; /**< Number of differential equations and differential variables in first-order index-reduced DAE */

  /** Defines differential variable aliases. That is:
      `d/dt Y[diff_var_aliases[j].fst] = Y[diff_var_aliases[j].snd]`. */
  Pair_sunindextype* diff_var_aliases;

  /** `yy_diff_alias_row[i] >= 0` if `diff_var_aliases[j].fst = i` for some j
      and yy_diff_alias_row[i] is the equation number of this alias equation. */
  sunindextype* yy_diff_alias_row;

  /** `yp_diff_alias_row[i] >= 0` if `diff_var_aliases[j].snd = i` for some j
      and yy_diff_alias_row[i] is the equation number of this alias equation. */
  sunindextype* yp_diff_alias_row;

  uint8_t K;                    /**< Number of stages */
  uint8_t* spec;                /**< Dummy derivative specification */
  sunindextype NNZ_spec;        /**< Number of non-zero entries in `spec` */
  sunindextype* NZ_spec;        /**< Non-zero entries in `spec` */
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
void PIVDestroy(PivMem);

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

/** @brief Update the DD spec. **/
SUNErrCode PIVUpdateDDSpec(DAEStruct, const uint8_t[static 1], PivMem);

#endif
