#ifndef _DD_PIVOT_H
#define _DD_PIVOT_H

#include <assert.h>
#include <stddef.h>
#include <sundials/sundials_context.h>
#include <sundials/sundials_types.h>

#include "matrix.h"
#include "static_info.h"

/* ==========================================================================
 * Types and Interface
 * ========================================================================== */

/**
 * @brief Jacobian callback function for highest order derivatives (both
 *   equations and variables).
 *
 * given a structural analysis c,d ∈ ℕ[n], * dependent variables y ∈ ℝ[n], scalar
 * residual expressions e ∈ ℝ[n] then this function should compute the n×n
 * Jacobian: dʰe[i]/dy[j]ᵏ, where h = c[i] and k = d[j].
 *
 * @param[in] t is the independent variable.
 * @param[in] Y are the dependent variables and their derivatives.
 * @param[out] J holds the values of the n×n Jacobian. Only non-zero values
 *               needs to written to `J`.
 *
 * @param[inout] user_data points to user-defined data.
 *
 * @return a value `0` on success, a positive values if a recoverable error
 *         occurred and a negative value of a non-recoverable error occurred.
 */
typedef int PIVJacFn0(sunrealtype t, N_Vector Y, SUNMatrix J, void* user_data);

/* --------------------------------------------------------------------------
 * Pivot Memory
 * -------------------------------------------------------------------------- */

typedef struct
{
  SUNContext sunctx; /**< Sundials context  */

  DDStaticInfo si; /**< Static DAE info */

  PIVJacFn0* jacfn0; /**< Jacobian callback function for highest order derivatives */

  PIVMatrix J_0; /**< Jacobian at stage k=0 */

  void* user_data; /**< User data */

  sunindextype N; /**< Number of variables and equations of the zero'th derivative order */

  uint8_t K;         /**< Number of stages */
  uint8_t* spec;     /**< Dummy derivative specification */
  uint8_t* old_spec; /**< Previous spec, used to detect changes in PIVPivot */
  sunbooleantype** known_k;     /**< Indicates known variables at each stage */
  sunindextype** vars_k;        /**< Pivoted variables in each stage */
  PIVMatrix* J_k;               /**< Jacobian subset for each stage */
  PIVMatrixWorkspace* wss;      /**< Workspace for each stage */
  sunindextype* vars_k_flat;    /**< `vars_k` as a flat array */
  sunbooleantype* known_k_flat; /**< `known_k` as a flat array */
} _PivMem;

/** @brief Holds pivoting state */
typedef _PivMem* PivMem;

/** @brief Creates pivot data based on static DAE info. `jacfn0` is required. */
PivMem PIVCreate(SUNContext, DDStaticInfo, PIVMatrix, PIVJacFn0);

/** @brief Sets user data for the Jacobian callback **/
SUNErrCode PIVSetUserData(PivMem, void*);

/** @brief Destroys pivot data. */
void PIVDestroy(PivMem*);

/** @brief Print pvito data. */
void PIVPrint(PivMem, FILE*);

/* /\** @brief Prints sub-matrix at the the given stage *\/ */
/* void PSPrintSubmat(const Structure[static 1], const PivMem[static 1], uint8_t, */
/*                    FILE*); */

/* --------------------------------------------------------------------------
 * Pivoting
 * -------------------------------------------------------------------------- */

/** @brief Pivots a DAE given its structure and Jacobian, and computes the DD
 *         spec. Sets *spec_changed to SUNTRUE if the spec changed, SUNFALSE
 *         otherwise. Returns a SUNErrCode. */
SUNErrCode PIVPivot(PivMem, sunrealtype, sunrealtype, N_Vector, sunbooleantype*);

/** @brief Returns the current dummy derivative spec of this pivot memory */
uint8_t* PIVGetSpec(PivMem);

#endif
