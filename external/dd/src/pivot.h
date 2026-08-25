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
 * @brief Jacobian callback for the highest-order sub-system.
 *
 * Given canonical offsets c, d ∈ ℕ₀ⁿ and residual expressions e₀…eₙ₋₁,
 * this function must compute the N×N Jacobian J where
 *
 *   Jᵢⱼ = ∂eᵢ⁽ᶜⁱ⁾/∂yⱼ⁽ᵈʲ⁾.
 *
 * Only non-zero entries need to be written to `J`.
 *
 * @param[in]    t         Independent variable.
 * @param[in]    Y         Augmented state vector (length N_all_orders).
 * @param[out]   J         N×N Jacobian matrix to populate.
 * @param[inout] user_data User-defined data pointer.
 *
 * @return 0 on success, positive for a recoverable error, negative for a
 *         non-recoverable error.
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
} _PIVMem;

/** @brief Holds pivoting state */
typedef _PIVMem* PIVMem;

/**
 * @brief Creates pivot memory from static DAE info.
 *
 * @param[in] sunctx  SUNDIALS context.
 * @param[in] si      Static DAE info.
 * @param[in] J_0     N×N Jacobian matrix (pre-allocated); used as workspace
 *                    to evaluate @ref PIVJacFn0 and extract sub-matrices.
 * @param[in] jacfn0  Jacobian callback (required; must not be NULL).
 *
 * @return A newly allocated @ref PivMem, or NULL on failure.
 */
PIVMem PIVCreate(SUNContext sunctx,
                 DDStaticInfo si,
                 PIVMatrix J_0,
                 PIVJacFn0* jacfn0);

/** @brief Sets user data for the Jacobian callback **/
SUNErrCode PIVSetUserData(PIVMem, void*);

/** @brief Destroys pivot data. */
void PIVDestroy(PIVMem*);

/** @brief Print pvito data. */
void PIVPrint(PIVMem, FILE*);

/* /\** @brief Prints sub-matrix at the the given stage *\/ */
/* void PSPrintSubmat(const Structure[static 1], const PivMem[static 1], uint8_t, */
/*                    FILE*); */

/* --------------------------------------------------------------------------
 * Pivoting
 * -------------------------------------------------------------------------- */

/**
 * @brief Evaluates the Jacobian, selects dummy derivatives by pivoting, and
 *        updates the DD spec.
 *
 * At each stage k, M_k variables are identified as algebraically determined
 * ("known") by pivoting the M_k × N_k sub-Jacobian. Known variables include
 * both dummy derivatives (differential variables treated algebraically) and
 * algebraic variables from the overdetermined index-reduced system. The
 * remaining N_k − M_k variables are true state variables to be integrated.
 * `pm->spec[j]` counts the number of stages at which variable j is a true
 * state variable (spec[j] = 0 means j is always algebraically determined).
 *
 * @param[in]  pm           Pivot memory.
 * @param[in]  tol          Pivot tolerance; a column is considered negligible
 *                          if its magnitude is below tol times the largest
 *                          column magnitude.
 * @param[in]  t            Current value of the independent variable.
 * @param[in]  Y            Current augmented state vector (length N_all_orders).
 * @param[out] spec_changed Set to SUNTRUE if the spec changed since the last
 *                          call, SUNFALSE otherwise.
 *
 * @return SUN_SUCCESS, or a SUNDIALS error code on failure.
 */
SUNErrCode PIVPivot(PIVMem pm,
                    sunrealtype tol,
                    sunrealtype t,
                    N_Vector Y,
                    sunbooleantype* spec_changed);

#endif
