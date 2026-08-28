#ifndef _DD_STAGED_PIVOT_H
#define _DD_STAGED_PIVOT_H

#include <sundials/sundials_context.h>
#include <sundials/sundials_types.h>

#include "dd_staged_pivot_matrix.h"
#include "dd_state_pivot.h"
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
typedef int DDStagedPivotJacFn0(sunrealtype t,
                                N_Vector Y,
                                SUNMatrix J,
                                void* user_data);

/**
 * @brief Creates a DDStatePivot (see dd_state_pivot.h) that selects dummy
 *        derivatives by staged pivoting of the sub-Jacobians, re-pivoting on
 *        every DDSPUpdate().
 *
 * The returned object owns its internal staged-pivot memory and frees it in
 * DDSPDestroy().
 *
 * @param[in] sunctx        SUNDIALS context.
 * @param[in] si            Static DAE info.
 * @param[in] J_0           N×N Jacobian matrix (pre-allocated); used as
 *                          workspace to evaluate @ref DDStagedPivotJacFn0 and
 *                          extract sub-matrices. Not owned.
 * @param[in] jacfn0 Jacobian callback (required; must not be NULL). Its
 *                   `user_data` argument is the pointer passed to the
 *                   corresponding DDSPUpdate() call.
 * @param[in] tol    Pivot tolerance.
 *
 * @return A newly allocated DDStatePivot, or NULL on failure.
 */
DDStatePivot DDSPStaged(SUNContext sunctx,
                        DDStaticInfo si,
                        DDStagedPivotMatrix J_0,
                        DDStagedPivotJacFn0* jacfn0,
                        sunrealtype tol);

#endif
