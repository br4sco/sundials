#ifndef _DD_STAGED_PIVOT_IMPL_H
#define _DD_STAGED_PIVOT_IMPL_H

#include <stddef.h>
#include <sundials/sundials_context.h>
#include <sundials/sundials_types.h>

#include "dd_staged_pivot.h"
#include "dd_staged_pivot_matrix.h"
#include "static_info.h"

/** @file
 * @brief Internal staged-pivoting algorithm behind @ref DDSPStaged.
 *
 * Included by dd_staged_pivot.c and by the white-box unit tests that assert on
 * the per-stage `known_k` / `spec` state. Regular consumers should use the
 * DDStatePivot returned by @ref DDSPStaged instead.
 */

/* --------------------------------------------------------------------------
 * Staged Pivot Memory
 * -------------------------------------------------------------------------- */

typedef struct
{
  SUNContext sunctx; /**< Sundials context  */

  DDStaticInfo si; /**< Static DAE info */

  DDStagedPivotJacFn0* jacfn0; /**< Jacobian callback function for highest order derivatives */

  DDStagedPivotMatrix J_0; /**< Jacobian at stage k=0 */

  sunindextype N; /**< Number of variables and equations of the zero'th derivative order */

  uint8_t K;     /**< Number of stages */
  uint8_t* spec; /**< Dummy derivative specification */
  uint8_t* old_spec; /**< Previous spec, used to detect changes in DDSPPivotStaged */
  sunbooleantype** known_k; /**< Indicates known variables at each stage */
  sunindextype** vars_k;    /**< Pivoted variables in each stage */
  DDStagedPivotMatrix* J_k; /**< Jacobian subset for each stage */
  DDStagedPivotMatrixWorkspace* wss; /**< Workspace for each stage */
  sunindextype* vars_k_flat;         /**< `vars_k` as a flat array */
  sunbooleantype* known_k_flat;      /**< `known_k` as a flat array */
} _DDStagedPivot;

/** @brief Holds staged pivoting state */
typedef _DDStagedPivot* DDStagedPivot;

/**
 * @brief Creates staged pivot memory from static DAE info.
 *
 * @param[in] sunctx  SUNDIALS context.
 * @param[in] si      Static DAE info.
 * @param[in] J_0     N×N Jacobian matrix (pre-allocated); used as workspace
 *                    to evaluate @ref DDStagedPivotJacFn0 and extract sub-matrices.
 * @param[in] jacfn0  Jacobian callback (required; must not be NULL).
 *
 * @return A newly allocated @ref DDStagedPivot, or NULL on failure.
 */
DDStagedPivot DDSPCreateStaged(SUNContext sunctx,
                               DDStaticInfo si,
                               DDStagedPivotMatrix J_0,
                               DDStagedPivotJacFn0* jacfn0);

/** @brief Destroys staged pivot data. */
void DDSPDestroyStaged(DDStagedPivot*);

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
 * @param[in]  pm           Staged pivot memory.
 * @param[in]  tol          Pivot tolerance; a column is considered negligible
 *                          if its magnitude is below tol times the largest
 *                          column magnitude.
 * @param[in]  t            Current value of the independent variable.
 * @param[in]  Y            Current augmented state vector (length N_all_orders).
 * @param[inout] user_data  Passed through to the Jacobian callback.
 * @param[out] spec_changed Set to SUNTRUE if the spec changed since the last
 *                          call, SUNFALSE otherwise.
 *
 * @return SUN_SUCCESS, or a SUNDIALS error code on failure.
 */
SUNErrCode DDSPPivotStaged(DDStagedPivot pm,
                           sunrealtype tol,
                           sunrealtype t,
                           N_Vector Y,
                           void* user_data,
                           sunbooleantype* spec_changed);

#endif
