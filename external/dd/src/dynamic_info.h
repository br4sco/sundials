#ifndef _DYNAMIC_INFO_H
#define _DYNAMIC_INFO_H

#include <sundials/sundials_types.h>

#include "macros.h"
#include "static_info.h"

/* ==========================================================================
 * DD State Memory
 * ========================================================================== */

/** @brief A pair `(fst, snd)` of augmented-state indices encoding one trivial
 * alias equation `Y[snd] - Yp[fst] = 0`. */
DD_DEFINE_PAIR(sunindextype, sunindextype, sunindextype);

/**
 * @brief Dynamic (dummy-derivative-dependent) state of a high-index DAE.
 *
 * Whereas @ref DDStaticInfo encodes the fixed structural offsets of the
 * augmented system, DDDAEState encodes the *current* dummy derivative
 * specification (see `DDSetSpec` in dd.h): which augmented-state entries are
 * true differential variables and which are dummy derivatives aliased back
 * to a "value" entry. This partition changes whenever the specification is
 * updated, hence "dynamic". It is recomputed by @ref DDDAEStateUpdate.
 */
struct DDDAEStateRec
{
  SUNContext sunctx; /**< Context used for error handling. */
  DDStaticInfo si;   /**< Static info this state was built from; not owned,
                           must outlive the DDDAEState. */

  /**
   * Trivial alias equations introduced by the current dummy derivative
   * specification (array of length `si->N_diff`).
   * `diff_var_aliases[k] = {fst, snd}` encodes the equation
   * `Y[snd] - Yp[fst] = 0`: augmented-state entry `fst` is a true
   * differential variable and `snd` is its paired dummy-derivative
   * "value" entry.
   */
  Pair_sunindextype* diff_var_aliases;

  /**
   * For each augmented-state index `j`, the row of the alias equation in
   * which `j` plays the `Yp[fst]` role (coefficient `-cj` in the Jacobian),
   * or -1 if no such equation exists (array of length `si->N_all_orders`).
   * @see diff_var_aliases
   */
  sunindextype* yy_diff_alias_row;

  /**
   * For each augmented-state index `j`, the row of the alias equation in
   * which `j` plays the `Y[snd]` role (coefficient `1` in the Jacobian), or
   * -1 if no such equation exists (array of length `si->N_all_orders`).
   * @see diff_var_aliases
   */
  sunindextype* yp_diff_alias_row;
};

typedef struct DDDAEStateRec* DDDAEState;

/** @brief Destroys dynamic DAE state and sets `*state_ptr` to NULL; a no-op
 * if `state_ptr` or `*state_ptr` is NULL. */
void DDDAEStateDestroy(DDDAEState*);

/**
 * @brief Creates dynamic DAE state for the given static info.
 *
 * @param[in] sunctx Context used for error handling.
 * @param[in] si     Static DAE info; stored by reference (not copied) and
 *                    must outlive the returned state.
 *
 * @return A newly allocated @ref DDDAEState, or NULL on failure.
 */
DDDAEState DDDAEStateCreate(SUNContext, DDStaticInfo);

/**
 * @brief Creates a deep copy of dynamic DAE state.
 *
 * @param[in] state State to clone; the clone shares its `si`.
 *
 * @return A newly allocated @ref DDDAEState, or NULL on failure.
 */
DDDAEState DDDAEStateClone(DDDAEState);

/**
 * @brief Copies dynamic DAE state from `src` into `dst`.
 *
 * @param[in]  src Source state.
 * @param[out] dst Destination state; must share the same `si` as `src`.
 */
void DDDAEStateCopy(DDDAEState, DDDAEState);

/**
 * @brief Recomputes dynamic DAE state for a new dummy derivative
 *        specification.
 *
 * Rebuilds `diff_var_aliases`, `yy_diff_alias_row`, and `yp_diff_alias_row`
 * from `spec`.
 *
 * @param[in] state State to update.
 * @param[in] spec  Dummy derivative specification (array of length
 *                  `state->si->N`); see `DDSetSpec` in dd.h for the encoding.
 *
 * @return SUN_SUCCESS or an error code.
 */
SUNErrCode DDDAEStateUpdate(DDDAEState, uint8_t*);

#endif
