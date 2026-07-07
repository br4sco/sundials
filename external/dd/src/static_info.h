#ifndef _DD_STATIC_INFO_H
#define _DD_STATIC_INFO_H

#include <assert.h>
#include <stdio.h>
#include <sundials/sundials_types.h>

/* ==========================================================================
 * Types
 * ========================================================================== */

/**
 * @brief Static (time-invariant) structural information of a high-index DAE.
 *
 * Given a DAE with N equations e₀…eₙ₋₁ and N variables y₀…yₙ₋₁, the
 * structural analysis produces canonical differentiation offsets c ∈ ℕ₀ⁿ
 * (@ref eqnofs) and d ∈ ℕ₀ⁿ (@ref varofs) such that the index-reduced system
 * is obtained by differentiating equation i exactly cᵢ times and introducing
 * derivatives of yⱼ up to order dⱼ.
 *
 * The augmented state vector Y has length @ref N_all_orders and contains
 * yⱼ, ẏⱼ, …, yⱼ⁽ᵈʲ⁾ for each j.
 *
 * The reduction is organised into K = maxⱼ(dⱼ) + 1 stages. At stage k the
 * active sub-system is square and pivoting selects which variables are
 * algebraically determined ("known") at that stage.
 */
typedef struct
{
  SUNContext sunctx; /**< SUNDIALS context */

  sunindextype N; /**< Number of original (zero'th-order) equations and variables */

  /** Length of the adjoint state vector: Σⱼ max(dⱼ, 1) */
  sunindextype N_backwards;

  /** Total equations in the augmented system: Σᵢ (cᵢ + 1) */
  sunindextype M_all_orders;

  /** Total variables in the augmented system: Σⱼ (dⱼ + 1) */
  sunindextype N_all_orders;

  /** Number of differential variables in the first-order order-reduced
      low-index DAE */
  sunindextype N_diff;

  /**
   * Canonical equation offsets c ∈ ℕ₀ⁿ (array of length N).
   * cᵢ is the number of times equation i must be differentiated in the
   * structural reduction.
   */
  uint8_t* eqnofs;

  /**
   * Canonical variable offsets d ∈ ℕ₀ⁿ (array of length N).
   * dⱼ is the highest derivative order of yⱼ that appears in the
   * augmented system.
   */
  uint8_t* varofs;

  /**
   * Derivative chain for each variable (array of N pointers).
   * `var_deriv_chains[j][k]` is the index of yⱼ⁽ᵏ⁾ in the augmented
   * state vector Y, for k = 0…dⱼ.
   */
  sunindextype** var_deriv_chains;

  sunindextype* var_deriv_chains_flat; /**< @ref var_deriv_chains as a flat array */

  uint8_t K;         /**< Number of stages; K = maxⱼ(dⱼ) + 1 */
  sunindextype* M_k; /**< Number of equations active at stage k (length K) */
  sunindextype* N_k; /**< Number of variables active at stage k (length K) */
  sunindextype** eqns_k;     /**< Indices of equations active at each stage */
  sunindextype** vars_k;     /**< Indices of variables active at each stage */
  sunindextype* eqns_k_flat; /**< @ref eqns_k as a flat array */
  sunindextype* vars_k_flat; /**< @ref vars_k as a flat array */

  const char** eqn_names; /**< Optional equation names; NULL entries are allowed */
  const char** var_names; /**< Optional variable names; NULL entries are allowed */
} DDstaticInfoRec;

/** @brief Encodes static (time-invariant) structural information of a high-index DAE. */
typedef DDstaticInfoRec* DDStaticInfo;

/* ==========================================================================
 * Macros
 * ========================================================================== */

/** @brief Converts stage index to stage offset. */
#define DDSI_STAGE_FROM_INDEX(si, k) ((int)k - (int)si->K + 1)

/** @brief Derivative order of equation at specified stage index */
#define DDSI_EQN_ORDER(si, k, i) \
  (DDSI_STAGE_FROM_INDEX(si, k) + (int)si->eqnofs[i])

/** @brief Derivative order of variable at specified stage index */
#define DDSI_VAR_ORDER(si, k, j) \
  (DDSI_STAGE_FROM_INDEX(si, k) + (int)si->varofs[j])

/** @brief Return the name given to an equation or an empty string if no names
 * was given. */
#define DDSI_EQN_NAME(si, i) \
  (si->eqn_names && si->eqn_names[i] ? si->eqn_names[i] : "")

/** @brief Return the name given to a variale or an empty string if no names
 * was given. */
#define DDSI_VAR_NAME(si, j) \
  (si->var_names && si->var_names[j] ? si->var_names[j] : "")

/* ==========================================================================
 * Interface
 * ========================================================================== */

/**
 * @brief Creates static DAE info from the canonical structural offsets.
 *
 * @param[in] sunctx      SUNDIALS context.
 * @param[in] N           Number of equations and variables in the original DAE.
 * @param[in] eqnofs      Canonical equation offsets c ∈ ℕ₀ⁿ (length N).
 * @param[in] varofs      Canonical variable offsets d ∈ ℕ₀ⁿ (length N).
 * @param[in] var_deriv_chains Derivative chain indices (array of N pointers).
 *                             `var_deriv_chains[j]` must point to an array of
 *                             dⱼ+1 indices giving the position of
 *                             yⱼ⁽⁰⁾…yⱼ⁽ᵈʲ⁾ in the augmented state vector Y.
 * @param[in] eqn_names   Optional array of N equation name strings (may be NULL).
 * @param[in] var_names   Optional array of N variable name strings (may be NULL).
 *
 * @return A newly allocated @ref DDStaticInfo, or NULL on failure.
 */
DDStaticInfo DDStaticInfoCreate(SUNContext sunctx,
                                sunindextype N,
                                const uint8_t eqnofs[static N],
                                const uint8_t varofs[static N],
                                const sunindextype* var_deriv_chains[static N],
                                const char** eqn_names,
                                const char** var_names);

/** @brief Destroys static DAE info. */
void DDStaticInfoDestroy(DDStaticInfo si);

/** @brief Prints static DAE info. */
void DDStaticInfoPrint(DDStaticInfo si, FILE*);

#endif
