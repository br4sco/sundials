#ifndef _DD_STATIC_INFO_H
#define _DD_STATIC_INFO_H

#include <assert.h>
#include <stdio.h>
#include <sundials/sundials_types.h>

/* ==========================================================================
 * Types
 * ========================================================================== */

typedef struct
{
  SUNContext sunctx; /**< Sundials context  */
  sunindextype N; /**< Number of variables and equations of the zero'th derivative order */
  sunindextype N_backwards; /**< Number of variables and equations of the backwards DAE */
  sunindextype M_all_orders; /**< Number of equations of all derivative orders */
  sunindextype N_all_orders; /**< Number of variables of all derivative orders */
  sunindextype N_diff; /**< Number of differential equations and differential variables in first-order index-reduced DAE */

  /** Equation offset vector of size `N_zeroth_order` */
  uint8_t* eqnofs;

  /** Variable offset vector of size `N_zeroth_order` */
  uint8_t* varofs;

  sunindextype** var_deriv_chains; /**< Maps Var-diff-order to index 0:(N_all_orders - 1) */
  sunindextype* var_deriv_chains_flat; /**< `var_deriv_chains` as a flat array */

  uint8_t K;                 /**< Number of stages */
  sunindextype* M_k;         /**< Number of equations at the k'th stage */
  sunindextype* N_k;         /**< Number of variables at the k'th stage */
  sunindextype** eqns_k;     /**< Equations at the k'th stage */
  sunindextype** vars_k;     /**< Variables at the k'th stage */
  sunindextype* eqns_k_flat; /**< `eqns_k` as a flat array */
  sunindextype* vars_k_flat; /**< `vars_k` as a flat array */

  const char** eqn_names; /**< Equation names */
  const char** var_names; /**< Variable names */
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

/** @brief Creates static DAE info. */
DDStaticInfo DDstaticInfoCreate(SUNContext sunctx,
                                sunindextype N,
                                const uint8_t eqnofs[static N],
                                const uint8_t varofs[static N],
                                const sunindextype* var_idx_map[static N],
                                const char** eqn_names,
                                const char** var_names);

/** @brief Destroys static DAE info. */
void DDstaticInfoDestroy(DDStaticInfo si);

/** @brief Prints static DAE info. */
void DDstaticInfoPrint(DDStaticInfo si, FILE*);

#endif
