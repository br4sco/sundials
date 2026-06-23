#ifndef _DD_STRUCTURE_H
#define _DD_STRUCTURE_H

#include <assert.h>
#include <stdio.h>
#include <sundials/sundials_types.h>

/* ==========================================================================
 * Types
 * ========================================================================== */

typedef struct
{
  sunindextype N; /**< Number of variables and equations of the zero'th derivative order */
  sunindextype N_backwards; /**< Number of variables and equations of the backwards DAE */
  sunindextype M_all_orders; /**< Number of equations of all derivative orders */
  sunindextype N_all_orders; /**< Number of variables of all derivative orders */

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
} _DAEStruct;

/** @brief Encodes high-index DAE structure. */
typedef _DAEStruct* DAEStruct;

/* ==========================================================================
 * Macros
 * ========================================================================== */

/** @brief Converts stage index to stage offset. */
#define ST_STAGE_FROM_INDEX(st, k) ((int)k - (int)st->K + 1)

/** @brief Derivative order of equation at specified stage index */
#define ST_EQN_ORDER(st, k, i) (ST_STAGE_FROM_INDEX(st, k) + (int)st->eqnofs[i])

/** @brief Derivative order of variable at specified stage index */
#define ST_VAR_ORDER(st, k, j) (ST_STAGE_FROM_INDEX(st, k) + (int)st->varofs[j])

/** @brief Return the name given to an equation or an empty string if no names
 * was given. */
#define ST_EQN_NAME(st, i) \
  (st->eqn_names && st->eqn_names[i] ? st->eqn_names[i] : "")

/** @brief Return the name given to a variale or an empty string if no names
 * was given. */
#define ST_VAR_NAME(st, j) \
  (st->var_names && st->var_names[j] ? st->var_names[j] : "")

/* ==========================================================================
 * Interface
 * ========================================================================== */

/** @brief Creates DAE structure. */
DAEStruct STCreate(sunindextype N,
                   const uint8_t eqnofs[static N],
                   const uint8_t varofs[static N],
                   const sunindextype* var_idx_map[static N],
                   const char** eqn_names,
                   const char** var_names);

/** @brief Destroys DAE structure. */
void STDestroy(DAEStruct);

/** @brief Prints DAE structure. */
void STPrint(DAEStruct, FILE*);

#endif
