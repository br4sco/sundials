#ifndef _DD_STRUCTURE_H
#define _DD_STRUCTURE_H

#include <assert.h>
#include <sundials/sundials_types.h>

/* ==========================================================================
 * Types
 * ========================================================================== */

/** @brief Sparese CSR representation if Sigma matrix, where columns enumerate
    variables and rows enumerate equations. A stored zero at (i,j) means that a
    variable j has differentiation order zero in the equation i, while if (i,j)
    is not stored in this sparse matrix it means the variable j does not appear
    in equation i. */
typedef struct {
  sunindextype N;           /** Matrix size (we assume a square matrix). */
  sunindextype NNZ;         /** Number of non-zero elements. */
  sunindextype *colindices; /** Pointers to non-zero column indices. */
  uint8_t **rowindexptrs;   /** Pointers to the first element in each row. */
  uint8_t *data;            /** Non-zero values. */
} SigmaMatrix;

typedef struct {
  sunindextype DAE_size;          /**< DAE size */
  sunindextype DAE_backward_size; /**< Size of backwards DAE */
  sunindextype M;                 /**< Number of equation symbols */
  sunindextype N;                 /**< Number of variable symbols */

  /** Equation offset vector of size `DAE_size` */
  uint8_t *eqnofs;

  /** Variable offset vector of size `DAE_size` */
  uint8_t *varofs;

  sunindextype *eqn_to_idx;       /**< Maps equation to linear index */
  sunindextype *var_to_idx;       /**< Maps variable to linear index */
  sunindextype **var_idx_map;     /**< Maps Var-diff-order to index */
  sunindextype *var_idx_map_data; /**< `var_idx_map` data */

  uint8_t K;             /**< Number of stages */
  sunindextype *M_k;     /**< Number of equations at the k'th stage */
  sunindextype *N_k;     /**< Number of variables at the k'th stage */
  sunindextype **eqns_k; /**< Equations at the k'th stage */
  sunindextype **vars_k; /**< Variables at the k'th stage */
  sunindextype *eqns;    /**< Equations in all stages */
  sunindextype *vars;    /**< Variables in all stages */

  char **eqn_names; /**< Equation names */
  char **var_names; /**< Variable names */
} _DAEStruct;

/** @brief Encodes high-index DAE structure. */
typedef _DAEStruct *DAEStruct;

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
#define ST_EQN_NAME(st, i)                                                     \
  (st->eqn_names && st->eqn_names[i] ? st->eqn_names[i] : "")

/** @brief Return the name given to a variale or an empty string if no names
 * was given. */
#define ST_VAR_NAME(st, j)                                                     \
  (st->varn_ames && st->var_names[j] ? st->var_names[j] : "")

/* ==========================================================================
 * Interface
 * ========================================================================== */

/** @brief Creates DAE structure. */
DAEStruct STCreate(sunindextype size, const uint8_t[static size],
                   const uint8_t[static size],
                   const sunindextype * [static size], char **, char **);

/** @brief Destroys DAE structure. */
void STDestroy(DAEStruct);

#endif
