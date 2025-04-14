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
typedef struct
{
  sunindextype N;           /** Matrix size (we assume a square matrix). */
  sunindextype NNZ;         /** Number of non-zero elements. */
  sunindextype* colindices; /** Pointers to non-zero column indices. */
  uint8_t** rowindexptrs;   /** Pointers to the first element in each row. */
  uint8_t* data;            /** Non-zero values. */
} SigmaMatrix;

/** @brief Encodes high-index DAE structure. */
typedef struct
{
  sunindextype st_DAE_N;       /** DAE size */
  sunindextype st_DAE_N1;      /** Number of variables in first order DAE */
  sunindextype st_M;           /** Number of equation symbols */
  sunindextype st_N;           /** Number of variable symbols */
  uint8_t* st_eqnofs;          /** Equation offset vector of size `N` */
  uint8_t* st_varofs;          /** Variable offset vector of size `N` */
  sunindextype* st_acc_eqnofs; /** Accumulated equation offsets */
  sunindextype* st_acc_varofs; /** Accumulated variable offsets */
  uint8_t st_K;                /** Number of stages */
  sunindextype* st_Mk;         /** Number of equations at the k'th stage */
  sunindextype* st_Nk;         /** Number of variables at the k'th stage */
  sunindextype** st_eqns;      /** Equations at the k'th stage */
  sunindextype** st_vars;      /** Variables at the k'th stage */
  sunindextype* st_eqnsdata;   /** Equations in all stages (length `st_M`) */
  sunindextype* st_varsdata;   /** Variables in all stages (length `st_N`) */
  char** st_eqnnames;          /** Equation names (length `st_M`) */
  char** st_varnames;          /** Variable names (length `st_N`) */
} Structure;

/* ==========================================================================
 * Macros
 * ========================================================================== */

/** @brief Converts stage index to stage offset. */
#define ST_STAGE_FROM_INDEX(st, k) ((int)k - (int)st->st_K + 1)

/** @brief Derivative order of equation at specified stage index */
#define ST_EQN_ORDER(st, k, i) \
  (ST_STAGE_FROM_INDEX(st, k) + (int)st->st_eqnofs[i])

/** @brief Derivative order of variable at specified stage index */
#define ST_VAR_ORDER(st, k, j) \
  (ST_STAGE_FROM_INDEX(st, k) + (int)st->st_varofs[j])

/** @brief Return the name given to an equation or an empty string if no names
 * was given. */
#define ST_EQN_NAME(st, i) \
  (st->st_eqnnames && st->st_eqnnames[i] ? st->st_eqnnames[i] : "")

/** @brief Return the name given to a variale or an empty string if no names
 * was given. */
#define ST_VAR_NAME(st, j) \
  (st->st_varnames && st->st_varnames[j] ? st->st_varnames[j] : "")

/* ==========================================================================
 * Interface
 * ========================================================================== */

/** @brief Creates DAE structure. */
Structure* STCreate(sunindextype size, const uint8_t[static size],
                    const uint8_t[static size], char**, char**);

/** @brief Destroys DAE structure. */
void STDestroy(Structure*);

#endif
