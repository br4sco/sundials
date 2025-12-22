#ifndef _DD_H
#define _DD_H

#include <stddef.h>
#include <sundials/sundials_core.h>

#include "matrix.h"
#include "structure.h"
#include "sundials/sundials_errors.h"
#include "sundials/sundials_nvector.h"
#include "sunmatrix/sunmatrix_sparse.h"

/* ==========================================================================
 * Types, Constants, and Macro Definitions
 * ========================================================================== */

/* --------------------------------------------------------------------------
 * Constants
 * -------------------------------------------------------------------------- */

/* --------------------------------------------------------------------------
 * Callback Functions
 * -------------------------------------------------------------------------- */

/**
 * @brief DAE residual callback function.
 *
 * Should compute F(t,Y) = 0, where t is the independent variable and Y are the
 * dependent variables and their derivative.
 *
 * The residual function F includes the scalar residuals of, a possibly
 * high-index DAE, and their derivatives w.r.t. t according to the
 * structural analysis of the DAE.
 *
 * @param[in] t is the independent variable.
 * @param[in] Y are the dependent variables and their derivatives.
 * @param[out] R holds the residual values and their derivatives.
 * @param[inout] user_data points to user-defined data.
 *
 * @return a value `0` on success, a positive values if a recoverable error
 *         occurred and a negative value of a non-recoverable error occurred.
 */
typedef int DDResFn(sunrealtype t, N_Vector Y, N_Vector R, void* user_data);

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
typedef int DDJacFn0(sunrealtype t, N_Vector Y, SUNMatrix J, void* user_data);

/**
 * @brief Jacobian callback function, type 1, for `DDResFn`. Supports matrix
 * type `SUNMATRIX_DENSE` and `SUNMATRIX_SPARSE` of kind `CSR_MAT`.
 *
 * Should compute the m×n Jacobian for the residual F(t,Y) ∈ ℝ[m] w.r.t.
 * Y ∈ ℝ[n].
 *
 * @param[in] t is the independent variable.
 * @param[in] Y are the dependent variables and their derivatives.
 * @param[in] R holds the residual values and their derivatives.
 *
 * @param[inout] J holds the values of the n×n Jacobian. Only non-zero values
 *                 needs to be written to `J` and only rows 0 to m-1 should be
 *                 updated. Rows m to n-1 holds values computed by the solver.
 *
 * @param[inout] user_data points to user-defined data.
 * @param[inout] tmp1 user controlled workspace data.
 * @param[inout] tmp2 user controlled workspace data.
 * @param[inout] tmp3 user controlled workspace data.
 *
 * @return a value `0` on success, a positive values if a recoverable error
 *         occurred and a negative value of a non-recoverable error occurred.
 */
typedef int DDLsJacFn1(
  sunrealtype t,
  N_Vector Y,
  N_Vector R,
  SUNMatrix J,
  void* user_data,
  N_Vector tmp1,
  N_Vector tmp2,
  N_Vector tmp3
);

/**
 * @brief Jacobian callback function, type 2, for `DDResFn`.
 *
 * Should compute the n×n Jacobian dH/dY + cj⋅dH/dYp for the residual
 * H(t,Y,Yp) ∈ ℝ[n], where:
 *
 * ```
 * H(t,Y,Yp) = | F(t,Y)  |
 *             | G(Y,Yp) |,
 * ```
 * F(t,Y) ∈ ℝ[m] is implemented by `DDResFn`, G(Y,Yp) ∈ ℝ[n-m], Y,Yp ∈ ℝ[n], and
 * cj ∈ ℝ. The vector Yp holds the derivatives of Y in a first-order view of the
 * DAE.
 *
 * Given id ∈ {0,1}[n], which indicates that variable j is a differential
 * variable in the first-order DAE if id[j] = 1, the residual G computes
 * ```
 * Y[j]  - Yp[k],
 * ```
 * for j,k ∈ {1...n}, assuming Y[j] and Yp[k] denotes the same dependent variable
 * at the same differentiation order. Hence, assuming j is the first such
 * variable, then J[m,j] = 1 and J[m,k] = -cj.
 *
 * @param[in] yy_diff_alias_row indicates if it for j holds that `yy[j] = yp[k]`
 *                              for some k in the first-order view of the DAE.
 *
 * @param[in] yy_diff_alias_row indicates if it for j holds that `yy[k] = yp[j]`
 *                              for some k in the first-order view of the DAE.
 *
 * @param[in] t is the independent variable.
 * @param[in] cj is proportional to the inverse of the step-size.
 * @param[in] Y are the dependent variables and their derivatives
 * @param[in] R holds the residual values and their derivatives.
 * @param[out] J holds the values of the n×n Jacobian. Only non-zero values
 *               needs to be written to `J`.
 *
 * @param[inout] user_data points to user-defined data.
 * @param[inout] tmp1 user controlled workspace data.
 * @param[inout] tmp2 user controlled workspace data.
 * @param[inout] tmp3 user controlled workspace data.
 *
 * @return a value `0` on success, a positive values if a recoverable error
 *         occurred and a negative value of a non-recoverable error occurred.
 */
typedef int DDLsJacFn2(
  const sunindextype yy_diff_alias_row[static 1],
  const sunindextype yp_diff_alias_row[static 1],
  sunrealtype t,
  sunrealtype cj,
  N_Vector Y,
  N_Vector R,
  SUNMatrix J,
  void* user_data,
  N_Vector tmp1,
  N_Vector tmp2,
  N_Vector tmp3
);

/**
 * @brief Callback function for column-wise computing the a sparse Jacobian for
 * `DDResFn` on CSC format.
 *
 * Should compute the j'th column of the m×n Jacobian for the residual
 * F(t,Y) ∈ ℝ[m] w.r.t. Y ∈ ℝ[n].
 *
 * @param[in] j is the column to compute. This function is called with
 *              `j = 0..n-1` with increasing `j`.
 *
 * @param[in] t is the independent variable.
 * @param[in] Y are the dependent variables and their derivatives
 * @param[in] R holds the residual values and their derivatives.
 * @param[out] J holds the values of the n×n Jacobian. Only non-zero values
 *               needs to be written to `J` and only the first m elements in the
 *               j'th column should be updated.
 *
 * @param[inout] nnz holds the number of non-zero elements in `J` from previous
 *                   calls to this function and should hold the number of
 *                   non-zero elements of `J` after the call returns.
 *
 * @param[inout] user_data points to user-defined data.
 * @param[inout] tmp1 user controlled workspace data.
 * @param[inout] tmp2 user controlled workspace data.
 * @param[inout] tmp3 user controlled workspace data.
 *
 * @return a value `0` on success, a positive values if a recoverable error
 *         occurred and a negative value of a non-recoverable error occurred.
 */
typedef int DDLsJacColFn_CSC(
  sunindextype j,
  sunrealtype t,
  N_Vector Y,
  N_Vector R,
  SUNMatrix J,
  sunindextype* nnz,
  void* user_data,
  N_Vector tmp1,
  N_Vector tmp2,
  N_Vector tmp3
);

/** @brief Enumerates Jacobian callback function types. */
typedef enum DDLsJacFnId
{
  DD_JAC_1, /**< type 1 callback, valid for `SUNMATRIX_DENSE` and `SUNMATRIX_SPARSE` of kind `CSR_MAT` */
  DD_JAC_2  /**< type 2 callback */
} DDLsJacFnId;

/** @brief Jacobian callback function. */
typedef struct
{
  DDLsJacFnId id;

  union
  {
    DDLsJacFn1* jacfn1; /**< type 1 jacobian callback function */
    DDLsJacFn2* jacfn2; /**< type 2 jacobian callback function  */
  } fn;
} DDLsJacFn;

/**
 * @brief Computes the Jacobian in sparse CSC format column-wise by calling `fn`
 * to compute each column.
 *
 * Assuming a structural analysis d ∈ ℕ[n] of an, possibly high-index, DAE of
 * size n.
 *
 * @param[in] fn computes the the j'th M × 1 column of the Jacobian.
 * @param[in] yy_diff_alias_row indicates if it for j holds that `yy[j] = yp[k]`
 *                              for some k in the first-order view of the DAE.
 *
 * @param[in] yy_diff_alias_row indicates if it for j holds that `yy[k] = yp[j]`
 *                              for some k in the first-order view of the DAE.
 *
 * @param[in] t is the independent variable.
 * @param[in] cj is proportional to the inverse of the step-size.
 * @param[in] Y are the dependent variables and their derivatives.
 * @param[in] R holds the residual values and their derivatives.
 * @param[out] J holds the values of the n×n Jacobian. This matrix must be of
 *               type `SUNMATRIX_SPARSE` and of kind `CSC_MAT`.
 *
 * @param[inout] user_data points to user-defined data.
 * @param[inout] tmp1 user controlled workspace data.
 * @param[inout] tmp2 user controlled workspace data.
 * @param[inout] tmp3 user controlled workspace data.
 *
 * @return a value `0` on success, a positive values if a recoverable error
 *         occurred and a negative value of a non-recoverable error occurred.
 */
static inline int DDJacFn_CSC(
  sunindextype M,
  DDLsJacColFn_CSC* fn,
  const sunindextype yy_diff_alias_row[static 1],
  const sunindextype yp_diff_alias_row[static 1],
  sunrealtype t,
  sunrealtype cj,
  N_Vector Y,
  N_Vector R,
  SUNMatrix J,
  void* user_data,
  N_Vector tmp1,
  N_Vector tmp2,
  N_Vector tmp3
)
{
  SUNFunctionBegin(J->sunctx);

  SUNCheck(SUNMatGetID(J) == SUNMATRIX_SPARSE, SUN_ERR_ARG_WRONGTYPE);
  SUNCheck(SM_SPARSETYPE_S(J) == CSC_MAT, SUN_ERR_ARG_OUTOFRANGE);
  SUNCheck((0 < M) && (M < SM_ROWS_S(J)), SUN_ERR_ARG_OUTOFRANGE);

  const sunrealtype ONE = SUN_RCONST(1.0);
  const sunindextype N  = SM_COLUMNS_S(J);

  SUNCheck(M < N, SUN_ERR_ARG_OUTOFRANGE);

  sunindextype nnz = 0;
  for (sunindextype j = 0; j < N; ++j)
  {
    SM_INDEXPTRS_S(J)[j] = nnz;
    int flag             = fn(j, t, Y, R, J, &nnz, user_data, tmp1, tmp2, tmp3);
    if (flag != 0) { return flag; }

    sunindextype yy_alias_row = yy_diff_alias_row[j];
    if (yy_alias_row >= 0)
    {
      SM_DATA_S(J)[nnz]      = -cj;
      SM_INDEXVALS_S(J)[nnz] = yy_alias_row;
      nnz += 1;
    }

    sunindextype yp_alias_row = yp_diff_alias_row[j];
    if (yp_alias_row >= 0)
    {
      SM_DATA_S(J)[nnz]      = ONE;
      SM_INDEXVALS_S(J)[nnz] = yp_alias_row;
      nnz += 1;
    }
  }

  SM_INDEXPTRS_S(J)[SM_NP_S(J)] = nnz;

  return SUN_SUCCESS;
}

/**
 * @brief Forward sensitivity residual callback function.
 *
 * For a parametrized residual F(t,Y,p) = 0 (see `DDResFn` and `IDASensResFn`),
 * with parameters p ∈ ℝ[Ns], this function should compute:
 *
 * (dF/dY)Ys[i] + (dF/dp[i]), with Ys[i] = dY/dp[i].
 *
 * @param[in] Ns is the number of parameters to compute the residual for.
 * @param[in] t is the independent variable.
 * @param[in] Y are the dependent variables and their derivatives
 * @param[in] R holds the residual values and their derivatives.
 * @param[in] YS are the sensitivities of `Y` (i.e. `YS[i] = dY/dp[i]`).
 * @param[out] RS holds the sensitivity residual values, where RS[i] is the
 *                residual w.r.t. to the i'th parameter.
 *
 * @param[inout] user_data points to user-defined data.
 * @param[inout] tmp1 user controlled workspace data.
 * @param[inout] tmp2 user controlled workspace data.
 * @param[inout] tmp3 user controlled workspace data.
 *
 * @return a value `0` on success, a positive values if a recoverable error
 *         occurred and a negative value of a non-recoverable error occurred.
 */
typedef int DDSensResFn(
  int Ns,
  sunrealtype t,
  N_Vector Y,
  N_Vector R,
  N_Vector YS[static Ns],
  N_Vector RS[static Ns],
  void* user_data,
  N_Vector tmp1,
  N_Vector tmp2,
  N_Vector tmp3
);

/**
 * @brief Adjoint sensitivity residual callback function.
 *
 * given a structural analysis c,d ∈ ℕ[n] and scalar residual expressions
 * e ∈ ℝ[n], this function should compute the first-order adjoint DAE
 * (see `IDAResFnB`) for the low-index DAE
 *
 * F(t, Y) = (𝓓(c[0])e[0], … , 𝓓(c[n])e[n]), where 𝓓(k) = dᵏ/dtᵏ.
 *
 * @param[in] t is the independent variable.
 * @param[in] Y are the dependent variables and their derivatives.
 * @param[in] yB is the dependent variables of the adjoint DAE.
 * @param[in] ypB are the derivatives of `yB`, i.e., `d/dt yB = ypB`.
 * @param[out] rB holds the values of the adjoint residual.
 * @param[inout] user_data points to user-defined data.
 *
 * @return a value `0` on success, a positive values if a recoverable error
 *         occurred and a negative value of a non-recoverable error occurred.
 */
typedef int DDResFnB(
  sunrealtype t,
  N_Vector Y,
  N_Vector yB,
  N_Vector ypB,
  N_Vector rB,
  void* user_data
);

/* ==========================================================================
 * Solver Interface
 * ========================================================================== */

/** @brief Solver session. */
typedef struct DDMemRec* DDMem;

/** @brief Creates a solver object. */
DDMem DDCreate(SUNContext);

/** @brief Frees a solver object. */
void DDFree(DDMem*);

/** @brief Initializes a solver session. */
int DDInit(DDMem, DAEStruct, sunrealtype, DDJacFn0, DDMatrix, DDResFn, sunrealtype, N_Vector);

/** @brief Re-initializes a solver session. */
int DDReInit(DDMem, sunrealtype, N_Vector);

/** @brief Integrates the DAE. */
int DDSolve(DDMem, sunrealtype, sunrealtype[static 1], N_Vector, int);

typedef enum PivotResult
{
  PIVOT_SUCCESS     = 0,
  PIVOT_UNNECESSARY = 1,
  PIVOT_FAIL        = -1
} PivotResult;

/** @brief Pivots the DAE if necessary */
PivotResult DDPivot(DDMem);

/** @brief Frees forward sensitivity related data. */
void DDSensFree(DDMem);

/** @brief Forward sensitivity initialization. */
int DDSensInit(DDMem, int Ns, int, DDSensResFn, N_Vector[static Ns]);

/** @brief Re-initialize forward sensitivity computation. */
int DDSensReInit(DDMem, int, N_Vector*);

/** @brief Integrates the DAE with check-pointing. */
int DDSolveF(DDMem, sunrealtype, sunrealtype[static 1], N_Vector, int, int[static 1]);

/** @brief Calculates consistent initial values. */
int DDCalcIC(DDMem, int, sunrealtype);

/** @brief Initializes an adjoint problem. */
int DDAdjInit(DDMem, long, int);

/** @brief Frees data associated with the adjoint problem. */
void DDAdjFree(DDMem);

/** @brief Creates a backwards problem. */
int DDCreateB(DDMem, int[static 1]);

/** @brief Initializes backwards problem. */
int DDInitB(DDMem, int, DDResFnB, sunrealtype, N_Vector, N_Vector);

/** @brief Calculate consistent initial values for the backwards problem. */
int DDCalcICB(DDMem, int, sunrealtype, N_Vector);

/** @brief Solve the backwards problems. */
int DDSolveB(DDMem, sunrealtype, int);

/* --------------------------------------------------------------------------
 * Setters and Getters
 * -------------------------------------------------------------------------- */

/** @brief Gets the associated IDA memory. This memory should not be modified.
 */
void* DDGetIDAMem(DDMem);

/** @brief Sets linear solver. */
int DDSetLinearSolver(DDMem, SUNLinearSolver, SUNMatrix);

/** @brief Set solver tolerances. */
int DDSSTolerances(DDMem, sunrealtype, sunrealtype);

/** @brief Sets the stop-time for the independent variable */
int DDSetStopTime(DDMem, sunrealtype);

/** @brief Sets user data. */
int DDSetUserData(DDMem, void*);

/** @brief Get forward mode sensitivities. */
int DDGetSens(DDMem, sunrealtype*, N_Vector*);

/** @brief Get return flag name. The caller is responsible for de-allocation. */
char* DDGetReturnFlagName(long int);

/** @brief Get consistent initial values. */
int DDGetConsistentIC(DDMem, N_Vector);

/** @brief Get consistent initial sensitivities. */
int DDGetSensConsistentIC(DDMem, N_Vector*);

/** @brief Set sensitivity parameters. */
int DDSetSensParams(DDMem, sunrealtype*, sunrealtype*, int*);

/** @brief Compute sensitivity tolerances based on DAE state tolerances. */
int DDSensEEtolerances(DDMem);

/** @brief Sets linear solver for a backwards problem. */
int DDSetLinearSolverB(DDMem, int, SUNLinearSolver, SUNMatrix);

/** @brief Sets the Jacobian function for matrix-based linear solver. */
int DDSetJacFn(DDMem, DDLsJacFn);

/** @brief Sets tolerances for a backwards problem. */
int DDSStolerancesB(DDMem, int, sunrealtype, sunrealtype);

/** @brief Sets user data for a backwards problem. */
int DDSetUserDataB(DDMem, int, void*);

/** @brief Sets ID vector for a backwards problem. */
int DDSetIdB(DDMem, int, N_Vector);

/** @brief Get a current backward solution. */
int DDGetB(DDMem, int, sunrealtype[static 1], N_Vector, N_Vector);

/** @brief Get corrected initial values for a backwards problem. */
int DDGetConsistentICB(DDMem, int, N_Vector, N_Vector);

#endif
