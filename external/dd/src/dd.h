#ifndef _DD_H
#define _DD_H

#include <stddef.h>
#include <sundials/sundials_core.h>

#include "matrix.h"
#include "pivot.h"
#include "static_info.h"
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
typedef int DDLsJacFn1(sunrealtype t,
                       N_Vector Y,
                       N_Vector R,
                       SUNMatrix J,
                       void* user_data,
                       N_Vector tmp1,
                       N_Vector tmp2,
                       N_Vector tmp3);

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
typedef int DDLsJacFn2(const sunindextype yy_diff_alias_row[static 1],
                       const sunindextype yp_diff_alias_row[static 1],
                       sunrealtype t,
                       sunrealtype cj,
                       N_Vector Y,
                       N_Vector R,
                       SUNMatrix J,
                       void* user_data,
                       N_Vector tmp1,
                       N_Vector tmp2,
                       N_Vector tmp3);

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
typedef int DDLsJacColFn_CSC(sunindextype j,
                             sunrealtype t,
                             N_Vector Y,
                             N_Vector R,
                             SUNMatrix J,
                             sunindextype* nnz,
                             void* user_data,
                             N_Vector tmp1,
                             N_Vector tmp2,
                             N_Vector tmp3);

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
 * @param[in] M is the number of rows that `fn` fills per column.
 * @param[in] fn computes the the j'th M × 1 column of the Jacobian.
 * @param[in] yy_diff_alias_row indicates if it for j holds that `yy[j] = yp[k]`
 *                              for some k in the first-order view of the DAE.
 * @param[in] yp_diff_alias_row indicates if it for j holds that `yy[k] = yp[j]`
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
static inline int DDJacFn_CSC(sunindextype M,
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
                              N_Vector tmp3)
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
typedef int DDSensResFn(int Ns,
                        sunrealtype t,
                        N_Vector Y,
                        N_Vector R,
                        N_Vector YS[static Ns],
                        N_Vector RS[static Ns],
                        void* user_data,
                        N_Vector tmp1,
                        N_Vector tmp2,
                        N_Vector tmp3);

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
typedef int DDResFnB(sunrealtype t,
                     N_Vector Y,
                     N_Vector yB,
                     N_Vector ypB,
                     N_Vector rB,
                     void* user_data);

/* ==========================================================================
 * Solver Interface
 * ========================================================================== */

/** @brief Solver session. */
typedef struct DDMemRec* DDMem;

/** @brief Creates a solver object. @see IDACreate */
DDMem DDCreate(SUNContext sunctx);

/** @brief Frees a solver object. @see IDAFree */
void DDFree(DDMem* dd_mem);

/**
 * @brief Initializes a solver session.
 *
 * @param[in] dd_mem  Solver object created by DDCreate().
 * @param[in] si      Static DAE info.
 * @param[in] resfn   DAE residual callback.
 * @param[in] spec    Initial dummy derivative specification (array of length N);
 *                    see DDSetSpec() for the encoding.
 * @param[in] t0      Initial value of the independent variable.
 * @param[in] Y0      Initial augmented state vector (length N_all_orders).
 *
 * @return IDA_SUCCESS or an IDA error code.
 * @see IDAInit
 */
int DDInit(DDMem dd_mem,
           DDStaticInfo si,
           DDResFn* resfn,
           uint8_t* spec,
           sunrealtype t0,
           N_Vector Y0);

/**
 * @brief Re-initializes a solver session with new initial conditions.
 *
 * @param[in] dd_mem  Solver object.
 * @param[in] t0      New initial value of the independent variable.
 * @param[in] Y0      New initial augmented state vector (length N_all_orders).
 *
 * @return IDA_SUCCESS or an IDA error code.
 * @see IDAReInit
 */
int DDReInit(DDMem dd_mem, sunrealtype t0, N_Vector Y0);

/**
 * @brief Integrates the DAE to a requested output time.
 *
 * @param[in]  dd_mem  Solver object.
 * @param[in]  tout    Output time to integrate towards.
 * @param[out] tret    Actual time reached.
 * @param[out] Y       Augmented state vector at tret.
 * @param[in]  itask   Task flag: IDA_NORMAL to advance to tout, or
 *                     IDA_ONE_STEP to take a single internal step.
 *
 * @return IDA_SUCCESS, IDA_TSTOP_RETURN, or an IDA error code.
 * @see IDASolve
 */
int DDSolve(DDMem dd_mem,
            sunrealtype tout,
            sunrealtype tret[static 1],
            N_Vector Y,
            int itask);

/**
 * @brief Updates the dummy derivative specification and re-initialises the
 *        solver to reflect the new variable classification.
 *
 * @param[in] dd_mem  Solver object.
 * @param[in] spec    Dummy derivative specification (array of length N).
 *                    spec[i] = d introduces d alias equations
 *                    yᵢ⁽⁰⁾' = yᵢ⁽¹⁾, …, yᵢ⁽ᵈ⁻¹⁾' = yᵢ⁽ᵈ⁾, making
 *                    yᵢ⁽⁰⁾…yᵢ⁽ᵈ⁻¹⁾ true state variables and yᵢ⁽ᵈ⁾ the
 *                    dummy derivative. spec[i] = 0 means yᵢ is fully
 *                    algebraically determined.
 *
 * @return SUN_SUCCESS or an error code.
 */
int DDSetSpec(DDMem dd_mem, uint8_t* spec);

/** @brief Frees forward sensitivity data. @see IDASensFree */
void DDSensFree(DDMem dd_mem);

/**
 * @brief Initializes forward sensitivity computation.
 *
 * @param[in] dd_mem     Solver object.
 * @param[in] Ns         Number of sensitivity vectors (i.e. number of parameters).
 * @param[in] ism        Sensitivity method: IDA_SIMULTANEOUS or IDA_STAGGERED.
 * @param[in] sensresfn  Sensitivity residual callback, or NULL to use the
 *                       default difference-quotient approximation.
 * @param[in] yS0        Initial sensitivity vectors (length Ns, each of length
 *                       N_all_orders).
 *
 * @return IDA_SUCCESS or an IDA error code.
 * @see IDASensInit
 */
int DDSensInit(DDMem dd_mem,
               int Ns,
               int ism,
               DDSensResFn* sensresfn,
               N_Vector yS0[static Ns]);

/**
 * @brief Re-initializes forward sensitivity computation.
 *
 * @param[in] dd_mem  Solver object.
 * @param[in] ism     Sensitivity method: IDA_SIMULTANEOUS or IDA_STAGGERED.
 * @param[in] yS0     New initial sensitivity vectors.
 *
 * @return IDA_SUCCESS or an IDA error code.
 * @see IDASensReInit
 */
int DDSensReInit(DDMem dd_mem, int ism, N_Vector* yS0);

/**
 * @brief Integrates the DAE forward with check-pointing for adjoint sensitivity.
 *
 * @param[in]  dd_mem  Solver object.
 * @param[in]  tout    Output time to integrate towards.
 * @param[out] tret    Actual time reached.
 * @param[out] Y       Augmented state vector at tret.
 * @param[in]  itask   Task flag: IDA_NORMAL or IDA_ONE_STEP.
 * @param[out] ncheck  Number of check-points stored so far.
 *
 * @return IDA_SUCCESS, IDA_TSTOP_RETURN, or an IDA error code.
 * @see IDASolveF
 */
int DDSolveF(DDMem dd_mem,
             sunrealtype tout,
             sunrealtype tret[static 1],
             N_Vector Y,
             int itask,
             int ncheck[static 1]);

/**
 * @brief Computes consistent initial values for the augmented DAE.
 *
 * @param[in] dd_mem  Solver object.
 * @param[in] icopt   IDA_YA_YDP_INIT to compute the algebraic components of Y
 *                    and all components of Yp, or IDA_Y_INIT to compute all
 *                    components of Y. Algebraic vs. differential is determined
 *                    by the current DD specification; see DDSetSpec().
 * @param[in] tout1   First output time, used to estimate the scale of t.
 *
 * @return IDA_SUCCESS or an IDA error code.
 * @see IDACalcIC
 */
int DDCalcIC(DDMem dd_mem, int icopt, sunrealtype tout1);

/**
 * @brief Initializes adjoint sensitivity computation.
 *
 * @param[in] dd_mem  Solver object.
 * @param[in] steps   Number of integration steps between check-points.
 * @param[in] interp  Interpolation type: IDA_POLYNOMIAL or IDA_HERMITE.
 *
 * @return IDA_SUCCESS or an IDA error code.
 * @see IDAAdjInit
 */
int DDAdjInit(DDMem dd_mem, long steps, int interp);

/** @brief Frees adjoint sensitivity data. @see IDAAdjFree */
void DDAdjFree(DDMem dd_mem);

/** @brief Creates a backward problem and returns its index. @see IDACreateB */
int DDCreateB(DDMem dd_mem, int indexB[static 1]);

/**
 * @brief Initializes a backward problem.
 *
 * @param[in] dd_mem   Solver object.
 * @param[in] indexB   Index of the backward problem returned by DDCreateB().
 * @param[in] resfnB   Adjoint residual callback.
 * @param[in] tB0      Initial time for the backward problem (typically the
 *                     final forward time).
 * @param[in] yyB0     Initial state vector for the backward problem.
 * @param[in] ypB0     Initial derivative vector for the backward problem.
 *
 * @return IDA_SUCCESS or an IDA error code.
 * @see IDAInitB
 */
int DDInitB(DDMem dd_mem,
            int indexB,
            DDResFnB* resfnB,
            sunrealtype tB0,
            N_Vector yyB0,
            N_Vector ypB0);

/**
 * @brief Computes consistent initial values for a backward problem.
 *
 * @param[in]    dd_mem  Solver object.
 * @param[in]    indexB  Index of the backward problem.
 * @param[in]    tBout1  First output time, used to estimate the scale of t.
 * @param[inout] yyB     State vector; corrected values on return.
 *
 * @return IDA_SUCCESS or an IDA error code.
 * @see IDACalcICB
 */
int DDCalcICB(DDMem dd_mem, int indexB, sunrealtype tBout1, N_Vector yyB);

/**
 * @brief Integrates all backward problems towards tBout.
 *
 * @param[in] dd_mem  Solver object.
 * @param[in] tBout   Target time for the backward integration.
 * @param[in] itaskB  Task flag: IDA_NORMAL or IDA_ONE_STEP.
 *
 * @return IDA_SUCCESS or an IDA error code.
 * @see IDASolveB
 */
int DDSolveB(DDMem dd_mem, sunrealtype tBout, int itaskB);

/* --------------------------------------------------------------------------
 * Setters and Getters
 * -------------------------------------------------------------------------- */

/** @brief Returns the underlying IDA memory (read-only). */
void* DDGetIDAMem(DDMem dd_mem);

/** @brief Sets the linear solver. @see IDASetLinearSolver */
int DDSetLinearSolver(DDMem dd_mem, SUNLinearSolver LS, SUNMatrix J);

/** @brief Sets scalar absolute and relative tolerances. @see IDASStolerances */
int DDSSTolerances(DDMem dd_mem, sunrealtype reltol, sunrealtype abstol);

/** @brief Sets the stop time. @see IDASetStopTime */
int DDSetStopTime(DDMem dd_mem, sunrealtype tstop);

/** @brief Sets the user data pointer passed to callbacks. @see IDASetUserData */
int DDSetUserData(DDMem dd_mem, void* user_data);

/** @brief Returns forward sensitivity vectors at the current time. @see IDAGetSens */
int DDGetSens(DDMem dd_mem, sunrealtype* tret, N_Vector* yS);

/** @brief Returns the name of an IDA return flag. Caller must free the result. @see IDAGetReturnFlagName */
char* DDGetReturnFlagName(long int flag);

/** @brief Returns the corrected initial state vector. @see IDAGetConsistentIC */
int DDGetConsistentIC(DDMem dd_mem, N_Vector yy0_mod);

/** @brief Returns the corrected initial sensitivity vectors. @see IDAGetSensConsistentIC */
int DDGetSensConsistentIC(DDMem dd_mem, N_Vector* yyS0_mod);

/** @brief Sets sensitivity parameters. @see IDASetSensParams */
int DDSetSensParams(DDMem dd_mem, sunrealtype* p, sunrealtype* pbar, int* plist);

/** @brief Estimates sensitivity tolerances from state tolerances. @see IDASensEEtolerances */
int DDSensEEtolerances(DDMem dd_mem);

/** @brief Sets the linear solver for a backward problem. @see IDASetLinearSolverB */
int DDSetLinearSolverB(DDMem dd_mem, int indexB, SUNLinearSolver LS, SUNMatrix J);

/** @brief Sets the Jacobian callback for a matrix-based linear solver. @see IDASetJacFn */
int DDSetJacFn(DDMem dd_mem, DDLsJacFn jacfn);

/** @brief Sets scalar tolerances for a backward problem. @see IDASStolerancesB */
int DDSStolerancesB(DDMem dd_mem,
                    int indexB,
                    sunrealtype reltol,
                    sunrealtype abstol);

/** @brief Sets the user data pointer for a backward problem. @see IDASetUserDataB */
int DDSetUserDataB(DDMem dd_mem, int indexB, void* user_data);

/** @brief Sets the differential/algebraic ID vector for a backward problem. @see IDASetIdB */
int DDSetIdB(DDMem dd_mem, int indexB, N_Vector id);

/** @brief Returns the current solution of a backward problem. @see IDAGetB */
int DDGetB(DDMem dd_mem,
           int indexB,
           sunrealtype tret[static 1],
           N_Vector yy,
           N_Vector yp);

/** @brief Returns the corrected initial values for a backward problem. @see IDAGetConsistentICB */
int DDGetConsistentICB(DDMem dd_mem,
                       int indexB,
                       N_Vector yyB0_mod,
                       N_Vector ypB0_mod);

/**
 * @brief Returns the current dummy derivative specification (length N).
 *
 * The returned pointer is owned by `dd_mem` and remains valid until the next
 * call to DDSetSpec() or DDFree(). See DDSetSpec() for the encoding.
 */
const uint8_t* DDGetSpec(DDMem dd_mem);

/**
 * @brief Returns the IDA differential/algebraic ID vector derived from the
 *        current DD specification.
 *
 * ID[j] = 1.0 for differential variables, 0.0 for algebraic variables, as
 * determined by the current spec. The returned vector is owned by `dd_mem`.
 */
N_Vector DDGetId(DDMem dd_mem);

#endif
