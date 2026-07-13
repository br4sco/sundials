#ifndef _DD_H
#define _DD_H

#include <stddef.h>
#include <sundials/sundials_core.h>

#include "static_info.h"
#include "sundials/sundials_nvector.h"

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
 * Should compute F(t,Y) = 0, where t is the independent variable and Y
 * contains the dependent variables and their derivatives.
 *
 * Given a structural analysis c,d ∈ ℕ[n], scalar residual expressions
 * e ∈ ℝ[n], and dependent variables y ∈ ℝ[n] of a possibly high-order,
 * high-index DAE, this function should express the DAE:
 *
 * F(t, Y) = (𝓓(0)e[0], 𝓓(1)e[0], … , 𝓓(c[0])e[0],
 *               ⋮
 *            𝓓(0)e[n], 𝓓(1)e[n], … , 𝓓(c[n])e[n]),
 *
 * where 𝓓(k) = dᵏ/dtᵏ and
 *
 * Y = (𝓓(0)y[0], 𝓓(1)y[0], … , 𝓓(d[0])y[0],
 *        ⋮
 *      𝓓(0)y[n], 𝓓(1)y[n], … , 𝓓(d[n])y[n]).
 *
 * Each 𝓓(k)e[j] may appear at any row of R; the index of 𝓓(k)y[j] in Y
 * is `var_deriv_chains[j][k]` (see `DDStaticInfoCreate`).
 *
 * @param[in] t is the independent variable.
 * @param[in] Y are the dependent variables and their derivatives.
 * @param[out] R holds the values of F(t,Y).
 * @param[inout] user_data points to user-defined data.
 *
 * @return a value `0` on success, a positive value if a recoverable error
 *         occurred and a negative value if a non-recoverable error occurred.
 */
typedef int DDResFn(sunrealtype t, N_Vector Y, N_Vector R, void* user_data);

/**
 * @brief Forward quadrature right-hand side callback function.
 *
 * Computes the integrand f_Q(t,Y) of the forward quadrature ODE
 * dyQ/dt = f_Q(t,Y), co-integrated alongside the DAE without affecting its
 * error control. The integrated value yQ(t) = ∫ f_Q(t,Y) dt is retrieved
 * via `DDGetQuad`.
 *
 * @param[in] t is the independent variable.
 * @param[in] Y are the dependent variables and their derivatives.
 * @param[out] rrQ holds the values of f_Q(t,Y).
 * @param[inout] user_data points to user-defined data.
 *
 * @return a value `0` on success, a positive value if a recoverable error
 *         occurred and a negative value if a non-recoverable error occurred.
 *
 * @see IDAQuadRhsFn
 */
typedef int DDQuadRhsFn(sunrealtype t, N_Vector Y, N_Vector rrQ, void* user_data);

/**
 * @brief Jacobian callback function, type 1, for `DDResFn`. Supports matrix
 * type `SUNMATRIX_DENSE` and `SUNMATRIX_SPARSE` of kind `CSR_MAT`.
 *
 * Should compute the m×n Jacobian for the residual F(t,Y) ∈ ℝ[m] w.r.t.
 * Y ∈ ℝ[n].
 *
 * @param[in] t is the independent variable.
 * @param[in] Y are the dependent variables and their derivatives.
 * @param[in] R holds the values of F(t,Y).
 *
 * @param[inout] J holds the values of the n×n Jacobian. Only non-zero values
 *                 need to be written to `J` and only rows 0 to m-1 should be
 *                 updated. Rows m to n-1 hold alias equation values determined
 *                 by the current dummy derivative specification (see
 *                 `DDSetSpec`).
 *
 * @param[inout] user_data points to user-defined data.
 * @param[inout] tmp1 user controlled workspace data.
 * @param[inout] tmp2 user controlled workspace data.
 * @param[inout] tmp3 user controlled workspace data.
 *
 * @return a value `0` on success, a positive value if a recoverable error
 *         occurred and a negative value if a non-recoverable error occurred.
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
 *
 * F(t,Y) ∈ ℝ[m] is implemented by `DDResFn`, G(Y,Yp) ∈ ℝ[n-m],
 * Y,Yp ∈ ℝ[n], and cj ∈ ℝ. Yp holds the derivatives of Y in a
 * first-order view of the DAE.
 *
 * Each scalar residual of G has the form Y[j] - Yp[k] = 0 for some
 * j,k ∈ {0...n-1}, where Y[j] and Yp[k] denote the same dependent
 * variable at the same differentiation order. The structure of G is
 * determined by the current dummy derivative specification (see
 * `DDSetSpec`). For such an equation at row r, column j of J has entry
 * 1 at row r and column k of J has entry -cj at row r.
 *
 * @param[in] yy_diff_alias_row for each column j, `yy_diff_alias_row[j]`
 *                              is the row r of the alias equation
 *                              G[r] = Y[k] - Yp[j] for some k, or -1 if
 *                              no such equation exists. Column j of J has
 *                              entry -cj at row r.
 *
 * @param[in] yp_diff_alias_row for each column j, `yp_diff_alias_row[j]`
 *                              is the row r of the alias equation
 *                              G[r] = Y[j] - Yp[k] for some k, or -1 if
 *                              no such equation exists. Column j of J has
 *                              entry 1 at row r.
 *
 * @param[in] t is the independent variable.
 * @param[in] cj is proportional to the inverse of the step-size.
 * @param[in] Y are the dependent variables and their derivatives.
 * @param[in] R holds the values of F(t,Y).
 * @param[out] J holds the values of the n×n Jacobian. Only non-zero values
 *               need to be written to `J`.
 *
 * @param[inout] user_data points to user-defined data.
 * @param[inout] tmp1 user controlled workspace data.
 * @param[inout] tmp2 user controlled workspace data.
 * @param[inout] tmp3 user controlled workspace data.
 *
 * @return a value `0` on success, a positive value if a recoverable error
 *         occurred and a negative value if a non-recoverable error occurred.
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
 * @brief Callback for computing a sparse Jacobian column by column for
 * `DDResFn` in CSC format.
 *
 * Should compute the j-th column of the m×n Jacobian for the residual
 * F(t,Y) ∈ ℝ[m] w.r.t. Y ∈ ℝ[n].
 *
 * @param[in] j is the column to compute. Called with j = 0..n-1 in
 *              increasing order.
 *
 * @param[in] t is the independent variable.
 * @param[in] Y are the dependent variables and their derivatives.
 * @param[in] R holds the values of F(t,Y).
 * @param[out] J holds the sparse Jacobian in CSC format. The callback
 *               should append at most m non-zero entries for column j
 *               starting at index `*nnz` in J's data and index arrays,
 *               and update `*nnz` accordingly.
 *
 * @param[inout] nnz on entry, the number of non-zero elements written to
 *                   `J` by previous calls; on return, the updated count
 *                   after appending the entries for column j.
 *
 * @param[inout] user_data points to user-defined data.
 * @param[inout] tmp1 user controlled workspace data.
 * @param[inout] tmp2 user controlled workspace data.
 * @param[inout] tmp3 user controlled workspace data.
 *
 * @return a value `0` on success, a positive value if a recoverable error
 *         occurred and a negative value if a non-recoverable error occurred.
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
 * @brief Assembles the n×n sparse CSC Jacobian by calling `fn` for each
 * column and appending alias equation entries.
 *
 * For each column j = 0..n-1, calls `fn` to fill rows 0 to M-1, then
 * appends the alias equation Jacobian entries for rows M..n-1 from
 * `yy_diff_alias_row` and `yp_diff_alias_row`.
 *
 * @param[in] M is the number of user equation rows that `fn` fills per
 *              column (i.e. the number of rows in F).
 * @param[in] fn computes the j-th M × 1 column of the user Jacobian.
 * @param[in] yy_diff_alias_row for each column j, `yy_diff_alias_row[j]`
 *                              is the row r of the alias equation
 *                              G[r] = Y[k] - Yp[j] for some k, or -1 if
 *                              no such equation exists. Column j of J gets
 *                              entry -cj at row r.
 * @param[in] yp_diff_alias_row for each column j, `yp_diff_alias_row[j]`
 *                              is the row r of the alias equation
 *                              G[r] = Y[j] - Yp[k] for some k, or -1 if
 *                              no such equation exists. Column j of J gets
 *                              entry 1 at row r.
 *
 * @param[in] t is the independent variable.
 * @param[in] cj is proportional to the inverse of the step-size.
 * @param[in] Y are the dependent variables and their derivatives.
 * @param[in] R holds the values of F(t,Y).
 * @param[out] J holds the values of the n×n Jacobian. This matrix must be
 *               of type `SUNMATRIX_SPARSE` and of kind `CSC_MAT`.
 *
 * @param[inout] user_data points to user-defined data.
 * @param[inout] tmp1 user controlled workspace data.
 * @param[inout] tmp2 user controlled workspace data.
 * @param[inout] tmp3 user controlled workspace data.
 *
 * @return a value `0` on success, a positive value if a recoverable error
 *         occurred and a negative value if a non-recoverable error occurred.
 */
int DDJacFn_CSC(sunindextype M,
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
                N_Vector tmp3);

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
 * @param[in] Y are the dependent variables and their derivatives.
 * @param[in] R holds the values of F(t,Y).
 * @param[in] YS are the sensitivities of `Y` (i.e. `YS[i] = dY/dp[i]`).
 * @param[out] RS holds the sensitivity residual values, where RS[i] is the
 *                residual w.r.t. the i-th parameter.
 *
 * @param[inout] user_data points to user-defined data.
 * @param[inout] tmp1 user controlled workspace data.
 * @param[inout] tmp2 user controlled workspace data.
 * @param[inout] tmp3 user controlled workspace data.
 *
 * @return a value `0` on success, a positive value if a recoverable error
 *         occurred and a negative value if a non-recoverable error occurred.
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
 * @brief First-order backwards problem residual callback function.
 *
 * The backwards problem is assumed to have at most differential index 1 or
 * differential index 2 if it is semi-explicit. More concretely, its system
 * Jacobian
 *
 * ∂F_B/∂yyB + α ⋅ ∂F_B/∂ypB
 *
 * has to be regular, where α > 0 is a solver supplied scale factor.
 *
 * Unlike the forward problem, backward integration does not involve any
 * dynamic state changes.
 *
 * Given a structural analysis c,d ∈ ℕ[n] and scalar residual expressions
 * e ∈ ℝ[n], this function can express adjoint DAEs for the complete
 * first-order forward system
 *
 * ```
 * F(t, Y, Ẏ) = | H(t, Y) |
 *              | G(Y, Ẏ) |,
 * ```
 *
 * where G(Y, Ẏ) ∈ ℝ[n-m] are the order-reducing alias equations (see
 * `DDLsJacFn2`), Ẏ = dY/dt, and
 *
 * H(t, Y) = (𝓓(c[0])e[0], … , 𝓓(c[n])e[n]), where 𝓓(k) = dᵏ/dtᵏ.
 *
 * @par First adjoint DAE
 * For computing parameter sensitivities of the functional ∫ g(t,Y,p) dt,
 * where g = g(t,Y,p) and p are the parameters, set
 *
 * rrB = (∂F/∂Ẏ)^* ypB - (∂F/∂Y)^* yyB + (∂g/∂Y)^*,
 *
 * where * denotes conjugate transpose and ∂F/∂Ẏ is constant.
 *
 * @par Second adjoint DAE
 * For computing parameter sensitivities of g(T,p), where T is the final
 * time of the forward problem, set
 *
 * rrB = (∂F/∂Ẏ)^* ypB - (∂F/∂Y)^* yyB.
 *
 * @param[in] t is the independent variable.
 * @param[in] Y are the dependent variables and their derivatives.
 * @param[in] yyB are the dependent variables of the backwards DAE.
 * @param[in] ypB are the derivatives of `yyB`, i.e., `d/dt yyB = ypB`.
 * @param[out] rrB holds the values of the adjoint residual.
 * @param[inout] user_dataB points to user-defined data.
 *
 * @return a value `0` on success, a positive value if a recoverable error
 *         occurred and a negative value if a non-recoverable error occurred.
 *
 * @see IDAResFnB
 */
typedef int DDResFnB(sunrealtype t,
                     N_Vector Y,
                     N_Vector yyB,
                     N_Vector ypB,
                     N_Vector rrB,
                     void* user_dataB);

/**
 * @brief Jacobian callback function for the backward problem (see
 * `IDALsJacFnB`).
 *
 * Should compute the Jacobian
 *
 * JB = ∂F_B/∂yyB + cj ⋅ ∂F_B/∂ypB,
 *
 * where F_B is the backward residual implemented by `DDResFnB` and
 * cj ∈ ℝ is a solver-supplied scale factor.
 *
 * @param[in] t is the independent variable.
 * @param[in] cj is proportional to the inverse of the step-size.
 * @param[in] Y are the forward dependent variables and their derivatives.
 * @param[in] yyB are the dependent variables of the backward DAE.
 * @param[in] ypB are the derivatives of `yyB`, i.e., `d/dt yyB = ypB`.
 * @param[in] rrB holds the values of the backward residual F_B.
 * @param[out] JB holds the values of the Jacobian. Only non-zero values
 *               need to be written to `JB`.
 * @param[inout] user_dataB points to user-defined data.
 * @param[inout] tmp1 user controlled workspace data.
 * @param[inout] tmp2 user controlled workspace data.
 * @param[inout] tmp3 user controlled workspace data.
 *
 * @return a value `0` on success, a positive value if a recoverable error
 *         occurred and a negative value if a non-recoverable error occurred.
 */
typedef int DDLsJacFnB(sunrealtype t,
                       sunrealtype cj,
                       N_Vector Y,
                       N_Vector yyB,
                       N_Vector ypB,
                       N_Vector rrB,
                       SUNMatrix JB,
                       void* user_dataB,
                       N_Vector tmp1,
                       N_Vector tmp2,
                       N_Vector tmp3);

/**
 * @brief Quadrature right-hand side callback for the backward problem (see
 * `IDAQuadRhsFnB`).
 *
 * Computes the integrand of the backward quadrature ODE. The integrated
 * values are retrieved via `DDGetQuadB`.
 *
 * @par Example: first adjoint quadrature
 * For computing parameter sensitivities of the functional ∫ g(t,Y,p) dt,
 * where yyB solves the first adjoint DAE (see `DDResFnB`), set
 *
 * rhsBQ = (∂g/∂p)^* - (∂F/∂p)^* yyB,
 *
 * where * denotes conjugate transpose and F is the complete first-order
 * forward system (see `DDResFnB`).
 *
 * @param[in] t is the independent variable.
 * @param[in] Y are the forward dependent variables and their derivatives.
 * @param[in] yyB are the dependent variables of the backward DAE.
 * @param[in] ypB are the derivatives of `yyB`, i.e., `d/dt yyB = ypB`.
 * @param[out] rhsBQ holds the values of the quadrature right-hand side.
 * @param[inout] user_dataB points to user-defined data.
 *
 * @return a value `0` on success, a positive value if a recoverable error
 *         occurred and a negative value if a non-recoverable error occurred.
 */
typedef int DDQuadRhsFnB(sunrealtype t,
                         N_Vector Y,
                         N_Vector yyB,
                         N_Vector ypB,
                         N_Vector rhsBQ,
                         void* user_dataB);

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

/**
 * @brief Initializes forward ("pure") quadrature integration.
 *
 * @param[in] dd_mem  Solver object.
 * @param[in] rhsQ    Quadrature right-hand side callback.
 * @param[in] yQ0     Initial value of the quadrature vector.
 *
 * @return IDA_SUCCESS or an IDA error code.
 * @see IDAQuadInit
 */
int DDQuadInit(DDMem dd_mem, DDQuadRhsFn rhsQ, N_Vector yQ0);

/**
 * @brief Re-initializes forward quadrature integration with new initial values.
 *
 * @param[in] dd_mem  Solver object.
 * @param[in] yQ0     New initial value of the quadrature vector.
 *
 * @return IDA_SUCCESS or an IDA error code.
 * @see IDAQuadReInit
 */
int DDQuadReInit(DDMem dd_mem, N_Vector yQ0);

/** @brief Frees forward quadrature integration data. @see IDAQuadFree */
void DDQuadFree(DDMem dd_mem);

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
 * Additional Setters and Getters
 * -------------------------------------------------------------------------- */

/** @brief Returns the underlying IDA memory (read-only). */
void* DDGetIDAMem(DDMem dd_mem);

/** @brief Sets the linear solver. @see IDASetLinearSolver */
int DDSetLinearSolver(DDMem dd_mem, SUNLinearSolver LS, SUNMatrix J);

/** @brief Sets the Jacobian callback for a matrix-based linear solver. @see IDASetJacFn */
int DDSetJacFn(DDMem dd_mem, DDLsJacFn jacfn);

/** @brief Sets scalar absolute and relative tolerances. @see IDASStolerances */
int DDSSTolerances(DDMem dd_mem, sunrealtype reltol, sunrealtype abstol);

/** @brief Sets the stop time. @see IDASetStopTime */
int DDSetStopTime(DDMem dd_mem, sunrealtype tstop);

/** @brief Sets the user data pointer passed to callbacks. @see IDASetUserData */
int DDSetUserData(DDMem dd_mem, void* user_data);

/** @brief Returns the quadrature variables at the current time. @see IDAGetQuad */
int DDGetQuad(DDMem dd_mem, sunrealtype* tret, N_Vector yQ);

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

/** @brief Sets the Jacobian callback for a backward problem. @see IDASetJacFnB */
int DDSetJacFnB(DDMem dd_mem, int indexB, DDLsJacFnB jacfn);

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

/** @brief Initializes quadrature integration for a backward problem. @see IDAQuadInitB */
int DDQuadInitB(DDMem dd_mem, int indexB, DDQuadRhsFnB rhsQB, N_Vector yQB0);

/** @brief Returns the quadrature variables of a backward problem. @see IDAGetQuadB */
int DDGetQuadB(DDMem dd_mem, int indexB, sunrealtype* tret, N_Vector yQB);

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
