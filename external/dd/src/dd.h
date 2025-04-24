#ifndef _DD_H
#define _DD_H

#include <stddef.h>
#include <sundials/sundials_core.h>

#include "matrix.h"
#include "structure.h"
#include "sundials/sundials_errors.h"
#include "sunmatrix/sunmatrix_sparse.h"

/* ==========================================================================
 * Types, Constants, and Macro Definitions
 * ========================================================================== */

/* --------------------------------------------------------------------------
 * Constants
 * -------------------------------------------------------------------------- */

/* --------------------------------------------------------------------------
 * Types
 * -------------------------------------------------------------------------- */

/** @brief Residual callback function. */
typedef int DDResFn(sunrealtype, N_Vector, N_Vector, void*);

/** @brief Jacobian callback function for highest order derivatives (both
    equations and variables). */
typedef int DDJacFn0(sunrealtype, N_Vector, DDMatrix[static 1], void*);

/** @brief Jacobian callback function, version 1, for `DDResFn`. */
typedef int DDLsJacFn1(sunrealtype,
                       N_Vector,
                       N_Vector,
                       SUNMatrix,
                       void*,
                       N_Vector,
                       N_Vector,
                       N_Vector);

/** @brief Jacobian callback function, version 2, for `DDResFn`. */
typedef int DDLsJacFn2(sunrealtype,
                       sunrealtype,
                       N_Vector,
                       N_Vector,
                       SUNMatrix,
                       N_Vector,
                       void*,
                       N_Vector,
                       N_Vector,
                       N_Vector);

/** @brief Callback function for computing the a sparse Jacobian on CSC format
    column-wise. */
typedef int DDLsJacColFn_CSC(sunindextype,
                             sunrealtype,
                             N_Vector,
                             N_Vector,
                             SUNMatrix,
                             sunindextype*,
                             void*,
                             N_Vector,
                             N_Vector,
                             N_Vector);

typedef enum DDLsJacFnId
{
  DD_JAC_1,
  DD_JAC_2
} DDLsJacFnId;

/** @brief Jacobian callback function. */
typedef struct
{
  DDLsJacFnId id;

  union
  {
    DDLsJacFn1* jacfn1;
    DDLsJacFn2* jacfn2;
  } fn;
} DDLsJacFn;

/** @brief Forward sensitivity residual callback function. */
typedef int DDSensResFn(int Ns,
                        sunrealtype,
                        N_Vector,
                        N_Vector,
                        N_Vector[static Ns],
                        N_Vector[static Ns],
                        void*,
                        N_Vector,
                        N_Vector,
                        N_Vector);

/** @brief Solver session. */
typedef struct DDMemRec* DDMem;

/** @brief Adjoint sensitivity residual callback function. */
typedef int DDResFnB(sunrealtype, N_Vector, N_Vector, N_Vector, N_Vector, void*);

/* ==========================================================================
 * Sparse CSC Jacobian Callback Helper
 * ========================================================================== */

/** @brief Computes the Jacobian on sparse CSC column-wise by calling `fn` for
    each column. */
static inline int DDJacFn_CSC(sunindextype M,
                              sunindextype N,
                              const uint8_t* varofs,
                              DDLsJacColFn_CSC* fn,
                              sunrealtype t,
                              sunrealtype cj,
                              N_Vector Y,
                              N_Vector R,
                              SUNMatrix J,
                              N_Vector id,
                              void* user_data,
                              N_Vector tmp1,
                              N_Vector tmp2,
                              N_Vector tmp3)
{
  SUNFunctionBegin(J->sunctx);

  SUNCheck(SUNMatGetID(J) == SUNMATRIX_SPARSE, SUN_ERR_ARG_WRONGTYPE);
  SUNCheck(SM_SPARSETYPE_S(J) == CSC_MAT, SUN_ERR_ARG_OUTOFRANGE);
  SUNCheck((0 < M) && (M < SM_ROWS_S(J)), SUN_ERR_ARG_OUTOFRANGE);

  const sunrealtype ONE     = SUN_RCONST(1.0);
  const sunrealtype* id_arr = N_VGetArrayPointer(id);

  sunindextype j = 0, nnz = 0, row = M;

  for (sunindextype var = 0; var < N; ++var)
  {
    SM_INDEXPTRS_S(J)[j] = nnz;

    int flag = fn(j, t, Y, R, J, &nnz, user_data, tmp1, tmp2, tmp3);
    if (flag != 0) { return flag; }

    if (id_arr[j] == ONE)
    {
      SM_DATA_S(J)[nnz]      = -cj;
      SM_INDEXVALS_S(J)[nnz] = row;
      nnz += 1;
    }

    j += 1;

    for (sunindextype l = 0; l < varofs[var]; ++l)
    {
      SM_INDEXPTRS_S(J)[j] = nnz;

      int flag = fn(j, t, Y, R, J, &nnz, user_data, tmp1, tmp2, tmp3);
      if (flag != 0) { return flag; }

      if (id_arr[j - 1] == ONE)
      {
        SM_DATA_S(J)[nnz]      = ONE;
        SM_INDEXVALS_S(J)[nnz] = row;
        nnz += 1;
        row += 1;
      }

      if (id_arr[j] == ONE)
      {
        SM_DATA_S(J)[nnz]      = -cj;
        SM_INDEXVALS_S(J)[nnz] = row;
        nnz += 1;
      }

      j += 1;
    }
  }

  SM_INDEXPTRS_S(J)[SM_NP_S(J)] = nnz;

  return SUN_SUCCESS;
}

/* ==========================================================================
 * Solver Interface
 * ========================================================================== */

/** @brief Creates a solver object. */
DDMem DDCreate(SUNContext);

/** @brief Frees a solver object. */
void DDFree(DDMem*);

/** @brief Initializes a solver session. */
int DDInit(DDMem,
           Structure[static 1],
           sunrealtype,
           DDJacFn0,
           DDMatrix[static 1],
           DDResFn,
           sunrealtype,
           N_Vector);

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

/** @brief Forward sensitivitiy initialization. */
int DDSensInit(DDMem, int Ns, int, DDSensResFn, N_Vector[static Ns]);

/** @brief Re-initialize forward sensitivitiy computaiton. */
int DDSensReInit(DDMem, int, N_Vector*);

/** @brief Integrates the DAE with checkpointing. */
int DDSolveF(DDMem, sunrealtype, sunrealtype[static 1], N_Vector, int, int[static 1]);

/** @brief Calculates consistent initial values. */
int DDCalcIC(DDMem, int, sunrealtype);

/** @brief Initialises an adjoint problem. */
int DDAdjInit(DDMem, long, int);

/** @brief Frees data associated with the adjoint problem. */
void DDAdjFree(DDMem);

/** @brief Creates a backwards problem. */
int DDCreateB(DDMem, int[static 1]);

/** @brief Initialises backwards problem. */
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

/** @brief Sets the stoptime for the independent variable */
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

/** @brief Compute sensitivity tolarences based on DAE state tolerances. */
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
