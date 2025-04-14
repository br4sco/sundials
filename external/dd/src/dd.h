#ifndef _DD_H
#define _DD_H

#include <stddef.h>
#include <sundials/sundials_core.h>

#include "matrix.h"
#include "structure.h"

/* ==========================================================================
 * Types, Constants, and Macro Definitions
 * ========================================================================== */

/* --------------------------------------------------------------------------
 * Constants
 * -------------------------------------------------------------------------- */

/* return values */

/** @brief Performed a successful DD pivot. */
#define DD_PIVOT_SUCCESS 3

/** @brief Failed a DD pivot.  */
#define DD_PIVOT_FAIL -200

/** @brief Generic DD error. */
#define DD_GENERIC_ERROR -300

/* --------------------------------------------------------------------------
 * Types
 * -------------------------------------------------------------------------- */

/** @brief Residual callback function. */
typedef int DDResFn(sunrealtype, N_Vector, N_Vector, void*);

/** @brief Jacobian callback function for highest order derivatives (both
    equations and variables). */
typedef int DDJacFn0(sunrealtype, N_Vector, ExtSUNMatrix[static 1], void*);

/** @brief Jacobian callback function for `DDResFn`. */
typedef int DDLsJacFn(sunrealtype, N_Vector, N_Vector, SUNMatrix, void*,
                      N_Vector, N_Vector, N_Vector);

/** @brief Forward sensitivity residual callback function. */
typedef int DDSensResFn(int Ns, sunrealtype, N_Vector, N_Vector,
                        N_Vector[static Ns], N_Vector[static Ns], void*,
                        N_Vector, N_Vector, N_Vector);

/** @brief Solver session. */
typedef struct DDMemRec* DDMem;

/** @brief Adjoint sensitivity residual callback function. */
typedef int DDResFnB(sunrealtype, N_Vector, N_Vector, N_Vector, N_Vector, void*);

/* ==========================================================================
 * Solver Interface
 * ========================================================================== */

/** @brief Creates a solver object. */
DDMem DDCreate(SUNContext);

/** @brief Frees a solver object. */
void DDFree(DDMem*);

/** @brief Initializes a solver session. */
int DDInit(DDMem, Structure[static 1], sunrealtype, DDJacFn0,
           ExtSUNMatrix[static 1], DDResFn, sunrealtype, N_Vector);

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
int DDSolveF(DDMem, sunrealtype, sunrealtype[static 1], N_Vector, int,
             int[static 1]);

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
