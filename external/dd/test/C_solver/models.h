#ifndef _DD_MODELS_H
#define _DD_MODELS_H

#include <math.h>
#include <nvector/nvector_serial.h>
#include <sundials/sundials_core.h>
#include <sundials/sundials_macros.h>
#include <sunmatrix/sunmatrix_dense.h>

#include "dd.h"
#include "sundials/sundials_nvector.h"
#include "sundials/sundials_types.h"
#include "sunmatrix/sunmatrix_sparse.h"

/**
 * @file
 * @brief DAE models.
 */

#define DAEIR_REAL          sunrealtype
#define DAEIR_INT           sunindextype
#define DAEIR_SIZE          sunindextype
#define DAEIR_REAL_CONST(x) ((DAEIR_REAL)(x))

#define ZERO   SUN_RCONST(0.0)
#define ONE    SUN_RCONST(1.0)
#define TWO    SUN_RCONST(2.0)
#define FOUR   SUN_RCONST(4.0)
#define NV_Ith NV_Ith_S

#define SET_SPARSE(A, nnz, i, expr) \
  do                                \
  {                                 \
    SM_DATA_S(A)[nnz]      = expr;  \
    SM_INDEXVALS_S(A)[nnz] = i;     \
    nnz += 1;                       \
  }                                 \
  while (0)

/* --------------------------------------------------------------------------
 * Lotka-Volterra predator-prey model (ODE)
 * -------------------------------------------------------------------------- */

static const sunindextype LOTKA_VOLTERRA_N = 2;
static const uint8_t LOTKA_VOLTERRA_C[]    = {0, 0}; /* f₁, f₂ */
static const uint8_t LOTKA_VOLTERRA_D[]    = {1, 1}; /* x', y' */

static const sunindextype* LOTKA_VOLTERRA_VAR_IDX_MAP[] =
  {(const sunindextype[]){3, 0},  /* y', x */
   (const sunindextype[]){2, 1}}; /* x', y' */

static const sunindextype LOTKA_VOLTERRA_JAC_NNZ = 6;

#define LV_Ith(Y, i, k) (NV_Ith(Y, LOTKA_VOLTERRA_VAR_IDX_MAP[i][k]))

static const uint8_t LOTKA_VOLTERRA_NP = 4;

typedef struct
{
  sunrealtype a; /* Prey per-capita growth rate. */
  sunrealtype b; /* Prey death rate. */
  sunrealtype c; /* Predator per-capita growth rate. */
  sunrealtype d; /* Preadotr death rate. */
} LotkaVolterraParams;

int LotkaVolterraRes(SUNDIALS_MAYBE_UNUSED sunrealtype t,
                     N_Vector Y, /* {x, x', y, y'} */
                     N_Vector R, /* {f₁, f₂, ...} */
                     void* user_data)
{
  const LotkaVolterraParams p = *((LotkaVolterraParams*)user_data);

  const sunrealtype x = LV_Ith(Y, 0, 0), xp = LV_Ith(Y, 0, 1),

                    y = LV_Ith(Y, 1, 0), yp = LV_Ith(Y, 1, 1);

  NV_Ith(R, 0) = xp - p.a * x + p.b * x * y;
  NV_Ith(R, 1) = yp - p.d * x * y + p.c * y;

  return 0;
}

int LotkaVolterraResS(int Ns,
                      SUNDIALS_MAYBE_UNUSED sunrealtype t,
                      N_Vector Y,
                      SUNDIALS_MAYBE_UNUSED N_Vector R,
                      N_Vector* YS,
                      SUNDIALS_MAYBE_UNUSED N_Vector* RS,
                      void* user_data,
                      SUNDIALS_MAYBE_UNUSED N_Vector tmp1,
                      SUNDIALS_MAYBE_UNUSED N_Vector tmp2,
                      SUNDIALS_MAYBE_UNUSED N_Vector tmp3)
{
  assert(Ns == LOTKA_VOLTERRA_NP);

  LotkaVolterraParams p = *((LotkaVolterraParams*)user_data);

  const sunrealtype x = LV_Ith(Y, 0, 0), y = LV_Ith(Y, 1, 0);

  for (int i = 0; i < Ns; ++i)
  {
    const sunrealtype xS = LV_Ith(YS[i], 0, 0), xpS = LV_Ith(YS[i], 0, 1),

                      yS = LV_Ith(YS[i], 1, 0), ypS = LV_Ith(YS[i], 1, 1);

    NV_Ith(RS[i], 0) = (-p.a + p.b * y) * xS + xpS + p.b * x * yS;
    NV_Ith(RS[i], 1) = -p.d * y * xS + (-p.d * x + p.c) * yS + ypS;
  }

  NV_Ith(RS[0], 0) -= x;
  NV_Ith(RS[1], 0) += x * y;
  NV_Ith(RS[2], 1) += y;
  NV_Ith(RS[3], 1) -= x * y;

  return 0;
}

int LotkaVolterraJacf0(SUNDIALS_MAYBE_UNUSED sunrealtype t,
                       SUNDIALS_MAYBE_UNUSED N_Vector Y,
                       SUNMatrix A,
                       SUNDIALS_MAYBE_UNUSED void* user_data)
{
  SM_ELEMENT_D(A, 0, 0) = ONE;
  SM_ELEMENT_D(A, 1, 1) = ONE;

  return 0;
}

int LotkaVolterraJacfn_Dense(SUNDIALS_MAYBE_UNUSED sunrealtype t,
                             N_Vector Y,
                             SUNDIALS_MAYBE_UNUSED N_Vector R,
                             SUNMatrix J,
                             void* user_data,
                             SUNDIALS_MAYBE_UNUSED N_Vector tmp1,
                             SUNDIALS_MAYBE_UNUSED N_Vector tmp2,
                             SUNDIALS_MAYBE_UNUSED N_Vector tmp3)
{
  LotkaVolterraParams p  = *((LotkaVolterraParams*)user_data);
  const sunindextype xj  = LOTKA_VOLTERRA_VAR_IDX_MAP[0][0],
                     xpj = LOTKA_VOLTERRA_VAR_IDX_MAP[0][1],
                     yj  = LOTKA_VOLTERRA_VAR_IDX_MAP[1][0],
                     ypj = LOTKA_VOLTERRA_VAR_IDX_MAP[1][1];

  const sunrealtype x = NV_Ith(Y, xj), y = NV_Ith(Y, yj);

  /* x */
  SM_ELEMENT_D(J, 0, xj) = -p.a + p.b * y;
  SM_ELEMENT_D(J, 1, xj) = -p.d * y;

  /* x' */
  SM_ELEMENT_D(J, 0, xpj) = ONE;

  /* y */
  SM_ELEMENT_D(J, 0, yj) = p.b * x;
  SM_ELEMENT_D(J, 1, yj) = -p.d * x + p.c;

  /* y' */
  SM_ELEMENT_D(J, 1, ypj) = ONE;

  return 0;
}

int LotkaVolterraJacfn_CSR(SUNDIALS_MAYBE_UNUSED sunrealtype t,
                           N_Vector Y,
                           SUNDIALS_MAYBE_UNUSED N_Vector R,
                           SUNMatrix J,
                           void* user_data,
                           SUNDIALS_MAYBE_UNUSED N_Vector tmp1,
                           SUNDIALS_MAYBE_UNUSED N_Vector tmp2,
                           SUNDIALS_MAYBE_UNUSED N_Vector tmp3)
{
  LotkaVolterraParams p = *((LotkaVolterraParams*)user_data);

  const sunindextype xj  = LOTKA_VOLTERRA_VAR_IDX_MAP[0][0],
                     xpj = LOTKA_VOLTERRA_VAR_IDX_MAP[0][1],
                     yj  = LOTKA_VOLTERRA_VAR_IDX_MAP[1][0],
                     ypj = LOTKA_VOLTERRA_VAR_IDX_MAP[1][1];

  const sunrealtype x = NV_Ith(Y, xj), y = NV_Ith(Y, yj);

  sunindextype nnz = 0;

  /* row 0 */
  SM_INDEXPTRS_S(J)[0] = nnz;
  SET_SPARSE(J, nnz, xj, -p.a + p.b * y);
  SET_SPARSE(J, nnz, xpj, ONE);
  SET_SPARSE(J, nnz, yj, p.b * x);

  /* row 1 */
  SM_INDEXPTRS_S(J)[1] = nnz;
  SET_SPARSE(J, nnz, xj, -p.d * y);
  SET_SPARSE(J, nnz, yj, -p.d * x + p.c);
  SET_SPARSE(J, nnz, ypj, ONE);

  SM_INDEXPTRS_S(J)[2] = nnz;

  return 0;
}

static inline int LotkaVolterraJacColFn_CSC(sunindextype j,
                                            SUNDIALS_MAYBE_UNUSED sunrealtype t,
                                            N_Vector Y,
                                            SUNDIALS_MAYBE_UNUSED N_Vector R,
                                            SUNMatrix J,
                                            sunindextype* nnz,
                                            void* user_data,
                                            SUNDIALS_MAYBE_UNUSED N_Vector tmp1,
                                            SUNDIALS_MAYBE_UNUSED N_Vector tmp2,
                                            SUNDIALS_MAYBE_UNUSED N_Vector tmp3)
{
  LotkaVolterraParams p = *((LotkaVolterraParams*)user_data);

  const sunindextype xj  = LOTKA_VOLTERRA_VAR_IDX_MAP[0][0],
                     xpj = LOTKA_VOLTERRA_VAR_IDX_MAP[0][1],
                     yj  = LOTKA_VOLTERRA_VAR_IDX_MAP[1][0],
                     ypj = LOTKA_VOLTERRA_VAR_IDX_MAP[1][1];

  const sunrealtype x = NV_Ith(Y, xj), y = NV_Ith(Y, yj);

  if (j == xj)
  {
    SET_SPARSE(J, *nnz, 0, -p.a + p.b * y);
    SET_SPARSE(J, *nnz, 1, -p.d * y);
  }
  else if (j == xpj) { SET_SPARSE(J, *nnz, 0, ONE); }
  else if (j == yj)
  {
    SET_SPARSE(J, *nnz, 0, p.b * x);
    SET_SPARSE(J, *nnz, 1, -p.d * x + p.c);
  }
  else if (j == ypj) { SET_SPARSE(J, *nnz, 1, ONE); }
  else
  {
    return -1;
  }

  return 0;
}

int LotkaVolterraJacfn_CSC(const sunindextype yy_diff_alias_row[static 1],
                           const sunindextype yp_diff_alias_row[static 1],
                           SUNDIALS_MAYBE_UNUSED sunrealtype t,
                           sunrealtype cj,
                           N_Vector Y,
                           SUNDIALS_MAYBE_UNUSED N_Vector R,
                           SUNMatrix J,
                           void* user_data,
                           SUNDIALS_MAYBE_UNUSED N_Vector tmp1,
                           SUNDIALS_MAYBE_UNUSED N_Vector tmp2,
                           SUNDIALS_MAYBE_UNUSED N_Vector tmp3)
{
  return DDJacFn_CSC(2,
                     LotkaVolterraJacColFn_CSC,
                     yy_diff_alias_row,
                     yp_diff_alias_row,
                     t,
                     cj,
                     Y,
                     R,
                     J,
                     user_data,
                     tmp1,
                     tmp2,
                     tmp3);
}

/* --------------------------------------------------------------------------
 * Pendulum
 * -------------------------------------------------------------------------- */

static const char* PENDULUM_EQN_NAMES[] = {"f₁", "f₂", "f₃"};
static const char* PENDULUM_VAR_NAMES[] = {"x", "y", "λ"};
static const sunindextype PENDULUM_N    = 3;
static const uint8_t PENDULUM_C[]       = {0, 0, 2}; /* f₁, f₂, f₃'' */
static const uint8_t PENDULUM_D[]       = {2, 2, 0}; /* x'', y'', λ */

static const sunindextype* PENDULUM_VAR_IDX_MAP[] =
  {(const sunindextype[]){0, 1, 2}, /* x, x', x'' */
   (const sunindextype[]){3, 4, 5}, /* y, y', y'' */
   (const sunindextype[]){6} /* λ */};

static const sunindextype PENDULUM_JAC_NNZ = 18;

#define P_Ith(Y, i, k) (NV_Ith(Y, PENDULUM_VAR_IDX_MAP[i][k]))

static const uint8_t PENDULUM_NP         = 2;
static const sunindextype PENDULUM_ADJ_N = 7;

typedef struct
{
  sunrealtype m;        /* Arm mass. */
  sunrealtype param[2]; /* Arm length, Gravitational constant */
} PendulumData;

int pendulum_generated_res(DAEIR_REAL g_t_9,
                           DAEIR_REAL const* g_yy_10,
                           DAEIR_REAL const* g_pp_11,
                           DAEIR_REAL* g_rr_12)
{
  DAEIR_REAL const(*const restrict v_x_0)[1] =
    (DAEIR_REAL const(*)[1])(g_yy_10 + 0);
  DAEIR_REAL const(*const restrict v_y_1)[1] =
    (DAEIR_REAL const(*)[1])(g_yy_10 + 3);
  DAEIR_REAL const(*const restrict v_lambda_2)[1] =
    (DAEIR_REAL const(*)[1])(g_yy_10 + 6);
  DAEIR_REAL const(*const restrict p_p_0) = (DAEIR_REAL const(*))(g_pp_11 + 0);
  DAEIR_REAL const(*const restrict p_L_1) = (DAEIR_REAL const(*))(g_pp_11 + 2);
  DAEIR_REAL const g_0                    = p_p_0[0];
  DAEIR_REAL const g_1                    = v_x_0[2][0];
  DAEIR_REAL const g_2                    = p_L_1[0];
  DAEIR_REAL const g_3                    = (v_lambda_2[0][0] / g_2);
  DAEIR_REAL const g_4                    = v_x_0[0][0];
  DAEIR_REAL const g_5                    = v_x_0[1][0];
  g_rr_12[0]                              = ((g_0 * g_1) + (g_3 * g_4));
  DAEIR_REAL const g_6                    = v_y_1[2][0];
  DAEIR_REAL const g_7                    = v_y_1[0][0];
  DAEIR_REAL const g_8                    = v_y_1[1][0];
  g_rr_12[1] = (((g_0 * g_6) + (g_3 * g_7)) + (g_0 * p_p_0[1]));
  g_rr_12[2] = (((g_4 * g_4) + (g_7 * g_7)) - (g_2 * g_2));
  g_rr_12[3] = (((g_5 * g_4) + (g_4 * g_5)) + ((g_8 * g_7) + (g_7 * g_8)));
  g_rr_12[4] =
    ((((g_1 * g_4) + ((DAEIR_REAL_CONST(2) * g_5) * g_5)) + (g_4 * g_1)) +
     (((g_6 * g_7) + ((DAEIR_REAL_CONST(2) * g_8) * g_8)) + (g_7 * g_6)));
  return 0;
}

/* int PendulumRes( */
/*   SUNDIALS_MAYBE_UNUSED sunrealtype t, */
/*   N_Vector Y, /\* {x, x', x'', y, y', y'', λ} *\/ */
/*   N_Vector R, /\* {f₁, f₂, f₃, f₃', f₃'', reserved} *\/ */
/*   SUNDIALS_MAYBE_UNUSED void* user_data */
/* ) */
/* { */
/*   const sunrealtype x = P_Ith(Y, 0, 0), xp = P_Ith(Y, 0, 1), */
/*                     xpp = P_Ith(Y, 0, 2), */
/*  */
/*                     y = P_Ith(Y, 1, 0), yp = P_Ith(Y, 1, 1), */
/*                     ypp = P_Ith(Y, 1, 2), */
/*  */
/*                     lambda = P_Ith(Y, 2, 0); */
/*  */
/*   PendulumData* data  = (PendulumData*)user_data; */
/*   const sunrealtype m = data->m, l = data->param[0], g = data->param[1]; */
/*  */
/*   NV_Ith(R, 0) = m * xpp + lambda / l * x; */
/*   NV_Ith(R, 1) = m * ypp + lambda / l * y + m * g; */
/*   NV_Ith(R, 2) = x * x + y * y - l * l; */
/*   NV_Ith(R, 3) = TWO * (x * xp + y * yp); */
/*   NV_Ith(R, 4) = TWO * (x * xpp + xp * xp + y * ypp + yp * yp); */
/*  */
/*   return 0; */
/* } */

int PendulumRes(SUNDIALS_MAYBE_UNUSED sunrealtype t,
                N_Vector Y, /* {x, x', x'', y, y', y'', λ} */
                N_Vector R, /* {f₁, f₂, f₃, f₃', f₃'', reserved} */
                SUNDIALS_MAYBE_UNUSED void* user_data)
{
  PendulumData* data  = (PendulumData*)user_data;
  const sunrealtype m = data->m, l = data->param[0], g = data->param[1];
  const sunrealtype pp[3] = {m, g, l};
  const sunrealtype* yy   = N_VGetArrayPointer(Y);
  sunrealtype* rr         = N_VGetArrayPointer(R);

  return pendulum_generated_res(t, yy, pp, rr);
}

int pendulum_generated_jac(DAEIR_REAL g_t_13,
                           DAEIR_REAL const* g_yy_14,
                           DAEIR_REAL const* g_pp_15,
                           DAEIR_REAL* g_jac_16)
{
  DAEIR_REAL const(*const restrict v_x_0)[1] =
    (DAEIR_REAL const(*)[1])(g_yy_14 + 0);
  DAEIR_REAL const(*const restrict v_y_1)[1] =
    (DAEIR_REAL const(*)[1])(g_yy_14 + 3);
  DAEIR_REAL const(*const restrict v_lambda_2)[1] =
    (DAEIR_REAL const(*)[1])(g_yy_14 + 6);
  DAEIR_REAL const(*const restrict p_p_0) = (DAEIR_REAL const(*))(g_pp_15 + 0);
  DAEIR_REAL const(*const restrict p_L_1) = (DAEIR_REAL const(*))(g_pp_15 + 2);
  DAEIR_REAL const g_0                    = v_x_0[2][0];
  DAEIR_REAL const g_1                    = v_x_0[0][0];
  DAEIR_REAL const g_2                    = v_x_0[1][0];
  g_jac_16[((7 * 0) + 0)]                 = (v_lambda_2[0][0] / p_L_1[0]);
  g_jac_16[((7 * 0) + 2)]                 = (g_1 + g_1);
  g_jac_16[((7 * 0) + 3)]                 = (g_2 + g_2);
  g_jac_16[((7 * 0) + 4)]                 = (g_0 + g_0);
  DAEIR_REAL const g_3                    = v_x_0[0][0];
  DAEIR_REAL const g_4                    = (DAEIR_REAL_CONST(2) * v_x_0[1][0]);
  g_jac_16[((7 * 1) + 3)]                 = (g_3 + g_3);
  g_jac_16[((7 * 1) + 4)]                 = (g_4 + g_4);
  DAEIR_REAL const g_5                    = v_x_0[0][0];
  g_jac_16[((7 * 2) + 0)]                 = p_p_0[0];
  g_jac_16[((7 * 2) + 4)]                 = (g_5 + g_5);
  DAEIR_REAL const g_6                    = v_y_1[2][0];
  DAEIR_REAL const g_7                    = v_y_1[0][0];
  DAEIR_REAL const g_8                    = v_y_1[1][0];
  g_jac_16[((7 * 3) + 1)]                 = (v_lambda_2[0][0] / p_L_1[0]);
  g_jac_16[((7 * 3) + 2)]                 = (g_7 + g_7);
  g_jac_16[((7 * 3) + 3)]                 = (g_8 + g_8);
  g_jac_16[((7 * 3) + 4)]                 = (g_6 + g_6);
  DAEIR_REAL const g_9                    = v_y_1[0][0];
  DAEIR_REAL const g_10                   = (DAEIR_REAL_CONST(2) * v_y_1[1][0]);
  g_jac_16[((7 * 4) + 3)]                 = (g_9 + g_9);
  g_jac_16[((7 * 4) + 4)]                 = (g_10 + g_10);
  DAEIR_REAL const g_11                   = v_y_1[0][0];
  g_jac_16[((7 * 5) + 1)]                 = p_p_0[0];
  g_jac_16[((7 * 5) + 4)]                 = (g_11 + g_11);
  DAEIR_REAL const g_12                   = (DAEIR_REAL_CONST(1) / p_L_1[0]);
  g_jac_16[((7 * 6) + 0)]                 = (g_12 * v_x_0[0][0]);
  g_jac_16[((7 * 6) + 1)]                 = (g_12 * v_y_1[0][0]);
  return 0;
}

int PendulumJacfn_Dense(SUNDIALS_MAYBE_UNUSED sunrealtype t,
                        N_Vector Y,
                        SUNDIALS_MAYBE_UNUSED N_Vector R,
                        SUNMatrix J,
                        void* user_data,
                        SUNDIALS_MAYBE_UNUSED N_Vector tmp1,
                        SUNDIALS_MAYBE_UNUSED N_Vector tmp2,
                        SUNDIALS_MAYBE_UNUSED N_Vector tmp3)
{
  PendulumData* data  = (PendulumData*)user_data;
  const sunrealtype m = data->m, l = data->param[0], g = data->param[1];
  const sunrealtype pp[3] = {m, g, l};
  const sunrealtype* yy   = N_VGetArrayPointer(Y);
  sunrealtype* jac        = SM_DATA_D(J);

  return pendulum_generated_jac(t, yy, pp, jac);
}

/* int PendulumJacfn_Dense( */
/*   SUNDIALS_MAYBE_UNUSED sunrealtype t, */
/*   N_Vector Y, */
/*   SUNDIALS_MAYBE_UNUSED N_Vector R, */
/*   SUNMatrix J, */
/*   void* user_data, */
/*   SUNDIALS_MAYBE_UNUSED N_Vector tmp1, */
/*   SUNDIALS_MAYBE_UNUSED N_Vector tmp2, */
/*   SUNDIALS_MAYBE_UNUSED N_Vector tmp3 */
/* ) */
/* { */
/*   const sunrealtype x = P_Ith(Y, 0, 0), xp = P_Ith(Y, 0, 1), */
/*                     xpp = P_Ith(Y, 0, 2), */
/*  */
/*                     y = P_Ith(Y, 1, 0), yp = P_Ith(Y, 1, 1), */
/*                     ypp = P_Ith(Y, 1, 2), */
/*  */
/*                     lambda = P_Ith(Y, 2, 0); */
/*  */
/*   PendulumData* data  = (PendulumData*)user_data; */
/*   const sunrealtype m = data->m, l = data->param[0]; */
/*  */
/*   /\* column 0 *\/ */
/*   SM_ELEMENT_D(J, 0, 0) = lambda / l; */
/*   SM_ELEMENT_D(J, 2, 0) = TWO * x; */
/*   SM_ELEMENT_D(J, 3, 0) = TWO * xp; */
/*   SM_ELEMENT_D(J, 4, 0) = TWO * xpp; */
/*  */
/*   /\* column 1 *\/ */
/*   SM_ELEMENT_D(J, 3, 1) = TWO * x; */
/*   SM_ELEMENT_D(J, 4, 1) = FOUR * xp; */
/*  */
/*   /\* column 2 *\/ */
/*   SM_ELEMENT_D(J, 0, 2) = m; */
/*   SM_ELEMENT_D(J, 4, 2) = TWO * x; */
/*  */
/*   /\* column 3 *\/ */
/*   SM_ELEMENT_D(J, 1, 3) = lambda / l; */
/*   SM_ELEMENT_D(J, 2, 3) = TWO * y; */
/*   SM_ELEMENT_D(J, 3, 3) = TWO * yp; */
/*   SM_ELEMENT_D(J, 4, 3) = TWO * ypp; */
/*  */
/*   /\* column 4 *\/ */
/*   SM_ELEMENT_D(J, 3, 4) = TWO * y; */
/*   SM_ELEMENT_D(J, 4, 4) = FOUR * yp; */
/*  */
/*   /\* column 6 *\/ */
/*   SM_ELEMENT_D(J, 1, 5) = m; */
/*   SM_ELEMENT_D(J, 4, 5) = TWO * y; */
/*  */
/*   /\* column 6 *\/ */
/*   SM_ELEMENT_D(J, 0, 6) = x / l; */
/*   SM_ELEMENT_D(J, 1, 6) = y / l; */
/*  */
/*   return 0; */
/* } */

int pendulum_generated_jac_csr(DAEIR_REAL g_t_13,
                               DAEIR_REAL const* g_yy_14,
                               DAEIR_REAL const* g_pp_15,
                               DAEIR_SIZE* g_rowptrs_16,
                               DAEIR_SIZE* g_colvals_17,
                               DAEIR_REAL* g_data_18)
{
  DAEIR_REAL const(*const restrict v_x_0)[1] =
    (DAEIR_REAL const(*)[1])(g_yy_14 + 0);
  DAEIR_REAL const(*const restrict v_y_1)[1] =
    (DAEIR_REAL const(*)[1])(g_yy_14 + 3);
  DAEIR_REAL const(*const restrict v_lambda_2)[1] =
    (DAEIR_REAL const(*)[1])(g_yy_14 + 6);
  DAEIR_REAL const(*const restrict p_p_0) = (DAEIR_REAL const(*))(g_pp_15 + 0);
  DAEIR_REAL const(*const restrict p_L_1) = (DAEIR_REAL const(*))(g_pp_15 + 2);
  DAEIR_REAL const g_0                    = v_x_0[2][0];
  DAEIR_REAL const g_1                    = v_x_0[0][0];
  DAEIR_REAL const g_2                    = v_x_0[1][0];
  g_colvals_17[0]                         = 0;
  g_data_18[0]                            = (v_lambda_2[0][0] / p_L_1[0]);
  g_colvals_17[6]                         = 0;
  g_data_18[6]                            = (g_1 + g_1);
  g_colvals_17[8]                         = 0;
  g_data_18[8]                            = (g_2 + g_2);
  g_colvals_17[12]                        = 0;
  g_data_18[12]                           = (g_0 + g_0);
  DAEIR_REAL const g_3                    = v_x_0[0][0];
  DAEIR_REAL const g_4                    = (DAEIR_REAL_CONST(2) * v_x_0[1][0]);
  g_colvals_17[9]                         = 1;
  g_data_18[9]                            = (g_3 + g_3);
  g_colvals_17[13]                        = 1;
  g_data_18[13]                           = (g_4 + g_4);
  DAEIR_REAL const g_5                    = v_x_0[0][0];
  g_colvals_17[1]                         = 2;
  g_data_18[1]                            = p_p_0[0];
  g_colvals_17[14]                        = 2;
  g_data_18[14]                           = (g_5 + g_5);
  DAEIR_REAL const g_6                    = v_y_1[2][0];
  DAEIR_REAL const g_7                    = v_y_1[0][0];
  DAEIR_REAL const g_8                    = v_y_1[1][0];
  g_colvals_17[3]                         = 3;
  g_data_18[3]                            = (v_lambda_2[0][0] / p_L_1[0]);
  g_colvals_17[7]                         = 3;
  g_data_18[7]                            = (g_7 + g_7);
  g_colvals_17[10]                        = 3;
  g_data_18[10]                           = (g_8 + g_8);
  g_colvals_17[15]                        = 3;
  g_data_18[15]                           = (g_6 + g_6);
  DAEIR_REAL const g_9                    = v_y_1[0][0];
  DAEIR_REAL const g_10                   = (DAEIR_REAL_CONST(2) * v_y_1[1][0]);
  g_colvals_17[11]                        = 4;
  g_data_18[11]                           = (g_9 + g_9);
  g_colvals_17[16]                        = 4;
  g_data_18[16]                           = (g_10 + g_10);
  DAEIR_REAL const g_11                   = v_y_1[0][0];
  g_colvals_17[4]                         = 5;
  g_data_18[4]                            = p_p_0[0];
  g_colvals_17[17]                        = 5;
  g_data_18[17]                           = (g_11 + g_11);
  DAEIR_REAL const g_12                   = (DAEIR_REAL_CONST(1) / p_L_1[0]);
  g_colvals_17[2]                         = 6;
  g_data_18[2]                            = (g_12 * v_x_0[0][0]);
  g_colvals_17[5]                         = 6;
  g_data_18[5]                            = (g_12 * v_y_1[0][0]);
  g_rowptrs_16[0]                         = 0;
  g_rowptrs_16[1]                         = 3;
  g_rowptrs_16[2]                         = 6;
  g_rowptrs_16[3]                         = 8;
  g_rowptrs_16[4]                         = 12;
  g_rowptrs_16[5]                         = 18;
  return 0;
}

int PendulumJacfn_CSR(SUNDIALS_MAYBE_UNUSED sunrealtype t,
                      N_Vector Y,
                      SUNDIALS_MAYBE_UNUSED N_Vector R,
                      SUNMatrix J,
                      void* user_data,
                      SUNDIALS_MAYBE_UNUSED N_Vector tmp1,
                      SUNDIALS_MAYBE_UNUSED N_Vector tmp2,
                      SUNDIALS_MAYBE_UNUSED N_Vector tmp3)
{
  PendulumData* data  = (PendulumData*)user_data;
  const sunrealtype m = data->m, l = data->param[0], g = data->param[1];
  const sunrealtype pp[3] = {m, g, l};
  const sunrealtype* yy   = N_VGetArrayPointer(Y);

  sunindextype* J_indexptrs = SM_INDEXPTRS_S(J);
  sunindextype* J_indexvals = SM_INDEXVALS_S(J);
  sunrealtype* J_data       = SM_DATA_S(J);

  return pendulum_generated_jac_csr(t, yy, pp, J_indexptrs, J_indexvals, J_data);
}

/* int PendulumJacfn_CSR( */
/*   SUNDIALS_MAYBE_UNUSED sunrealtype t, */
/*   N_Vector Y, */
/*   SUNDIALS_MAYBE_UNUSED N_Vector R, */
/*   SUNMatrix J, */
/*   void* user_data, */
/*   SUNDIALS_MAYBE_UNUSED N_Vector tmp1, */
/*   SUNDIALS_MAYBE_UNUSED N_Vector tmp2, */
/*   SUNDIALS_MAYBE_UNUSED N_Vector tmp3 */
/* ) */
/* { */
/*   const sunrealtype x = P_Ith(Y, 0, 0), xp = P_Ith(Y, 0, 1), */
/*                     xpp = P_Ith(Y, 0, 2), */
/*  */
/*                     y = P_Ith(Y, 1, 0), yp = P_Ith(Y, 1, 1), */
/*                     ypp = P_Ith(Y, 1, 2), */
/*  */
/*                     lambda = P_Ith(Y, 2, 0); */
/*  */
/*   PendulumData* data  = (PendulumData*)user_data; */
/*   const sunrealtype m = data->m, l = data->param[0]; */
/*  */
/*   sunindextype nnz = 0; */
/*  */
/*   /\* row 0 *\/ */
/*   SM_INDEXPTRS_S(J)[0] = nnz; */
/*   SET_SPARSE(J, nnz, 0, lambda / l); */
/*   SET_SPARSE(J, nnz, 2, m); */
/*   SET_SPARSE(J, nnz, 6, x / l); */
/*  */
/*   /\* row 1 *\/ */
/*   SM_INDEXPTRS_S(J)[1] = nnz; */
/*   SET_SPARSE(J, nnz, 3, lambda / l); */
/*   SET_SPARSE(J, nnz, 5, m); */
/*   SET_SPARSE(J, nnz, 6, y / l); */
/*  */
/*   /\* row 2 *\/ */
/*   SM_INDEXPTRS_S(J)[2] = nnz; */
/*   SET_SPARSE(J, nnz, 0, TWO * x); */
/*   SET_SPARSE(J, nnz, 3, TWO * y); */
/*  */
/*   /\* row 3 *\/ */
/*   SM_INDEXPTRS_S(J)[3] = nnz; */
/*   SET_SPARSE(J, nnz, 0, TWO * xp); */
/*   SET_SPARSE(J, nnz, 1, TWO * x); */
/*   SET_SPARSE(J, nnz, 3, TWO * yp); */
/*   SET_SPARSE(J, nnz, 4, TWO * y); */
/*  */
/*   /\* row 4 *\/ */
/*   SM_INDEXPTRS_S(J)[4] = nnz; */
/*   SET_SPARSE(J, nnz, 0, TWO * xpp); */
/*   SET_SPARSE(J, nnz, 1, FOUR * xp); */
/*   SET_SPARSE(J, nnz, 2, TWO * x); */
/*   SET_SPARSE(J, nnz, 3, TWO * ypp); */
/*   SET_SPARSE(J, nnz, 4, FOUR * yp); */
/*   SET_SPARSE(J, nnz, 5, TWO * y); */
/*  */
/*   SM_INDEXPTRS_S(J)[5] = nnz; */
/*  */
/*   return 0; */
/* } */

int pendulum_generated_jac_csc(DAEIR_SIZE g_j_13,
                               DAEIR_REAL g_t_14,
                               DAEIR_REAL const* g_yy_15,
                               DAEIR_REAL const* g_pp_16,
                               DAEIR_SIZE* g_rowvals_17,
                               DAEIR_REAL* g_data_18,
                               DAEIR_SIZE* g_nnz_19)
{
  DAEIR_REAL const(*const restrict v_x_0)[1] =
    (DAEIR_REAL const(*)[1])(g_yy_15 + 0);
  DAEIR_REAL const(*const restrict v_y_1)[1] =
    (DAEIR_REAL const(*)[1])(g_yy_15 + 3);
  DAEIR_REAL const(*const restrict v_lambda_2)[1] =
    (DAEIR_REAL const(*)[1])(g_yy_15 + 6);
  DAEIR_REAL const(*const restrict p_p_0) = (DAEIR_REAL const(*))(g_pp_16 + 0);
  DAEIR_REAL const(*const restrict p_L_1) = (DAEIR_REAL const(*))(g_pp_16 + 2);
  switch (g_j_13)
  {
  case 0:
  {
    DAEIR_REAL const g_0        = v_x_0[2][0];
    DAEIR_REAL const g_1        = v_x_0[0][0];
    DAEIR_REAL const g_2        = v_x_0[1][0];
    DAEIR_SIZE l_nnz_0          = g_nnz_19[0];
    g_rowvals_17[(l_nnz_0 + 0)] = 0;
    g_data_18[(l_nnz_0 + 0)]    = (v_lambda_2[0][0] / p_L_1[0]);
    g_rowvals_17[(l_nnz_0 + 1)] = 2;
    g_data_18[(l_nnz_0 + 1)]    = (g_1 + g_1);
    g_rowvals_17[(l_nnz_0 + 2)] = 3;
    g_data_18[(l_nnz_0 + 2)]    = (g_2 + g_2);
    g_rowvals_17[(l_nnz_0 + 3)] = 4;
    g_data_18[(l_nnz_0 + 3)]    = (g_0 + g_0);
    g_nnz_19[0]                 = (l_nnz_0 + 4);
    break;
  }
  case 1:
  {
    DAEIR_REAL const g_3        = v_x_0[0][0];
    DAEIR_REAL const g_4        = (DAEIR_REAL_CONST(2) * v_x_0[1][0]);
    DAEIR_SIZE l_nnz_1          = g_nnz_19[0];
    g_rowvals_17[(l_nnz_1 + 0)] = 3;
    g_data_18[(l_nnz_1 + 0)]    = (g_3 + g_3);
    g_rowvals_17[(l_nnz_1 + 1)] = 4;
    g_data_18[(l_nnz_1 + 1)]    = (g_4 + g_4);
    g_nnz_19[0]                 = (l_nnz_1 + 2);
    break;
  }
  case 2:
  {
    DAEIR_REAL const g_5        = v_x_0[0][0];
    DAEIR_SIZE l_nnz_2          = g_nnz_19[0];
    g_rowvals_17[(l_nnz_2 + 0)] = 0;
    g_data_18[(l_nnz_2 + 0)]    = p_p_0[0];
    g_rowvals_17[(l_nnz_2 + 1)] = 4;
    g_data_18[(l_nnz_2 + 1)]    = (g_5 + g_5);
    g_nnz_19[0]                 = (l_nnz_2 + 2);
    break;
  }
  case 3:
  {
    DAEIR_REAL const g_6        = v_y_1[2][0];
    DAEIR_REAL const g_7        = v_y_1[0][0];
    DAEIR_REAL const g_8        = v_y_1[1][0];
    DAEIR_SIZE l_nnz_3          = g_nnz_19[0];
    g_rowvals_17[(l_nnz_3 + 0)] = 1;
    g_data_18[(l_nnz_3 + 0)]    = (v_lambda_2[0][0] / p_L_1[0]);
    g_rowvals_17[(l_nnz_3 + 1)] = 2;
    g_data_18[(l_nnz_3 + 1)]    = (g_7 + g_7);
    g_rowvals_17[(l_nnz_3 + 2)] = 3;
    g_data_18[(l_nnz_3 + 2)]    = (g_8 + g_8);
    g_rowvals_17[(l_nnz_3 + 3)] = 4;
    g_data_18[(l_nnz_3 + 3)]    = (g_6 + g_6);
    g_nnz_19[0]                 = (l_nnz_3 + 4);
    break;
  }
  case 4:
  {
    DAEIR_REAL const g_9        = v_y_1[0][0];
    DAEIR_REAL const g_10       = (DAEIR_REAL_CONST(2) * v_y_1[1][0]);
    DAEIR_SIZE l_nnz_4          = g_nnz_19[0];
    g_rowvals_17[(l_nnz_4 + 0)] = 3;
    g_data_18[(l_nnz_4 + 0)]    = (g_9 + g_9);
    g_rowvals_17[(l_nnz_4 + 1)] = 4;
    g_data_18[(l_nnz_4 + 1)]    = (g_10 + g_10);
    g_nnz_19[0]                 = (l_nnz_4 + 2);
    break;
  }
  case 5:
  {
    DAEIR_REAL const g_11       = v_y_1[0][0];
    DAEIR_SIZE l_nnz_5          = g_nnz_19[0];
    g_rowvals_17[(l_nnz_5 + 0)] = 1;
    g_data_18[(l_nnz_5 + 0)]    = p_p_0[0];
    g_rowvals_17[(l_nnz_5 + 1)] = 4;
    g_data_18[(l_nnz_5 + 1)]    = (g_11 + g_11);
    g_nnz_19[0]                 = (l_nnz_5 + 2);
    break;
  }
  case 6:
  {
    DAEIR_REAL const g_12       = (DAEIR_REAL_CONST(1) / p_L_1[0]);
    DAEIR_SIZE l_nnz_6          = g_nnz_19[0];
    g_rowvals_17[(l_nnz_6 + 0)] = 0;
    g_data_18[(l_nnz_6 + 0)]    = (g_12 * v_x_0[0][0]);
    g_rowvals_17[(l_nnz_6 + 1)] = 1;
    g_data_18[(l_nnz_6 + 1)]    = (g_12 * v_y_1[0][0]);
    g_nnz_19[0]                 = (l_nnz_6 + 2);
    break;
  }
  }
  return 0;
}

static inline int PendulumJacColFn_CSC(sunindextype j,
                                       SUNDIALS_MAYBE_UNUSED sunrealtype t,
                                       N_Vector Y,
                                       SUNDIALS_MAYBE_UNUSED N_Vector R,
                                       SUNMatrix J,
                                       sunindextype* nnz,
                                       void* user_data,
                                       SUNDIALS_MAYBE_UNUSED N_Vector tmp1,
                                       SUNDIALS_MAYBE_UNUSED N_Vector tmp2,
                                       SUNDIALS_MAYBE_UNUSED N_Vector tmp3)
{
  PendulumData* data  = (PendulumData*)user_data;
  const sunrealtype m = data->m, l = data->param[0], g = data->param[1];
  const sunrealtype pp[3] = {m, g, l};
  const sunrealtype* yy   = N_VGetArrayPointer(Y);

  sunindextype* J_indexvals = SM_INDEXVALS_S(J);
  sunrealtype* J_data       = SM_DATA_S(J);

  return pendulum_generated_jac_csc(j, t, yy, pp, J_indexvals, J_data, nnz);
}

/* static inline int PendulumJacColFn_CSC( */
/*   sunindextype j, */
/*   SUNDIALS_MAYBE_UNUSED sunrealtype t, */
/*   N_Vector Y, */
/*   SUNDIALS_MAYBE_UNUSED N_Vector R, */
/*   SUNMatrix J, */
/*   sunindextype* nnz, */
/*   void* user_data, */
/*   SUNDIALS_MAYBE_UNUSED N_Vector tmp1, */
/*   SUNDIALS_MAYBE_UNUSED N_Vector tmp2, */
/*   SUNDIALS_MAYBE_UNUSED N_Vector tmp3 */
/* ) */
/* { */
/*   const sunrealtype x = P_Ith(Y, 0, 0), xp = P_Ith(Y, 0, 1), */
/*                     xpp = P_Ith(Y, 0, 2), */
/*  */
/*                     y = P_Ith(Y, 1, 0), yp = P_Ith(Y, 1, 1), */
/*                     ypp = P_Ith(Y, 1, 2), */
/*  */
/*                     lambda = P_Ith(Y, 2, 0); */
/*  */
/*   PendulumData* data  = (PendulumData*)user_data; */
/*   const sunrealtype m = data->m, l = data->param[0]; */
/*  */
/*   switch (j) */
/*   { */
/*   case 0: */
/*     SET_SPARSE(J, *nnz, 0, lambda / l); */
/*     SET_SPARSE(J, *nnz, 2, TWO * x); */
/*     SET_SPARSE(J, *nnz, 3, TWO * xp); */
/*     SET_SPARSE(J, *nnz, 4, TWO * xpp); */
/*     break; */
/*   case 1: */
/*     SET_SPARSE(J, *nnz, 3, TWO * x); */
/*     SET_SPARSE(J, *nnz, 4, FOUR * xp); */
/*     break; */
/*   case 2: */
/*     SET_SPARSE(J, *nnz, 0, m); */
/*     SET_SPARSE(J, *nnz, 4, TWO * x); */
/*     break; */
/*   case 3: */
/*     SET_SPARSE(J, *nnz, 1, lambda / l); */
/*     SET_SPARSE(J, *nnz, 2, TWO * y); */
/*     SET_SPARSE(J, *nnz, 3, TWO * yp); */
/*     SET_SPARSE(J, *nnz, 4, TWO * ypp); */
/*     break; */
/*   case 4: */
/*     SET_SPARSE(J, *nnz, 3, TWO * y); */
/*     SET_SPARSE(J, *nnz, 4, FOUR * yp); */
/*     break; */
/*   case 5: */
/*     SET_SPARSE(J, *nnz, 1, m); */
/*     SET_SPARSE(J, *nnz, 4, TWO * y); */
/*     break; */
/*   case 6: */
/*     SET_SPARSE(J, *nnz, 0, x / l); */
/*     SET_SPARSE(J, *nnz, 1, y / l); */
/*     break; */
/*   } */
/*  */
/*   return 0; */
/* } */

int PendulumJacfn_CSC(const sunindextype yy_diff_alias_row[static 1],
                      const sunindextype yp_diff_alias_row[static 1],
                      SUNDIALS_MAYBE_UNUSED sunrealtype t,
                      sunrealtype cj,
                      N_Vector Y,
                      N_Vector R,
                      SUNMatrix J,
                      void* user_data,
                      SUNDIALS_MAYBE_UNUSED N_Vector tmp1,
                      SUNDIALS_MAYBE_UNUSED N_Vector tmp2,
                      SUNDIALS_MAYBE_UNUSED N_Vector tmp3)
{
  return DDJacFn_CSC(5,
                     PendulumJacColFn_CSC,
                     yy_diff_alias_row,
                     yp_diff_alias_row,
                     t,
                     cj,
                     Y,
                     R,
                     J,
                     user_data,
                     tmp1,
                     tmp2,
                     tmp3);
}

void PendulumY0(PendulumData* data, sunrealtype theta0, N_Vector Y)
{
  const sunrealtype m = data->m, l = data->param[0], g = data->param[1];
  N_VConst(ZERO, Y);
  P_Ith(Y, 0, 0) = l * sin(theta0);
  P_Ith(Y, 1, 0) = -l * cos(theta0);
  P_Ith(Y, 0, 2) = -g * sin(TWO * theta0) / TWO;
  P_Ith(Y, 1, 2) = -g * sin(theta0) * sin(theta0);
  P_Ith(Y, 2, 0) = m * g * cos(theta0);
}

void PendulumYS0(PendulumData* data, sunrealtype theta0, N_Vector* YS)
{
  const sunrealtype m = data->m;
  for (int i = 0; i < PENDULUM_NP; ++i) { N_VConst(ZERO, YS[i]); }
  P_Ith(YS[0], 0, 0) = sin(theta0);
  P_Ith(YS[0], 1, 0) = -cos(theta0);
  P_Ith(YS[1], 0, 2) = -sin(TWO * theta0) / TWO;
  P_Ith(YS[1], 1, 2) = -sin(theta0) * sin(theta0);
  P_Ith(YS[1], 2, 0) = m * cos(theta0);
}

int pendulum_generated_sens_res(DAEIR_REAL g_t_16,
                                DAEIR_REAL const* g_yy_17,
                                DAEIR_REAL const** g_yyS_19,
                                DAEIR_REAL const* g_pp_18,
                                DAEIR_REAL** g_rrS_20)
{
  DAEIR_REAL const(*const restrict v_x_0)[1] =
    (DAEIR_REAL const(*)[1])(g_yy_17 + 0);
  DAEIR_REAL const(*const restrict v_y_1)[1] =
    (DAEIR_REAL const(*)[1])(g_yy_17 + 3);
  DAEIR_REAL const(*const restrict v_lambda_2)[1] =
    (DAEIR_REAL const(*)[1])(g_yy_17 + 6);
  DAEIR_REAL const(*const restrict p_p_0) = (DAEIR_REAL const(*))(g_pp_18 + 0);
  DAEIR_REAL const(*const restrict p_L_1) = (DAEIR_REAL const(*))(g_pp_18 + 2);
  for (DAEIR_SIZE l_k_0 = 0; (2 - l_k_0); l_k_0 = (l_k_0 + 1))
  {
    DAEIR_REAL const* l_yyS_k_1 = g_yyS_19[l_k_0];
    DAEIR_REAL const(*const restrict sv_x_0)[1] =
      (DAEIR_REAL const(*)[1])(l_yyS_k_1 + 0);
    DAEIR_REAL const(*const restrict sv_y_1)[1] =
      (DAEIR_REAL const(*)[1])(l_yyS_k_1 + 3);
    DAEIR_REAL const(*const restrict sv_lambda_2)[1] =
      (DAEIR_REAL const(*)[1])(l_yyS_k_1 + 6);
    DAEIR_REAL const g_0  = p_p_0[0];
    DAEIR_REAL const g_1  = v_x_0[2][0];
    DAEIR_REAL const g_2  = sv_x_0[2][0];
    DAEIR_REAL const g_3  = p_L_1[0];
    DAEIR_REAL const g_4  = (v_lambda_2[0][0] / g_3);
    DAEIR_REAL const g_5  = (sv_lambda_2[0][0] / g_3);
    DAEIR_REAL const g_6  = v_x_0[0][0];
    DAEIR_REAL const g_7  = v_x_0[1][0];
    DAEIR_REAL const g_8  = sv_x_0[0][0];
    DAEIR_REAL const g_9  = sv_x_0[1][0];
    g_rrS_20[l_k_0][0]    = ((g_0 * g_2) + ((g_4 * g_8) + (g_5 * g_6)));
    DAEIR_REAL const g_10 = v_y_1[2][0];
    DAEIR_REAL const g_11 = sv_y_1[2][0];
    DAEIR_REAL const g_12 = v_y_1[0][0];
    DAEIR_REAL const g_13 = v_y_1[1][0];
    DAEIR_REAL const g_14 = sv_y_1[0][0];
    DAEIR_REAL const g_15 = sv_y_1[1][0];
    g_rrS_20[l_k_0][1]    = ((g_0 * g_11) + ((g_4 * g_14) + (g_5 * g_12)));
    g_rrS_20[l_k_0][2] =
      (((g_6 * g_8) + (g_8 * g_6)) + ((g_12 * g_14) + (g_14 * g_12)));
    g_rrS_20[l_k_0][3] =
      ((((g_7 * g_8) + (g_6 * g_9)) + ((g_9 * g_6) + (g_8 * g_7))) +
       (((g_13 * g_14) + (g_12 * g_15)) + ((g_15 * g_12) + (g_14 * g_13))));
    g_rrS_20[l_k_0][4] =
      (((((g_1 * g_8) + ((DAEIR_REAL_CONST(2) * g_7) * g_9)) + (g_6 * g_2)) +
        (((g_2 * g_6) + ((DAEIR_REAL_CONST(2) * g_9) * g_7)) + (g_8 * g_1))) +
       ((((g_10 * g_14) + ((DAEIR_REAL_CONST(2) * g_13) * g_15)) + (g_12 * g_11)) +
        (((g_11 * g_12) + ((DAEIR_REAL_CONST(2) * g_15) * g_13)) + (g_14 * g_10))));
  }
  {
    DAEIR_REAL const g_0 = p_L_1[0];
    DAEIR_REAL const g_1 =
      ((DAEIR_REAL_CONST(0) - (v_lambda_2[0][0] / g_0)) / g_0);
    g_rrS_20[0][0] = (g_rrS_20[0][0] + (g_1 * v_x_0[0][0]));
    g_rrS_20[0][1] = (g_rrS_20[0][1] + (g_1 * v_y_1[0][0]));
    g_rrS_20[0][2] = (g_rrS_20[0][2] + (DAEIR_REAL_CONST(0) - (g_0 + g_0)));
  }
  {
    g_rrS_20[1][1] = (g_rrS_20[1][1] + p_p_0[0]);
  }
  return 0;
}

/* int PendulumResS( */
/*   int Ns, */
/*   SUNDIALS_MAYBE_UNUSED sunrealtype t, */
/*   N_Vector Y, */
/*   SUNDIALS_MAYBE_UNUSED N_Vector R, */
/*   N_Vector* YS, */
/*   N_Vector* RS, */
/*   void* user_data, */
/*   SUNDIALS_MAYBE_UNUSED N_Vector tmp1, */
/*   SUNDIALS_MAYBE_UNUSED N_Vector tmp2, */
/*   SUNDIALS_MAYBE_UNUSED N_Vector tmp3 */
/* ) */
/* { */
/*   assert(Ns == PENDULUM_NP); */
/*   assert(user_data); */
/*   PendulumData* data  = (PendulumData*)user_data; */
/*   sunrealtype const m = data->m, l = data->param[0]; */
/*  */
/*   const sunrealtype x = P_Ith(Y, 0, 0), xp = P_Ith(Y, 0, 1), */
/*                     xpp = P_Ith(Y, 0, 2), */
/*  */
/*                     y = P_Ith(Y, 1, 0), yp = P_Ith(Y, 1, 1), */
/*                     ypp = P_Ith(Y, 1, 2), */
/*  */
/*                     lambda = P_Ith(Y, 2, 0); */
/*  */
/*   for (int i = 0; i < Ns; ++i) */
/*   { */
/*     const sunrealtype xS = P_Ith(YS[i], 0, 0), xpS = P_Ith(YS[i], 0, 1), */
/*                       xppS = P_Ith(YS[i], 0, 2), */
/*  */
/*                       yS = P_Ith(YS[i], 1, 0), ypS = P_Ith(YS[i], 1, 1), */
/*                       yppS = P_Ith(YS[i], 1, 2), */
/*  */
/*                       lambdaS = P_Ith(YS[i], 2, 0); */
/*  */
/*     NV_Ith(RS[i], 0) = lambda * xS / l + m * xppS + x * lambdaS / l; */
/*     NV_Ith(RS[i], 1) = lambda * yS / l + m * yppS + y * lambdaS / l; */
/*     NV_Ith(RS[i], 2) = TWO * (x * xS + y * yS); */
/*     NV_Ith(RS[i], 3) = TWO * (xp * xS + x * xpS + yp * yS + y * ypS); */
/*     NV_Ith(RS[i], 4) = TWO * (xpp * xS + TWO * xp * xpS + x * xppS + ypp * yS + */
/*                               TWO * yp * ypS + y * yppS); */
/*   } */
/*  */
/*   NV_Ith(RS[0], 0) -= lambda * x / (l * l); */
/*   NV_Ith(RS[0], 1) -= lambda * y / (l * l); */
/*   NV_Ith(RS[0], 2) -= TWO * l; */
/*   NV_Ith(RS[1], 1) += m; */
/*  */
/*   return 0; */
/* } */

int PendulumResS(int Ns,
                 SUNDIALS_MAYBE_UNUSED sunrealtype t,
                 N_Vector Y,
                 SUNDIALS_MAYBE_UNUSED N_Vector R,
                 N_Vector* YS,
                 N_Vector* RS,
                 void* user_data,
                 SUNDIALS_MAYBE_UNUSED N_Vector tmp1,
                 SUNDIALS_MAYBE_UNUSED N_Vector tmp2,
                 SUNDIALS_MAYBE_UNUSED N_Vector tmp3)
{
  assert(Ns == PENDULUM_NP);
  assert(user_data);
  PendulumData* data  = (PendulumData*)user_data;
  const sunrealtype m = data->m, l = data->param[0], g = data->param[1];
  const sunrealtype pp[3]   = {m, g, l};
  const sunrealtype* yy     = N_VGetArrayPointer(Y);
  const sunrealtype* yyS[2] = {N_VGetArrayPointer(YS[0]),
                               N_VGetArrayPointer(YS[1])};
  sunrealtype* rrS[2] = {N_VGetArrayPointer(RS[0]), N_VGetArrayPointer(RS[1])};

  return pendulum_generated_sens_res(t, yy, yyS, pp, rrS);
}

void PendulumYyBT(SUNDIALS_MAYBE_UNUSED PendulumData* data,
                  SUNDIALS_MAYBE_UNUSED N_Vector Y,
                  N_Vector yyBT,
                  N_Vector ypBT)
{
  N_VConst(ZERO, yyBT);
  N_VConst(ZERO, ypBT);
  NV_Ith(ypBT, 3) = ONE; /* l4p(T) = 1 */
}

sunrealtype PendulumG(SUNDIALS_MAYBE_UNUSED sunrealtype t,
                      N_Vector Y,
                      void* user_data)
{
  assert(user_data);
  PendulumData* data = (PendulumData*)user_data;
  sunrealtype l      = data->param[0];
  return P_Ith(Y, 0, 0) + l;
}

int PendulumResB(SUNDIALS_MAYBE_UNUSED sunrealtype t,
                 N_Vector Y,
                 N_Vector yyB,
                 N_Vector ypB,
                 N_Vector rrB,
                 void* user_dataB)
{
  assert(user_dataB);
  PendulumData* data = (PendulumData*)user_dataB;
  sunrealtype m = data->m, l = data->param[0];

  const sunrealtype x = P_Ith(Y, 0, 0), xp = P_Ith(Y, 0, 1),
                    xpp = P_Ith(Y, 0, 2),

                    y = P_Ith(Y, 1, 0), yp = P_Ith(Y, 1, 1),
                    ypp = P_Ith(Y, 1, 2),

                    lambda = P_Ith(Y, 2, 0),

                    l1 = NV_Ith(yyB, 0), l2 = NV_Ith(yyB, 1),
                    l3 = NV_Ith(yyB, 2), l4 = NV_Ith(yyB, 3),
                    l5 = NV_Ith(yyB, 4), l6 = NV_Ith(yyB, 5),
                    l7 = NV_Ith(yyB, 6),

                    l1p = NV_Ith(ypB, 0), l2p = NV_Ith(ypB, 1),
                    l3p = NV_Ith(ypB, 2), l4p = NV_Ith(ypB, 3),
                    l5p = NV_Ith(ypB, 4), l6p = NV_Ith(ypB, 5),
                    l7p = NV_Ith(ypB, 6);

  NV_Ith(rrB, 0) = -l4p - lambda / l * l1 - TWO * xpp * l3 + ONE;
  NV_Ith(rrB, 1) = -l6p - FOUR * xp * l3 - l4;
  NV_Ith(rrB, 2) = -m * l1 - TWO * x * l3 - l6;
  NV_Ith(rrB, 3) = -l5p - lambda / l * l2 - TWO * ypp * l3;
  NV_Ith(rrB, 4) = -l7p - FOUR * yp * l3 - l5;
  NV_Ith(rrB, 5) = -m * l2 - TWO * y * l3 - l7;
  NV_Ith(rrB, 6) = -x / l * l1 - y / l * l2;

  return 0;
}

int PendulumJacFnB(SUNDIALS_MAYBE_UNUSED sunrealtype t,
                   sunrealtype cj,
                   N_Vector Y,
                   N_Vector yyB,
                   N_Vector ypB,
                   SUNDIALS_MAYBE_UNUSED N_Vector rrB,
                   SUNMatrix JB,
                   void* user_dataB,
                   SUNDIALS_MAYBE_UNUSED N_Vector tmp1,
                   SUNDIALS_MAYBE_UNUSED N_Vector tmp2,
                   SUNDIALS_MAYBE_UNUSED N_Vector tmp3)
{
  assert(user_dataB);
  PendulumData* data = (PendulumData*)user_dataB;
  sunrealtype m = data->m, l = data->param[0];

  const sunrealtype x = P_Ith(Y, 0, 0), xp = P_Ith(Y, 0, 1),
                    xpp = P_Ith(Y, 0, 2),

                    y = P_Ith(Y, 1, 0), yp = P_Ith(Y, 1, 1),
                    ypp = P_Ith(Y, 1, 2),

                    lambda = P_Ith(Y, 2, 0),

                    l1 = NV_Ith(yyB, 0), l2 = NV_Ith(yyB, 1),
                    l3 = NV_Ith(yyB, 2), l4 = NV_Ith(yyB, 3),
                    l5 = NV_Ith(yyB, 4), l6 = NV_Ith(yyB, 5),
                    l7 = NV_Ith(yyB, 6),

                    l1p = NV_Ith(ypB, 0), l2p = NV_Ith(ypB, 1),
                    l3p = NV_Ith(ypB, 2), l4p = NV_Ith(ypB, 3),
                    l5p = NV_Ith(ypB, 4), l6p = NV_Ith(ypB, 5),
                    l7p = NV_Ith(ypB, 6);

  /* Row 0 */
  SM_ELEMENT_D(JB, 0, 0) = -lambda / l; /* l1 */
  SM_ELEMENT_D(JB, 0, 2) = -TWO * xpp;  /* l3 */
  SM_ELEMENT_D(JB, 0, 3) = -cj;         /* l4 */

  /* Row 1 */
  SM_ELEMENT_D(JB, 1, 2) = -FOUR * xp; /* l3 */
  SM_ELEMENT_D(JB, 1, 3) = -ONE;       /* l4 */
  SM_ELEMENT_D(JB, 1, 5) = -cj;        /* l6 */

  /* Row 2 */
  SM_ELEMENT_D(JB, 2, 0) = -m;       /* l1 */
  SM_ELEMENT_D(JB, 2, 2) = -TWO * x; /* l3 */
  SM_ELEMENT_D(JB, 2, 5) = -ONE;     /* l6 */

  /* Row 3 */
  SM_ELEMENT_D(JB, 3, 1) = -lambda / l; /* l2 */
  SM_ELEMENT_D(JB, 3, 2) = -TWO * ypp;  /* l3 */
  SM_ELEMENT_D(JB, 3, 4) = -cj;         /* l5 */

  /* Row 4 */
  SM_ELEMENT_D(JB, 4, 2) = -FOUR * yp; /* l3 */
  SM_ELEMENT_D(JB, 4, 4) = -ONE;       /* l5 */
  SM_ELEMENT_D(JB, 4, 6) = -cj;        /* l7 */

  /* Row 5 */
  SM_ELEMENT_D(JB, 5, 1) = -m;       /* l2 */
  SM_ELEMENT_D(JB, 5, 2) = -TWO * y; /* l3 */
  SM_ELEMENT_D(JB, 5, 6) = -ONE;     /* l7 */

  /* Row 6 */
  SM_ELEMENT_D(JB, 6, 0) = -x / l; /* l1 */
  SM_ELEMENT_D(JB, 6, 1) = -y / l; /* l2 */

  return 0;
}

int PendulumQuadRhsFnB(SUNDIALS_MAYBE_UNUSED sunrealtype t,
                       N_Vector Y,
                       N_Vector yyB,
                       SUNDIALS_MAYBE_UNUSED N_Vector ypB,
                       N_Vector rhsBQ,
                       void* user_dataB)
{
  assert(user_dataB);
  PendulumData* data = (PendulumData*)user_dataB;
  sunrealtype m = data->m, l = data->param[0], g = data->param[1];

  const sunrealtype x = P_Ith(Y, 0, 0), xp = P_Ith(Y, 0, 1),
                    xpp = P_Ith(Y, 0, 2),

                    y = P_Ith(Y, 1, 0), yp = P_Ith(Y, 1, 1),
                    ypp = P_Ith(Y, 1, 2),

                    lambda = P_Ith(Y, 2, 0),

                    l1 = NV_Ith(yyB, 0), l2 = NV_Ith(yyB, 1),
                    l3 = NV_Ith(yyB, 2), l4 = NV_Ith(yyB, 3),
                    l5 = NV_Ith(yyB, 4), l6 = NV_Ith(yyB, 5),
                    l7 = NV_Ith(yyB, 6);

  NV_Ith(rhsBQ, 0) = -xpp * l1 - (ypp + g) * l2;
  NV_Ith(rhsBQ, 1) = ONE + lambda / (l * l) * (x * l1 + y * l2);
  NV_Ith(rhsBQ, 2) = -m * l2;

  return 0;
}

int PendulumJacf0(SUNDIALS_MAYBE_UNUSED sunrealtype t,
                  N_Vector Y,
                  SUNMatrix A,
                  SUNDIALS_MAYBE_UNUSED void* user_data)
{
  PendulumData* data = (PendulumData*)user_data;
  sunrealtype m = data->m, l = data->param[0];

  const sunrealtype x = P_Ith(Y, 0, 0), y = P_Ith(Y, 1, 0);

  SM_ELEMENT_D(A, 0, 0) = m;
  SM_ELEMENT_D(A, 0, 2) = x / l;

  SM_ELEMENT_D(A, 1, 1) = m;
  SM_ELEMENT_D(A, 1, 2) = y / l;

  SM_ELEMENT_D(A, 2, 0) = TWO * x;
  SM_ELEMENT_D(A, 2, 1) = TWO * y;

  return 0;
}

/* --------------------------------------------------------------------------
 * Linear system from: S. E. Mattsson and G. Söderlind, “Index Reduction in
 * Differential-Algebraic Equations Using Dummy Derivatives,” SIAM J. Sci.
 * Comput., vol. 14, no. 3, pp. 677–692, May 1993, doi: 10.1137/0914043.
 * -------------------------------------------------------------------------- */

static const sunindextype LINSYS_N = 4;
static const uint8_t LINSYS_C[]    = {2, 2, 1, 0};
static const uint8_t LINSYS_D[]    = {2, 2, 2, 1};

static const sunindextype* LINSYS_VAR_IDX_MAP[] = {
  (const sunindextype[]){0, 1, 2},
  (const sunindextype[]){3, 4, 5},
  (const sunindextype[]){6, 7, 8},
  (const sunindextype[]){9, 10},
};

static const sunindextype LINSYS_JAC_NNZ = 23;

int LinsysJacf0(SUNDIALS_MAYBE_UNUSED sunrealtype t,
                SUNDIALS_MAYBE_UNUSED N_Vector Y,
                SUNMatrix A,
                SUNDIALS_MAYBE_UNUSED void* user_data)
{
  SM_ELEMENT_D(A, 0, 0) = ONE;
  SM_ELEMENT_D(A, 0, 1) = ONE;
  SM_ELEMENT_D(A, 1, 0) = ONE;
  SM_ELEMENT_D(A, 1, 1) = ONE;
  SM_ELEMENT_D(A, 1, 2) = ONE;
  SM_ELEMENT_D(A, 2, 2) = ONE;
  SM_ELEMENT_D(A, 2, 3) = ONE;
  SM_ELEMENT_D(A, 3, 0) = TWO;
  SM_ELEMENT_D(A, 3, 1) = ONE;
  SM_ELEMENT_D(A, 3, 2) = ONE;
  SM_ELEMENT_D(A, 3, 3) = ONE;
  return 0;
}

/* --------------------------------------------------------------------------
 * Non-linear system from: R. McKenzie and J. Pryce, “Structural analysis based
 * dummy derivative selection for differential algebraic equations,” Bit Numer
 * Math, vol. 57, no. 2, pp. 433–462, Jun. 2017, doi: 10.1007/s10543-016-0642-9.
 * -------------------------------------------------------------------------- */

static const sunindextype NONLINSYS_N = 5;
static const uint8_t NONLINSYS_C[]    = {1, 0, 2, 2, 1};
static const uint8_t NONLINSYS_D[]    = {3, 2, 2, 2, 2};

static const sunindextype* NONLINSYS_VAR_IDX_MAP[] = {
  (const sunindextype[]){0, 1, 2, 3},
  (const sunindextype[]){4, 5, 6},
  (const sunindextype[]){7, 8, 9},
  (const sunindextype[]){10, 11, 12},
  (const sunindextype[]){13, 14, 15},
};

static const sunindextype NONLINSYS_JAC_NNZ = 39;

int NonlinsysJacf0(SUNDIALS_MAYBE_UNUSED sunrealtype t,
                   N_Vector Y,
                   SUNMatrix A,
                   SUNDIALS_MAYBE_UNUSED void* user_data)
{
  const sunrealtype d2x1 = NV_Ith(Y, NONLINSYS_VAR_IDX_MAP[0][2]);
  const sunrealtype dx1  = NV_Ith(Y, NONLINSYS_VAR_IDX_MAP[0][1]);
  const sunrealtype d2x2 = NV_Ith(Y, NONLINSYS_VAR_IDX_MAP[1][2]);
  const sunrealtype dx2  = NV_Ith(Y, NONLINSYS_VAR_IDX_MAP[1][1]);
  const sunrealtype x3   = NV_Ith(Y, NONLINSYS_VAR_IDX_MAP[2][0]);
  const sunrealtype d2x4 = NV_Ith(Y, NONLINSYS_VAR_IDX_MAP[3][2]);
  const sunrealtype x4   = NV_Ith(Y, NONLINSYS_VAR_IDX_MAP[3][0]);
  const sunrealtype dx5  = NV_Ith(Y, NONLINSYS_VAR_IDX_MAP[4][1]);

  SM_ELEMENT_D(A, 0, 0) = TWO * d2x1;
  SM_ELEMENT_D(A, 2, 0) = TWO * dx1;
  SM_ELEMENT_D(A, 1, 1) = -TWO * d2x2;
  SM_ELEMENT_D(A, 4, 1) = -TWO * dx2;
  SM_ELEMENT_D(A, 2, 2) = -TWO * x3;
  SM_ELEMENT_D(A, 3, 2) = -TWO * x3;
  SM_ELEMENT_D(A, 1, 3) = FOUR * d2x4;
  SM_ELEMENT_D(A, 3, 3) = FOUR * x4;
  SM_ELEMENT_D(A, 0, 4) = -FOUR * dx5;
  SM_ELEMENT_D(A, 4, 4) = FOUR * dx5;
  return 0;
}

#endif
