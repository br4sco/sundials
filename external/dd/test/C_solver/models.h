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

#define ZERO   SUN_RCONST(0.0)
#define ONE    SUN_RCONST(1.0)
#define TWO    SUN_RCONST(2.0)
#define FOUR   SUN_RCONST(4.0)
#define NV_Ith NV_Ith_S

#define SET_SPARSE(A, nnz, i, expr) \
  do {                              \
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

int LotkaVolterraRes(
  SUNDIALS_MAYBE_UNUSED sunrealtype t,
  N_Vector Y, /* {x, x', y, y'} */
  N_Vector R, /* {f₁, f₂, ...} */
  void* user_data
)
{
  const LotkaVolterraParams p = *((LotkaVolterraParams*)user_data);

  const sunrealtype x = LV_Ith(Y, 0, 0), xp = LV_Ith(Y, 0, 1),

                    y = LV_Ith(Y, 1, 0), yp = LV_Ith(Y, 1, 1);

  NV_Ith(R, 0) = xp - p.a * x + p.b * x * y;
  NV_Ith(R, 1) = yp - p.d * x * y + p.c * y;

  return 0;
}

int LotkaVolterraResS(
  int Ns,
  SUNDIALS_MAYBE_UNUSED sunrealtype t,
  N_Vector Y,
  SUNDIALS_MAYBE_UNUSED N_Vector R,
  N_Vector* YS,
  SUNDIALS_MAYBE_UNUSED N_Vector* RS,
  void* user_data,
  SUNDIALS_MAYBE_UNUSED N_Vector tmp1,
  SUNDIALS_MAYBE_UNUSED N_Vector tmp2,
  SUNDIALS_MAYBE_UNUSED N_Vector tmp3
)
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

int LotkaVolterraJacf0(
  SUNDIALS_MAYBE_UNUSED sunrealtype t,
  SUNDIALS_MAYBE_UNUSED N_Vector Y,
  SUNMatrix A,
  SUNDIALS_MAYBE_UNUSED void* user_data
)
{
  SM_ELEMENT_D(A, 0, 0) = ONE;
  SM_ELEMENT_D(A, 1, 1) = ONE;

  return 0;
}

int LotkaVolterraJacfn_Dense(
  SUNDIALS_MAYBE_UNUSED sunrealtype t,
  N_Vector Y,
  SUNDIALS_MAYBE_UNUSED N_Vector R,
  SUNMatrix J,
  void* user_data,
  SUNDIALS_MAYBE_UNUSED N_Vector tmp1,
  SUNDIALS_MAYBE_UNUSED N_Vector tmp2,
  SUNDIALS_MAYBE_UNUSED N_Vector tmp3
)
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

int LotkaVolterraJacfn_CSR(
  SUNDIALS_MAYBE_UNUSED sunrealtype t,
  N_Vector Y,
  SUNDIALS_MAYBE_UNUSED N_Vector R,
  SUNMatrix J,
  void* user_data,
  SUNDIALS_MAYBE_UNUSED N_Vector tmp1,
  SUNDIALS_MAYBE_UNUSED N_Vector tmp2,
  SUNDIALS_MAYBE_UNUSED N_Vector tmp3
)
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

static inline int LotkaVolterraJacColFn_CSC(
  sunindextype j,
  SUNDIALS_MAYBE_UNUSED sunrealtype t,
  N_Vector Y,
  SUNDIALS_MAYBE_UNUSED N_Vector R,
  SUNMatrix J,
  sunindextype* nnz,
  void* user_data,
  SUNDIALS_MAYBE_UNUSED N_Vector tmp1,
  SUNDIALS_MAYBE_UNUSED N_Vector tmp2,
  SUNDIALS_MAYBE_UNUSED N_Vector tmp3
)
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
  else { return -1; }

  return 0;
}

int LotkaVolterraJacfn_CSC(
  const sunindextype yy_diff_alias_row[static 1],
  const sunindextype yp_diff_alias_row[static 1],
  SUNDIALS_MAYBE_UNUSED sunrealtype t,
  sunrealtype cj,
  N_Vector Y,
  SUNDIALS_MAYBE_UNUSED N_Vector R,
  SUNMatrix J,
  void* user_data,
  SUNDIALS_MAYBE_UNUSED N_Vector tmp1,
  SUNDIALS_MAYBE_UNUSED N_Vector tmp2,
  SUNDIALS_MAYBE_UNUSED N_Vector tmp3
)
{
  return DDJacFn_CSC(2, LotkaVolterraJacColFn_CSC, yy_diff_alias_row, yp_diff_alias_row, t, cj, Y, R, J, user_data, tmp1, tmp2, tmp3);
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

static const sunrealtype PENDULUM_ADJ_ID[] = {ONE, ONE, ONE, ONE, ONE};
static const sunindextype PENDULUM_JAC_NNZ = 18;

#define P_Ith(Y, i, k) (NV_Ith(Y, PENDULUM_VAR_IDX_MAP[i][k]))

static const uint8_t PENDULUM_NP = 2;

typedef struct
{
  sunrealtype m;        /* Arm mass. */
  sunrealtype param[2]; /* Arm length, Gravitational constant */
} PendulumData;

int PendulumRes(
  SUNDIALS_MAYBE_UNUSED sunrealtype t,
  N_Vector Y, /* {x, x', x'', y, y', y'', λ} */
  N_Vector R, /* {f₁, f₂, f₃, f₃', f₃'', reserved} */
  SUNDIALS_MAYBE_UNUSED void* user_data
)
{
  const sunrealtype x = P_Ith(Y, 0, 0), xp = P_Ith(Y, 0, 1),
                    xpp = P_Ith(Y, 0, 2),

                    y = P_Ith(Y, 1, 0), yp = P_Ith(Y, 1, 1),
                    ypp = P_Ith(Y, 1, 2),

                    lambda = P_Ith(Y, 2, 0);

  PendulumData* data  = (PendulumData*)user_data;
  const sunrealtype m = data->m, l = data->param[0], g = data->param[1];

  NV_Ith(R, 0) = m * xpp + lambda / l * x;
  NV_Ith(R, 1) = m * ypp + lambda / l * y + m * g;
  NV_Ith(R, 2) = x * x + y * y - l * l;
  NV_Ith(R, 3) = TWO * (x * xp + y * yp);
  NV_Ith(R, 4) = TWO * (x * xpp + xp * xp + y * ypp + yp * yp);

  return 0;
}

int PendulumJacfn_Dense(
  SUNDIALS_MAYBE_UNUSED sunrealtype t,
  N_Vector Y,
  SUNDIALS_MAYBE_UNUSED N_Vector R,
  SUNMatrix J,
  void* user_data,
  SUNDIALS_MAYBE_UNUSED N_Vector tmp1,
  SUNDIALS_MAYBE_UNUSED N_Vector tmp2,
  SUNDIALS_MAYBE_UNUSED N_Vector tmp3
)
{
  const sunrealtype x = P_Ith(Y, 0, 0), xp = P_Ith(Y, 0, 1),
                    xpp = P_Ith(Y, 0, 2),

                    y = P_Ith(Y, 1, 0), yp = P_Ith(Y, 1, 1),
                    ypp = P_Ith(Y, 1, 2),

                    lambda = P_Ith(Y, 2, 0);

  PendulumData* data  = (PendulumData*)user_data;
  const sunrealtype m = data->m, l = data->param[0];

  /* column 0 */
  SM_ELEMENT_D(J, 0, 0) = lambda / l;
  SM_ELEMENT_D(J, 2, 0) = TWO * x;
  SM_ELEMENT_D(J, 3, 0) = TWO * xp;
  SM_ELEMENT_D(J, 4, 0) = TWO * xpp;

  /* column 1 */
  SM_ELEMENT_D(J, 3, 1) = TWO * x;
  SM_ELEMENT_D(J, 4, 1) = FOUR * xp;

  /* column 2 */
  SM_ELEMENT_D(J, 0, 2) = m;
  SM_ELEMENT_D(J, 4, 2) = TWO * x;

  /* column 3 */
  SM_ELEMENT_D(J, 1, 3) = lambda / l;
  SM_ELEMENT_D(J, 2, 3) = TWO * y;
  SM_ELEMENT_D(J, 3, 3) = TWO * yp;
  SM_ELEMENT_D(J, 4, 3) = TWO * ypp;

  /* column 4 */
  SM_ELEMENT_D(J, 3, 4) = TWO * y;
  SM_ELEMENT_D(J, 4, 4) = FOUR * yp;

  /* column 6 */
  SM_ELEMENT_D(J, 1, 5) = m;
  SM_ELEMENT_D(J, 4, 5) = TWO * y;

  /* column 6 */
  SM_ELEMENT_D(J, 0, 6) = x / l;
  SM_ELEMENT_D(J, 1, 6) = y / l;

  return 0;
}

int PendulumJacfn_CSR(
  SUNDIALS_MAYBE_UNUSED sunrealtype t,
  N_Vector Y,
  SUNDIALS_MAYBE_UNUSED N_Vector R,
  SUNMatrix J,
  void* user_data,
  SUNDIALS_MAYBE_UNUSED N_Vector tmp1,
  SUNDIALS_MAYBE_UNUSED N_Vector tmp2,
  SUNDIALS_MAYBE_UNUSED N_Vector tmp3
)
{
  const sunrealtype x = P_Ith(Y, 0, 0), xp = P_Ith(Y, 0, 1),
                    xpp = P_Ith(Y, 0, 2),

                    y = P_Ith(Y, 1, 0), yp = P_Ith(Y, 1, 1),
                    ypp = P_Ith(Y, 1, 2),

                    lambda = P_Ith(Y, 2, 0);

  PendulumData* data  = (PendulumData*)user_data;
  const sunrealtype m = data->m, l = data->param[0];

  sunindextype nnz = 0;

  /* row 0 */
  SM_INDEXPTRS_S(J)[0] = nnz;
  SET_SPARSE(J, nnz, 0, lambda / l);
  SET_SPARSE(J, nnz, 2, m);
  SET_SPARSE(J, nnz, 6, x / l);

  /* row 1 */
  SM_INDEXPTRS_S(J)[1] = nnz;
  SET_SPARSE(J, nnz, 3, lambda / l);
  SET_SPARSE(J, nnz, 5, m);
  SET_SPARSE(J, nnz, 6, y / l);

  /* row 2 */
  SM_INDEXPTRS_S(J)[2] = nnz;
  SET_SPARSE(J, nnz, 0, TWO * x);
  SET_SPARSE(J, nnz, 3, TWO * y);

  /* row 3 */
  SM_INDEXPTRS_S(J)[3] = nnz;
  SET_SPARSE(J, nnz, 0, TWO * xp);
  SET_SPARSE(J, nnz, 1, TWO * x);
  SET_SPARSE(J, nnz, 3, TWO * yp);
  SET_SPARSE(J, nnz, 4, TWO * y);

  /* row 4 */
  SM_INDEXPTRS_S(J)[4] = nnz;
  SET_SPARSE(J, nnz, 0, TWO * xpp);
  SET_SPARSE(J, nnz, 1, FOUR * xp);
  SET_SPARSE(J, nnz, 2, TWO * x);
  SET_SPARSE(J, nnz, 3, TWO * ypp);
  SET_SPARSE(J, nnz, 4, FOUR * yp);
  SET_SPARSE(J, nnz, 5, TWO * y);

  SM_INDEXPTRS_S(J)[5] = nnz;

  return 0;
}

static inline int PendulumJacColFn_CSC(
  sunindextype j,
  SUNDIALS_MAYBE_UNUSED sunrealtype t,
  N_Vector Y,
  SUNDIALS_MAYBE_UNUSED N_Vector R,
  SUNMatrix J,
  sunindextype* nnz,
  void* user_data,
  SUNDIALS_MAYBE_UNUSED N_Vector tmp1,
  SUNDIALS_MAYBE_UNUSED N_Vector tmp2,
  SUNDIALS_MAYBE_UNUSED N_Vector tmp3
)
{
  const sunrealtype x = P_Ith(Y, 0, 0), xp = P_Ith(Y, 0, 1),
                    xpp = P_Ith(Y, 0, 2),

                    y = P_Ith(Y, 1, 0), yp = P_Ith(Y, 1, 1),
                    ypp = P_Ith(Y, 1, 2),

                    lambda = P_Ith(Y, 2, 0);

  PendulumData* data  = (PendulumData*)user_data;
  const sunrealtype m = data->m, l = data->param[0];

  switch (j)
  {
  case 0:
    SET_SPARSE(J, *nnz, 0, lambda / l);
    SET_SPARSE(J, *nnz, 2, TWO * x);
    SET_SPARSE(J, *nnz, 3, TWO * xp);
    SET_SPARSE(J, *nnz, 4, TWO * xpp);
    break;
  case 1:
    SET_SPARSE(J, *nnz, 3, TWO * x);
    SET_SPARSE(J, *nnz, 4, FOUR * xp);
    break;
  case 2:
    SET_SPARSE(J, *nnz, 0, m);
    SET_SPARSE(J, *nnz, 4, TWO * x);
    break;
  case 3:
    SET_SPARSE(J, *nnz, 1, lambda / l);
    SET_SPARSE(J, *nnz, 2, TWO * y);
    SET_SPARSE(J, *nnz, 3, TWO * yp);
    SET_SPARSE(J, *nnz, 4, TWO * ypp);
    break;
  case 4:
    SET_SPARSE(J, *nnz, 3, TWO * y);
    SET_SPARSE(J, *nnz, 4, FOUR * yp);
    break;
  case 5:
    SET_SPARSE(J, *nnz, 1, m);
    SET_SPARSE(J, *nnz, 4, TWO * y);
    break;
  case 6:
    SET_SPARSE(J, *nnz, 0, x / l);
    SET_SPARSE(J, *nnz, 1, y / l);
    break;
  }

  return 0;
}

int PendulumJacfn_CSC(
  const sunindextype yy_diff_alias_row[static 1],
  const sunindextype yp_diff_alias_row[static 1],
  SUNDIALS_MAYBE_UNUSED sunrealtype t,
  sunrealtype cj,
  N_Vector Y,
  N_Vector R,
  SUNMatrix J,
  void* user_data,
  SUNDIALS_MAYBE_UNUSED N_Vector tmp1,
  SUNDIALS_MAYBE_UNUSED N_Vector tmp2,
  SUNDIALS_MAYBE_UNUSED N_Vector tmp3
)
{
  return DDJacFn_CSC(5, PendulumJacColFn_CSC, yy_diff_alias_row, yp_diff_alias_row, t, cj, Y, R, J, user_data, tmp1, tmp2, tmp3);
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

int PendulumResS(
  int Ns,
  SUNDIALS_MAYBE_UNUSED sunrealtype t,
  N_Vector Y,
  SUNDIALS_MAYBE_UNUSED N_Vector R,
  N_Vector* YS,
  N_Vector* RS,
  void* user_data,
  SUNDIALS_MAYBE_UNUSED N_Vector tmp1,
  SUNDIALS_MAYBE_UNUSED N_Vector tmp2,
  SUNDIALS_MAYBE_UNUSED N_Vector tmp3
)
{
  assert(Ns == PENDULUM_NP);
  assert(user_data);
  PendulumData* data  = (PendulumData*)user_data;
  sunrealtype const m = data->m, l = data->param[0];

  const sunrealtype x = P_Ith(Y, 0, 0), xp = P_Ith(Y, 0, 1),
                    xpp = P_Ith(Y, 0, 2),

                    y = P_Ith(Y, 1, 0), yp = P_Ith(Y, 1, 1),
                    ypp = P_Ith(Y, 1, 2),

                    lambda = P_Ith(Y, 2, 0);

  for (int i = 0; i < Ns; ++i)
  {
    const sunrealtype xS = P_Ith(YS[i], 0, 0), xpS = P_Ith(YS[i], 0, 1),
                      xppS = P_Ith(YS[i], 0, 2),

                      yS = P_Ith(YS[i], 1, 0), ypS = P_Ith(YS[i], 1, 1),
                      yppS = P_Ith(YS[i], 1, 2),

                      lambdaS = P_Ith(YS[i], 2, 0);

    NV_Ith(RS[i], 0) = lambda * xS / l + m * xppS + x * lambdaS / l;
    NV_Ith(RS[i], 1) = lambda * yS / l + m * yppS + y * lambdaS / l;
    NV_Ith(RS[i], 2) = TWO * (x * xS + y * yS);
    NV_Ith(RS[i], 3) = TWO * (xp * xS + x * xpS + yp * yS + y * ypS);
    NV_Ith(RS[i], 4) = TWO * (xpp * xS + TWO * xp * xpS + x * xppS + ypp * yS +
                              TWO * yp * ypS + y * yppS);
  }

  NV_Ith(RS[0], 0) -= lambda * x / (l * l);
  NV_Ith(RS[0], 1) -= lambda * y / (l * l);
  NV_Ith(RS[0], 2) -= TWO * l;
  NV_Ith(RS[1], 1) += m;

  return 0;
}

void PendulumYyBT(
  SUNDIALS_MAYBE_UNUSED PendulumData* data,
  SUNDIALS_MAYBE_UNUSED N_Vector Y,
  N_Vector yyBT,
  N_Vector ypBT
)
{
  N_VConst(ZERO, yyBT);
  N_VConst(ZERO, ypBT);
  NV_Ith(ypBT, 4) = ONE;
}

sunrealtype PendulumG(SUNDIALS_MAYBE_UNUSED sunrealtype t, N_Vector Y, void* user_data)
{
  assert(user_data);
  PendulumData* data = (PendulumData*)user_data;
  sunrealtype l      = data->param[0];
  return P_Ith(Y, 0, 0) + l;
}

int PendulumResB(
  SUNDIALS_MAYBE_UNUSED sunrealtype t,
  N_Vector Y,
  N_Vector yyB,
  N_Vector ypB,
  N_Vector rrB,
  void* user_data
)
{
  assert(user_data);
  PendulumData* data = (PendulumData*)user_data;
  sunrealtype m = data->m, l = data->param[0];

  const sunrealtype x = P_Ith(Y, 0, 0), xp = P_Ith(Y, 0, 1),
                    xpp = P_Ith(Y, 0, 2),

                    y = P_Ith(Y, 1, 0), yp = P_Ith(Y, 1, 1),
                    ypp = P_Ith(Y, 1, 2),

                    lambda = P_Ith(Y, 2, 0),

                    l1 = NV_Ith(yyB, 0), l2 = NV_Ith(yyB, 1),
                    l3 = NV_Ith(yyB, 2), l4 = NV_Ith(yyB, 3),
                    l5 = NV_Ith(yyB, 4),

                    l1p = NV_Ith(ypB, 0), l2p = NV_Ith(ypB, 1),
                    l3p = NV_Ith(ypB, 2), l4p = NV_Ith(ypB, 3),
                    l5p = NV_Ith(ypB, 4);

  NV_Ith(rrB, 0) = -l1 * lambda / l - TWO * xpp * l3 - l4p;
  NV_Ith(rrB, 1) = m * l1p - TWO * l3 * xp + TWO * x * l3p - l4;
  NV_Ith(rrB, 2) = -lambda * l2 / l - TWO * ypp * l3 - l5p + ONE;
  NV_Ith(rrB, 3) = m * l2p - TWO * l3 * yp + TWO * y * l3p - l5;
  NV_Ith(rrB, 4) = (-x * l1 - y * l2) / l;

  return 0;
}

int PendulumJacf0(
  SUNDIALS_MAYBE_UNUSED sunrealtype t,
  N_Vector Y,
  SUNMatrix A,
  SUNDIALS_MAYBE_UNUSED void* user_data
)
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

#endif
