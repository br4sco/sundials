#ifndef _DD_MODELS_H
#define _DD_MODELS_H

#include <math.h>
#include <sundials/sundials_core.h>
#include <sundials/sundials_macros.h>
#include <sunmatrix/sunmatrix_dense.h>

#include "matrix.h"
#include "sundials/sundials_nvector.h"

/**
 * @file
 * @brief DAE models.
 */

#define ZERO SUN_RCONST(0.0)
#define ONE  SUN_RCONST(1.0)
#define TWO  SUN_RCONST(2.0)
#define FOUR SUN_RCONST(4.0)

/* --------------------------------------------------------------------------
 * Lotka-Volterra predator-prey model (ODE)
 * -------------------------------------------------------------------------- */

static const sunindextype LOTKA_VOLTERRA_N = 2;
static const uint8_t LOTKA_VOLTERRA_C[]    = {0, 0}; /* f₁, f₂ */
static const uint8_t LOTKA_VOLTERRA_D[]    = {1, 1}; /* x', y' */

#define LV_X(Y, l) (N_VGetArrayPointer(Y)[l])
#define LV_Y(Y, l) (N_VGetArrayPointer(Y)[2 + l])
#define LV_R1(R)   (N_VGetArrayPointer(R)[0])
#define LV_R2(R)   (N_VGetArrayPointer(R)[1])

static const uint8_t LOTKA_VOLTERRA_NP = 4;

typedef struct
{
  sunrealtype a; /* Prey per-capita growth rate. */
  sunrealtype b; /* Prey death rate. */
  sunrealtype c; /* Predator per-capita growth rate. */
  sunrealtype d; /* Preadotr death rate. */
} LotkaVolterraParams;

int LotkaVolterraRes(SUNDIALS_MAYBE_UNUSED sunrealtype tt,
                     N_Vector Y, /* {x, x', y, y'} */
                     N_Vector R, /* {f₁, f₂, reserved} */
                     void* user_data)
{
  const LotkaVolterraParams p = *((LotkaVolterraParams*)user_data);
  const sunrealtype x = LV_X(Y, 0), dx = LV_X(Y, 1), y = LV_Y(Y, 0),
                    dy = LV_Y(Y, 1);

  LV_R1(R) = dx - p.a * x + p.b * x * y;
  LV_R2(R) = dy - p.d * x * y + p.c * y;

  return 0;
}

int LotkaVolterraResS(int Ns, SUNDIALS_MAYBE_UNUSED sunrealtype t, N_Vector Y,
                      SUNDIALS_MAYBE_UNUSED N_Vector R, N_Vector* YS,
                      SUNDIALS_MAYBE_UNUSED N_Vector* RS, void* user_data,
                      SUNDIALS_MAYBE_UNUSED N_Vector tmp1,
                      SUNDIALS_MAYBE_UNUSED N_Vector tmp2,
                      SUNDIALS_MAYBE_UNUSED N_Vector tmp3)
{
  assert(Ns == LOTKA_VOLTERRA_NP);

  LotkaVolterraParams p = *((LotkaVolterraParams*)user_data);
  sunrealtype x = LV_X(Y, 0), y = LV_Y(Y, 0);

  for (int i = 0; i < Ns; ++i)
  {
    N_Vector v = YS[i];
    N_Vector r = RS[i];

    sunrealtype xS = LV_X(v, 0), xpS = LV_X(v, 1), yS = LV_Y(v, 0),
                ypS = LV_Y(v, 1);

    LV_R1(r) = (-p.a + p.b * y) * xS + xpS + p.b * x * yS;
    LV_R2(r) = -p.d * y * xS + (-p.d * x + p.c) * yS + ypS;
  }

  LV_R1(RS[0]) -= x;
  LV_R1(RS[1]) += x * y;
  LV_R2(RS[2]) += y;
  LV_R2(RS[3]) -= x * y;

  return 0;
}

int LotkaVolterraJacf0(SUNDIALS_MAYBE_UNUSED sunrealtype t,
                       SUNDIALS_MAYBE_UNUSED N_Vector Y,
                       ExtSUNMatrix jac0[static 1],
                       SUNDIALS_MAYBE_UNUSED void* user_data)
{
  SUNMatrix J0 = ExtSUNMatGetMat(jac0);

  SM_ELEMENT_D(J0, 0, 0) = ONE;
  SM_ELEMENT_D(J0, 1, 1) = ONE;

  return 0;
}

int LotkaVolterraJacf(SUNDIALS_MAYBE_UNUSED sunrealtype t, N_Vector Y,
                      SUNDIALS_MAYBE_UNUSED N_Vector R, SUNMatrix J,
                      void* user_data, SUNDIALS_MAYBE_UNUSED N_Vector tmp1,
                      SUNDIALS_MAYBE_UNUSED N_Vector tmp2,
                      SUNDIALS_MAYBE_UNUSED N_Vector tmp3)
{
  LotkaVolterraParams p = *((LotkaVolterraParams*)user_data);
  const sunrealtype x = LV_X(Y, 0), y = LV_Y(Y, 0);

  SM_ELEMENT_D(J, 0, 0) = -p.a + p.b * y;
  SM_ELEMENT_D(J, 1, 0) = -p.d * y;

  SM_ELEMENT_D(J, 0, 1) = ONE;

  SM_ELEMENT_D(J, 0, 2) = p.b * x;
  SM_ELEMENT_D(J, 1, 2) = -p.d * x + p.c;

  SM_ELEMENT_D(J, 1, 3) = ONE;

  return 0;
}

/* --------------------------------------------------------------------------
 * Pendulum
 * -------------------------------------------------------------------------- */

static char* PENDULUM_EQN_NAMES[]          = {"f₁", "f₂", "f₃"};
static char* PENDULUM_VAR_NAMES[]          = {"x", "y", "λ"};
static const sunindextype PENDULUM_N       = 3;
static const uint8_t PENDULUM_C[]          = {0, 0, 2}; /* f₁, f₂, f₃'' */
static const uint8_t PENDULUM_D[]          = {2, 2, 0}; /* x'', y'', λ */
static const sunrealtype PENDULUM_ADJ_ID[] = {ONE, ONE, ONE, ONE, ONE};

#define P_X(Y, l)   (N_VGetArrayPointer(Y)[l])
#define P_Y(Y, l)   (N_VGetArrayPointer(Y)[3 + l])
#define P_LAMBDA(Y) (N_VGetArrayPointer(Y)[6])
#define P_R1(R)     (N_VGetArrayPointer(R)[0])
#define P_R2(R)     (N_VGetArrayPointer(R)[1])
#define P_R3(R, l)  (N_VGetArrayPointer(R)[2 + l])

static const uint8_t PENDULUM_NP = 2;

typedef struct
{
  sunrealtype m;        /* Arm mass. */
  sunrealtype param[2]; /* Arm length, Gravitational constant */
} PendulumData;

int PendulumRes(SUNDIALS_MAYBE_UNUSED sunrealtype t,
                N_Vector Y, /* {x, x', x'', y, y', y'', λ} */
                N_Vector R, /* {f₁, f₂, f₃, f₃', f₃'', reserved} */
                SUNDIALS_MAYBE_UNUSED void* user_data)
{
  const sunrealtype x = P_X(Y, 0), xp = P_X(Y, 1), xpp = P_X(Y, 2),
                    y = P_Y(Y, 0), yp = P_Y(Y, 1), ypp = P_Y(Y, 2),
                    lambda = P_LAMBDA(Y);

  PendulumData* data  = (PendulumData*)user_data;
  const sunrealtype m = data->m, l = data->param[0], g = data->param[1];

  P_R1(R)    = m * xpp + lambda / l * x;
  P_R2(R)    = m * ypp + lambda / l * y + m * g;
  P_R3(R, 0) = x * x + y * y - l * l;
  P_R3(R, 1) = TWO * (x * xp + y * yp);
  P_R3(R, 2) = TWO * (x * xpp + xp * xp + y * ypp + yp * yp);

  return 0;
}

void PendulumY0(PendulumData* data, sunrealtype theta0, N_Vector Y)
{
  const sunrealtype m = data->m, l = data->param[0], g = data->param[1];
  N_VConst(ZERO, Y);
  P_X(Y, 0)   = l * sin(theta0);
  P_Y(Y, 0)   = -l * cos(theta0);
  P_X(Y, 2)   = -g * sin(TWO * theta0) / TWO;
  P_Y(Y, 2)   = -g * sin(theta0) * sin(theta0);
  P_LAMBDA(Y) = m * g * cos(theta0);
}

void PendulumYS0(PendulumData* data, sunrealtype theta0, N_Vector* YS)
{
  const sunrealtype m = data->m;
  for (int i = 0; i < PENDULUM_NP; ++i) { N_VConst(ZERO, YS[i]); }
  P_X(YS[0], 0)   = sin(theta0);
  P_Y(YS[0], 0)   = -cos(theta0);
  P_X(YS[1], 2)   = -sin(TWO * theta0) / TWO;
  P_Y(YS[1], 2)   = -sin(theta0) * sin(theta0);
  P_LAMBDA(YS[1]) = m * cos(theta0);
}

int PendulumResS(int Ns, SUNDIALS_MAYBE_UNUSED sunrealtype t, N_Vector Y,
                 SUNDIALS_MAYBE_UNUSED N_Vector R, N_Vector* YS, N_Vector* RS,
                 void* user_data, SUNDIALS_MAYBE_UNUSED N_Vector tmp1,
                 SUNDIALS_MAYBE_UNUSED N_Vector tmp2,
                 SUNDIALS_MAYBE_UNUSED N_Vector tmp3)
{
  assert(Ns == PENDULUM_NP);
  assert(user_data);
  PendulumData* data  = (PendulumData*)user_data;
  sunrealtype const m = data->m, l = data->param[0];

  sunrealtype const x = P_X(Y, 0), dx = P_X(Y, 1), ddx = P_X(Y, 2),
                    y = P_Y(Y, 0), dy = P_Y(Y, 1), ddy = P_Y(Y, 2),
                    lambda = P_LAMBDA(Y);

  for (int i = 0; i < Ns; ++i)
  {
    N_Vector YSi = YS[i];
    N_Vector RSi = RS[i];

    const sunrealtype xS = P_X(YSi, 0), dxS = P_X(YSi, 1), ddxS = P_X(YSi, 2),
                      yS = P_Y(YSi, 0), dyS = P_Y(YSi, 1), ddyS = P_Y(YSi, 2),
                      lambdaS = P_LAMBDA(YSi);

    P_R1(RSi)    = lambda * xS / l + m * ddxS + x * lambdaS / l;
    P_R2(RSi)    = lambda * yS / l + m * ddyS + y * lambdaS / l;
    P_R3(RSi, 0) = TWO * (x * xS + y * yS);
    P_R3(RSi, 1) = TWO * (dx * xS + x * dxS + dy * yS + y * dyS);
    P_R3(RSi, 2) = TWO * (ddx * xS + TWO * dx * dxS + x * ddxS + ddy * yS +
                          TWO * dy * dyS + y * ddyS);
  }

  P_R1(RS[0]) -= lambda * x / (l * l);
  P_R2(RS[0]) -= lambda * y / (l * l);
  P_R3(RS[0], 0) -= TWO * l;

  P_R2(RS[1]) += m;

  return 0;
}

void PendulumYyBT(SUNDIALS_MAYBE_UNUSED PendulumData* data,
                  SUNDIALS_MAYBE_UNUSED N_Vector Y, N_Vector yyBT, N_Vector ypBT)
{
  N_VConst(ZERO, yyBT);
  N_VConst(ZERO, ypBT);
  sunrealtype* ypBT_arr = N_VGetArrayPointer(ypBT);

  ypBT_arr[4] = ONE;
}

sunrealtype PendulumG(SUNDIALS_MAYBE_UNUSED sunrealtype t, N_Vector Y,
                      void* user_data)
{
  assert(user_data);
  PendulumData* data = (PendulumData*)user_data;
  sunrealtype l      = data->param[0];
  return P_X(Y, 0) + l;
}

int PendulumResB(SUNDIALS_MAYBE_UNUSED sunrealtype t, N_Vector Y, N_Vector yyB,
                 N_Vector ypB, N_Vector rrB, void* user_data)
{
  assert(user_data);
  PendulumData* data = (PendulumData*)user_data;
  sunrealtype m = data->m, l = data->param[0];

  sunrealtype* yyB_arr = N_VGetArrayPointer(yyB);
  sunrealtype* ypB_arr = N_VGetArrayPointer(ypB);

  const sunrealtype x = P_X(Y, 0), dx = P_X(Y, 1), ddx = P_X(Y, 2),

                    y = P_Y(Y, 0), dy = P_Y(Y, 1), ddy = P_Y(Y, 2),

                    lambda = P_LAMBDA(Y),

                    l1 = yyB_arr[0], l2 = yyB_arr[1], l3 = yyB_arr[2],
                    l4 = yyB_arr[3], l5 = yyB_arr[4],

                    dl1 = ypB_arr[0], dl2 = ypB_arr[1], dl3 = ypB_arr[2],
                    dl4 = ypB_arr[3], dl5 = ypB_arr[4];

  sunrealtype* r = N_VGetArrayPointer(rrB);

  r[0] = -l1 * lambda / l - TWO * ddx * l3 - dl4;
  r[1] = m * dl1 - TWO * l3 * dx + TWO * x * dl3 - l4;
  r[2] = -lambda * l2 / l - TWO * ddy * l3 - dl5 + ONE;
  r[3] = m * dl2 - TWO * l3 * dy + TWO * y * dl3 - l5;
  r[4] = (-x * l1 - y * l2) / l;

  return 0;
}

int PendulumJacf0(SUNDIALS_MAYBE_UNUSED sunrealtype t, N_Vector Y,
                  ExtSUNMatrix J0[static 1],
                  SUNDIALS_MAYBE_UNUSED void* user_data)
{
  PendulumData* data = (PendulumData*)user_data;
  sunrealtype m = data->m, l = data->param[0];

  sunrealtype x = P_X(Y, 0), y = P_Y(Y, 0);
  SUNMatrix A = ExtSUNMatGetMat(J0);

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

/* --------------------------------------------------------------------------
 * Non-linear system from: R. McKenzie and J. Pryce, “Structural analysis based
 * dummy derivative selection for differential algebraic equations,” Bit Numer
 * Math, vol. 57, no. 2, pp. 433–462, Jun. 2017, doi: 10.1007/s10543-016-0642-9.
 * -------------------------------------------------------------------------- */

static const sunindextype NONLINSYS_N = 5;
static const uint8_t NONLINSYS_C[]    = {1, 0, 2, 2, 1};
static const uint8_t NONLINSYS_D[]    = {3, 2, 2, 2, 2};

#endif
