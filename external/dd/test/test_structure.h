#include <math.h>
#include <sundials/sundials_core.h>
#include <sunmatrix/sunmatrix_dense.h>

#include "models.h"

#define ZERO SUN_RCONST(0.0)
#define ONE  SUN_RCONST(1.0)
#define TWO  SUN_RCONST(2.0)

SUNContext CTX;

/* --------------------------------------------------------------------------
 * Lotka-Volterra
 * -------------------------------------------------------------------------- */

static SUNMatrix jac_lotka_volterra_create(void)
{
  SUNMatrix mat = SUNDenseMatrix(LOTKA_VOLTERRA_N, LOTKA_VOLTERRA_N, CTX);

  SM_ELEMENT_D(mat, 0, 0) = ONE;
  SM_ELEMENT_D(mat, 0, 1) = ZERO;
  SM_ELEMENT_D(mat, 1, 0) = ZERO;
  SM_ELEMENT_D(mat, 1, 1) = ONE;

  return mat;
}

/* --------------------------------------------------------------------------
 * Pendulum
 * -------------------------------------------------------------------------- */

static SUNMatrix jac_pendulum_create(sunrealtype theta)
{
  SUNMatrix mat = SUNDenseMatrix(PENDULUM_N, PENDULUM_N, CTX);

  sunrealtype x           = cos(theta);
  sunrealtype y           = sin(theta);
  SM_ELEMENT_D(mat, 0, 0) = ONE;
  SM_ELEMENT_D(mat, 0, 2) = x;
  SM_ELEMENT_D(mat, 1, 1) = ONE;
  SM_ELEMENT_D(mat, 1, 2) = y;
  SM_ELEMENT_D(mat, 2, 0) = TWO * x;
  SM_ELEMENT_D(mat, 2, 1) = TWO * y;

  return mat;
}

/* --------------------------------------------------------------------------
 * Linear system
 * -------------------------------------------------------------------------- */

static SUNMatrix jac_linsys_create(void)
{
  SUNMatrix mat = SUNDenseMatrix(LINSYS_N, LINSYS_N, CTX);

  SM_ELEMENT_D(mat, 0, 0) = ONE;
  SM_ELEMENT_D(mat, 0, 1) = ONE;
  SM_ELEMENT_D(mat, 1, 0) = ONE;
  SM_ELEMENT_D(mat, 1, 1) = ONE;
  SM_ELEMENT_D(mat, 1, 2) = ONE;
  SM_ELEMENT_D(mat, 2, 2) = ONE;
  SM_ELEMENT_D(mat, 2, 3) = ONE;
  SM_ELEMENT_D(mat, 3, 0) = TWO;
  SM_ELEMENT_D(mat, 3, 1) = ONE;
  SM_ELEMENT_D(mat, 3, 2) = ONE;
  SM_ELEMENT_D(mat, 3, 3) = ONE;

  return mat;
}

/* --------------------------------------------------------------------------
 * Non-linear system
 * -------------------------------------------------------------------------- */

static SUNMatrix jac_nonlinsys_create(sunrealtype d2x1,
                                      sunrealtype dx1,
                                      sunrealtype d2x2,
                                      sunrealtype dx2,
                                      sunrealtype x3,
                                      sunrealtype d2x4,
                                      sunrealtype x4,
                                      sunrealtype dx5)
{
  SUNMatrix mat = SUNDenseMatrix(NONLINSYS_N, NONLINSYS_N, CTX);

  SM_ELEMENT_D(mat, 0, 0) = TWO * d2x1;
  SM_ELEMENT_D(mat, 2, 0) = TWO * dx1;

  SM_ELEMENT_D(mat, 1, 1) = -TWO * d2x2;
  SM_ELEMENT_D(mat, 4, 1) = -TWO * dx2;

  SM_ELEMENT_D(mat, 2, 2) = -TWO * x3;
  SM_ELEMENT_D(mat, 3, 2) = -TWO * x3;

  SM_ELEMENT_D(mat, 1, 3) = TWO * TWO * d2x4;
  SM_ELEMENT_D(mat, 3, 3) = TWO * TWO * x4;

  SM_ELEMENT_D(mat, 0, 4) = TWO * -TWO * dx5;
  SM_ELEMENT_D(mat, 4, 4) = TWO * TWO * dx5;

  return mat;
}
