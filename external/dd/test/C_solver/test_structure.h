#include <sundials/sundials_core.h>
#include <sunmatrix/sunmatrix_dense.h>

#define ZERO SUN_RCONST(0.0)
#define ONE  SUN_RCONST(1.0)
#define TWO  SUN_RCONST(2.0)

SUNContext CTX;
