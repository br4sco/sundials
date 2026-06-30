#include <stdbool.h>
#include <stdlib.h>

#include <sunmatrix/sunmatrix_dense.h>

#include "matrix.h"
#include "test.h"

int main(void)
{
  SUNContext ctx;
  SUNContext_Create(SUN_COMM_NULL, &ctx);

  SUNMatrix A = SUNDenseMatrix(3, 2, ctx);
  PIVMatrix B = PIVMatWrapDense(A);

  TEST_ASSERT(B != NULL);
  PIVMatDestroy(B);
  B = NULL;
  PIVMatDestroy(B);
  SUNMatDestroy(A);

  return EXIT_SUCCESS;
}
