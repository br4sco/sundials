#include <stdbool.h>
#include <stdlib.h>

#include <sunmatrix/sunmatrix_dense.h>

#include "matrix.h"
#include "test.h"

int main(void)
{
  SUNContext ctx;
  SUNContext_Create(SUN_COMM_NULL, &ctx);

  SUNMatrix mat        = SUNDenseMatrix(3, 2, ctx);
  ExtSUNMatrix* extmat = ExtSUNMatWrapDense(mat);

  TEST_ASSERT(extmat != NULL);
  ExtSUNMatDestroy(extmat);
  extmat = NULL;
  ExtSUNMatDestroy(extmat);
  SUNMatDestroy(mat);

  return EXIT_SUCCESS;
}
