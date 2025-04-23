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
  DDMatrix* B = DDMatWrapDense(A);
  TEST_ASSERT(B != NULL);

  DDMatrixWorkspace* ws = DDMatCreateWS(B);
  TEST_ASSERT(ws != NULL);

  DDMatDestroy(B);
  B = NULL;
  DDMatWSDestroy(ws);
  ws = NULL;
  DDMatWSDestroy(ws);
  SUNMatDestroy(A);
  return EXIT_SUCCESS;
}
