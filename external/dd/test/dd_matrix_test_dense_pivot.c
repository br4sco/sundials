#include <stdbool.h>
#include <stdlib.h>

#include <sunmatrix/sunmatrix_dense.h>

#include "matrix.h"
#include "test.h"

int main(void)
{
  SUNContext ctx;
  SUNContext_Create(SUN_COMM_NULL, &ctx);

  SUNMatrix A = SUNDenseMatrix(1, 2, ctx);
  TEST_ASSERT(A != NULL);
  DDMatrix* B = DDMatWrapDense(A);
  TEST_ASSERT(B != NULL);
  DDMatrixWorkspace* ws = DDMatCreateWS(B);
  TEST_ASSERT(ws != NULL);
  SM_ELEMENT_D(A, 0, 0)    = 0;
  SM_ELEMENT_D(A, 0, 1)    = 2;
  sunindextype colpivots[] = {0, 0};
  TEST_ASSERT(DDMatPivot(B, ws, 0.0, 2, colpivots) == SUN_SUCCESS);
  TEST_ASSERT(colpivots[0] == 1);
  TEST_ASSERT(colpivots[1] == 0);
  TEST_ASSERT(SM_ELEMENT_D(A, 0, 0) == 0);
  TEST_ASSERT(SM_ELEMENT_D(A, 0, 1) == 2);
  DDMatWSDestroy(ws);
  DDMatDestroy(B);
  SUNMatDestroy(A);

  return EXIT_SUCCESS;
}
