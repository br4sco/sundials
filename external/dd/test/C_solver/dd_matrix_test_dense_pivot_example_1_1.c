#include <stdbool.h>
#include <stdlib.h>

#include <sunmatrix/sunmatrix_dense.h>

#include "matrix.h"
#include "test.h"

int main(void)
{
  SUNContext ctx;
  SUNContext_Create(SUN_COMM_NULL, &ctx);

  SUNMatrix A = SUNDenseMatrix(2, 3, ctx);
  TEST_ASSERT(A != NULL);
  DDMatrix B = DDMatWrapDense(A);
  TEST_ASSERT(B != NULL);
  DDMatrixWorkspace ws = DDMatCreateWS(B);
  TEST_ASSERT(ws != NULL);
  SM_ELEMENT_D(A, 0, 0)    = 1;
  SM_ELEMENT_D(A, 0, 1)    = 1;
  SM_ELEMENT_D(A, 1, 0)    = 1;
  SM_ELEMENT_D(A, 1, 1)    = 1;
  SM_ELEMENT_D(A, 1, 2)    = 1;
  sunindextype colpivots[] = {0, 0, 0};
  TEST_ASSERT(DDMatPivot(B, ws, 0.0, 3, colpivots) == SUN_SUCCESS);
  TEST_ASSERT(colpivots[0] == 0);
  TEST_ASSERT(colpivots[1] == 2);
  TEST_ASSERT(colpivots[2] == 1);
  TEST_ASSERT(SM_ELEMENT_D(A, 0, 0) == 1);
  TEST_ASSERT(SM_ELEMENT_D(A, 0, 1) == 1);
  TEST_ASSERT(SM_ELEMENT_D(A, 0, 2) == 0);
  TEST_ASSERT(SM_ELEMENT_D(A, 1, 0) == 0);
  TEST_ASSERT(SM_ELEMENT_D(A, 1, 1) == 0);
  TEST_ASSERT(SM_ELEMENT_D(A, 1, 2) == 1);
  DDMatWSDestroy(ws);
  DDMatDestroy(B);
  SUNMatDestroy(A);

  return EXIT_SUCCESS;
}
