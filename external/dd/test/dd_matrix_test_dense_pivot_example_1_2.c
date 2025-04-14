#include <stdbool.h>
#include <stdlib.h>

#include <sunmatrix/sunmatrix_dense.h>

#include "matrix.h"
#include "test.h"

int main(void)
{
  SUNContext ctx;
  SUNContext_Create(SUN_COMM_NULL, &ctx);

  SUNMatrix mat = SUNDenseMatrix(3, 4, ctx);
  TEST_ASSERT(mat != NULL);
  ExtSUNMatrix* extmat = ExtSUNMatWrapDense(mat);
  TEST_ASSERT(extmat != NULL);
  ExtSUNMatrixWS* ws = ExtSUNMatCreateWS(extmat);
  TEST_ASSERT(ws != NULL);
  SM_ELEMENT_D(mat, 0, 0)  = 1;
  SM_ELEMENT_D(mat, 0, 1)  = 1;
  SM_ELEMENT_D(mat, 1, 0)  = 1;
  SM_ELEMENT_D(mat, 1, 1)  = 1;
  SM_ELEMENT_D(mat, 1, 2)  = 1;
  SM_ELEMENT_D(mat, 2, 2)  = 0;
  SM_ELEMENT_D(mat, 2, 3)  = 1;
  sunindextype colpivots[] = {0, 0, 0, 0};
  TEST_ASSERT(ExtSUNMatPivot(extmat, ws, 0.0, 4, colpivots));
  TEST_ASSERT(colpivots[0] == 0);
  TEST_ASSERT(colpivots[1] == 2);
  TEST_ASSERT(colpivots[2] == 3);
  TEST_ASSERT(colpivots[3] == 1);
  TEST_ASSERT(SM_ELEMENT_D(mat, 0, 0) == 1);
  TEST_ASSERT(SM_ELEMENT_D(mat, 0, 1) == 1);
  TEST_ASSERT(SM_ELEMENT_D(mat, 0, 2) == 0);
  TEST_ASSERT(SM_ELEMENT_D(mat, 0, 3) == 0);
  TEST_ASSERT(SM_ELEMENT_D(mat, 1, 0) == 0);
  TEST_ASSERT(SM_ELEMENT_D(mat, 1, 1) == 0);
  TEST_ASSERT(SM_ELEMENT_D(mat, 1, 2) == 1);
  TEST_ASSERT(SM_ELEMENT_D(mat, 1, 3) == 0);
  TEST_ASSERT(SM_ELEMENT_D(mat, 2, 0) == 0);
  TEST_ASSERT(SM_ELEMENT_D(mat, 2, 1) == 0);
  TEST_ASSERT(SM_ELEMENT_D(mat, 2, 2) == 0);
  TEST_ASSERT(SM_ELEMENT_D(mat, 2, 3) == 1);
  ExtSUNMatWSDestroy(ws);
  ExtSUNMatDestroy(extmat);
  SUNMatDestroy(mat);

  return EXIT_SUCCESS;
}
