#include <stdbool.h>
#include <stdlib.h>

#include <sunmatrix/sunmatrix_dense.h>

#include "matrix.h"
#include "test.h"

int main(void)
{
  SUNContext ctx;
  SUNContext_Create(SUN_COMM_NULL, &ctx);

  SUNMatrix mat = SUNDenseMatrix(1, 2, ctx);
  TEST_ASSERT(mat != NULL);
  ExtSUNMatrix* extmat = ExtSUNMatWrapDense(mat);
  TEST_ASSERT(extmat != NULL);
  ExtSUNMatrixWS* ws = ExtSUNMatCreateWS(extmat);
  TEST_ASSERT(ws != NULL);
  SM_ELEMENT_D(mat, 0, 0)  = 0;
  SM_ELEMENT_D(mat, 0, 1)  = 2;
  sunindextype colpivots[] = {0, 0};
  TEST_ASSERT(ExtSUNMatPivot(extmat, ws, 0.0, 2, colpivots));
  TEST_ASSERT(colpivots[0] == 1);
  TEST_ASSERT(colpivots[1] == 0);
  TEST_ASSERT(SM_ELEMENT_D(mat, 0, 0) == 0);
  TEST_ASSERT(SM_ELEMENT_D(mat, 0, 1) == 2);
  ExtSUNMatWSDestroy(ws);
  ExtSUNMatDestroy(extmat);
  SUNMatDestroy(mat);

  return EXIT_SUCCESS;
}
