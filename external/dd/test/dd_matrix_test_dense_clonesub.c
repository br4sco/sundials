#include <stdbool.h>
#include <stdlib.h>

#include <sunmatrix/sunmatrix_dense.h>

#include "matrix.h"
#include "test.h"

int main(void)
{
  SUNContext ctx;
  SUNContext_Create(SUN_COMM_NULL, &ctx);

  SUNMatrix mat = SUNDenseMatrix(4, 4, ctx);
  for (sunindextype i = 0; i < SM_LDATA_D(mat); ++i)
  {
    SM_DATA_D(mat)[i] = i + 1;
  }
  sunindextype rows[]  = {1, 2, 3};
  sunindextype cols[]  = {1, 2};
  ExtSUNMatrix* extmat = ExtSUNMatWrapDense(mat);
  TEST_ASSERT(extmat != NULL);

  ExtSUNMatrix* newextmat = ExtSUNMatCloneSub(extmat, 3, rows, 2, cols);
  TEST_ASSERT(newextmat != NULL);

  SUNMatrix newmat = ExtSUNMatGetMat(newextmat);
  TEST_ASSERT(newmat != NULL);
  TEST_ASSERT(SM_ROWS_D(newmat) == 3);
  TEST_ASSERT(SM_COLUMNS_D(newmat) == 2);
  TEST_ASSERT(SM_ELEMENT_D(newmat, 0, 0) == 6);
  TEST_ASSERT(SM_ELEMENT_D(newmat, 1, 0) == 7);
  TEST_ASSERT(SM_ELEMENT_D(newmat, 2, 0) == 8);
  TEST_ASSERT(SM_ELEMENT_D(newmat, 0, 1) == 10);
  TEST_ASSERT(SM_ELEMENT_D(newmat, 1, 1) == 11);
  TEST_ASSERT(SM_ELEMENT_D(newmat, 2, 1) == 12);
  ExtSUNMatDestroy(extmat);
  ExtSUNMatDestroy(newextmat);
  SUNMatDestroy(mat);

  return EXIT_SUCCESS;
}
