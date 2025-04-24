#include <stdbool.h>
#include <stdlib.h>

#include <sunmatrix/sunmatrix_dense.h>

#include "matrix.h"
#include "test.h"

int main(void)
{
  SUNContext ctx;
  SUNContext_Create(SUN_COMM_NULL, &ctx);

  SUNMatrix A = SUNDenseMatrix(4, 4, ctx);
  DDMatrix* B = DDMatWrapDense(A);
  TEST_ASSERT(B != NULL);
  SUNMatrix C = SUNDenseMatrix(3, 2, ctx);
  DDMatrix* D = DDMatWrapDense(C);
  TEST_ASSERT(D != NULL);

  for (sunindextype i = 0; i < SM_LDATA_D(A); ++i) { SM_DATA_D(A)[i] = i + 1; }
  sunindextype rows[] = {1, 2, 3};
  sunindextype cols[] = {1, 2};
  TEST_ASSERT(DDCopySub(B, D, 3, rows, 2, cols) == SUN_SUCCESS);

  TEST_ASSERT(C != NULL);
  TEST_ASSERT(SM_ROWS_D(C) == 3);
  TEST_ASSERT(SM_COLUMNS_D(C) == 2);
  TEST_ASSERT(SM_ELEMENT_D(C, 0, 0) == 6);
  TEST_ASSERT(SM_ELEMENT_D(C, 1, 0) == 7);
  TEST_ASSERT(SM_ELEMENT_D(C, 2, 0) == 8);
  TEST_ASSERT(SM_ELEMENT_D(C, 0, 1) == 10);
  TEST_ASSERT(SM_ELEMENT_D(C, 1, 1) == 11);
  TEST_ASSERT(SM_ELEMENT_D(C, 2, 1) == 12);
  DDMatDestroy(B);
  DDMatDestroy(D);
  SUNMatDestroy(A);
  return EXIT_SUCCESS;
}
