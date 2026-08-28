#include <stdbool.h>
#include <stdlib.h>

#include <sunmatrix/sunmatrix_dense.h>

#include "dd_staged_pivot_matrix.h"
#include "test.h"

int main(void)
{
  SUNContext ctx;
  SUNContext_Create(SUN_COMM_NULL, &ctx);

  SUNMatrix A = SUNDenseMatrix(4, 4, ctx);
  for (sunindextype i = 0; i < SM_LDATA_D(A); ++i) { SM_DATA_D(A)[i] = i + 1; }
  sunindextype rows[]   = {1, 2, 3};
  sunindextype cols[]   = {1, 2};
  DDStagedPivotMatrix B = DDStagedPivotMatWrapDense(A);
  TEST_ASSERT(B != NULL);

  DDStagedPivotMatrix C = DDStagedPivotMatCloneSub(B, 3, rows, 2, cols);
  TEST_ASSERT(C != NULL);

  SUNMatrix D = DDStagedPivotMatGetSUNMat(C);
  TEST_ASSERT(D != NULL);
  TEST_ASSERT(SM_ROWS_D(D) == 3);
  TEST_ASSERT(SM_COLUMNS_D(D) == 2);
  TEST_ASSERT(SM_ELEMENT_D(D, 0, 0) == 6);
  TEST_ASSERT(SM_ELEMENT_D(D, 1, 0) == 7);
  TEST_ASSERT(SM_ELEMENT_D(D, 2, 0) == 8);
  TEST_ASSERT(SM_ELEMENT_D(D, 0, 1) == 10);
  TEST_ASSERT(SM_ELEMENT_D(D, 1, 1) == 11);
  TEST_ASSERT(SM_ELEMENT_D(D, 2, 1) == 12);
  DDStagedPivotMatDestroy(B);
  DDStagedPivotMatDestroy(C);
  SUNMatDestroy(A);

  return EXIT_SUCCESS;
}
