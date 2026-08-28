#include <stdbool.h>
#include <stdlib.h>

#include <sunmatrix/sunmatrix_dense.h>

#include "dd_staged_pivot_matrix.h"
#include "test.h"

int main(void)
{
  SUNContext ctx;
  SUNContext_Create(SUN_COMM_NULL, &ctx);

  SUNMatrix A           = SUNDenseMatrix(4, 4, ctx);
  DDStagedPivotMatrix B = DDStagedPivotMatWrapDense(A);
  TEST_ASSERT(B != NULL);
  SUNMatrix C           = SUNDenseMatrix(3, 2, ctx);
  DDStagedPivotMatrix D = DDStagedPivotMatWrapDense(C);
  TEST_ASSERT(D != NULL);

  for (sunindextype i = 0; i < SM_LDATA_D(A); ++i) { SM_DATA_D(A)[i] = i + 1; }
  sunindextype rows[] = {1, 2, 3};
  sunindextype cols[] = {1, 2};
  TEST_ASSERT(DDStagedPivotMatCopySub(B, D, rows, cols) == SUN_SUCCESS);

  TEST_ASSERT(C != NULL);
  TEST_ASSERT(SM_ROWS_D(C) == 3);
  TEST_ASSERT(SM_COLUMNS_D(C) == 2);
  TEST_ASSERT(SM_ELEMENT_D(C, 0, 0) == 6);
  TEST_ASSERT(SM_ELEMENT_D(C, 1, 0) == 7);
  TEST_ASSERT(SM_ELEMENT_D(C, 2, 0) == 8);
  TEST_ASSERT(SM_ELEMENT_D(C, 0, 1) == 10);
  TEST_ASSERT(SM_ELEMENT_D(C, 1, 1) == 11);
  TEST_ASSERT(SM_ELEMENT_D(C, 2, 1) == 12);
  DDStagedPivotMatDestroy(B);
  DDStagedPivotMatDestroy(D);
  SUNMatDestroy(A);
  return EXIT_SUCCESS;
}
