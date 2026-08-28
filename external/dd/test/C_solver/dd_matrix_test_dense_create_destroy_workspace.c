#include <stdbool.h>
#include <stdlib.h>

#include <sunmatrix/sunmatrix_dense.h>

#include "dd_staged_pivot_matrix.h"
#include "test.h"

int main(void)
{
  SUNContext ctx;
  SUNContext_Create(SUN_COMM_NULL, &ctx);

  SUNMatrix A           = SUNDenseMatrix(3, 2, ctx);
  DDStagedPivotMatrix B = DDStagedPivotMatWrapDense(A);
  TEST_ASSERT(B != NULL);

  DDStagedPivotMatrixWorkspace ws = DDStagedPivotMatCreateWS(B);
  TEST_ASSERT(ws != NULL);

  DDStagedPivotMatDestroy(B);
  B = NULL;
  DDStagedPivotMatWSDestroy(ws);
  ws = NULL;
  DDStagedPivotMatWSDestroy(ws);
  SUNMatDestroy(A);
  return EXIT_SUCCESS;
}
