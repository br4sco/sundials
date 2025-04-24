#include <stdbool.h>
#include <stdlib.h>
#include <sunmatrix/sunmatrix_dense.h>
#include <sunmatrix/sunmatrix_sparse.h>

#include "macros.h"
#include "matrix.h"
#include "models.h"
#include "sundials/sundials_matrix.h"
#include "sundials/sundials_types.h"
#include "test.h"

#define ZERO SUN_RCONST(0.0)
#define ONE  SUN_RCONST(1.0)
#define TWO  SUN_RCONST(2.0)

DD_DEFINE_OPTION(real, sunrealtype);

enum
{
  DENSE,
  SPARSE_CSR,
  SPARSE_CSC
} matrix_type;

Structure* st;
SUNMatrix A;
DDMatrix* DDA;
SUNContext sunctx;

static void setup(const sunrealtype* structure_csr,
                  const sunrealtype* structure_csc)
{
  SUNContext_Create(SUN_COMM_NULL, &sunctx);
  A = SUNDenseMatrix(st->st_N, st->st_N, sunctx);

  if (matrix_type == DENSE) { DDA = DDMatWrapDense(A); }
  else if ((matrix_type == SPARSE_CSR) || (matrix_type == SPARSE_CSC))
  {
    sunbooleantype is_csr = matrix_type == SPARSE_CSR;
    for (sunindextype i = 0; i < SM_LDATA_D(A); ++i)
    {
      SM_DATA_D(A)[i] = (is_csr ? structure_csr[i] : structure_csc[i]) * TWO;
    }

    SUNMatrix SA = SUNSparseFromDenseMatrix(A, ZERO, is_csr ? CSR_MAT : CSC_MAT);
    SUNMatDestroy(A);
    A   = SA;
    DDA = DDMatWrapSparse(A);
  }
  else { assert(0); /* Impossible */ }
}

static void cleanup(void)
{
  STDestroy(st);
  SUNMatDestroy(A);
  DDMatDestroy(DDA);
}

static Option_real Elem(SUNMatrix A, sunindextype i, sunindextype j)
{
  if (SUNMatGetID(A) == SUNMATRIX_DENSE)
  {
    if ((0 <= i) && (i < SM_ROWS_D(A)) && (0 <= j) && (j < SM_COLUMNS_D(A)))
    {
      return (Option_real){.val = SM_ELEMENT_D(A, i, j)};
    }
  }
  else if (SUNMatGetID(A) == SUNMATRIX_SPARSE)
  {
    if ((0 <= i) && (i < SM_ROWS_S(A)) && (0 <= j) && (j < SM_COLUMNS_S(A)))
    {
      sunbooleantype is_csr = SM_SPARSETYPE_S(A) == CSR_MAT;
      sunindextype k, ii = is_csr ? i : j, jj = is_csr ? j : i;

      for (k = SM_INDEXPTRS_S(A)[ii]; k < SM_INDEXPTRS_S(A)[ii + 1]; ++k)
      {
        if (SM_INDEXVALS_S(A)[k] == jj)
        {
          return (Option_real){.val = SM_DATA_S(A)[k]};
        }
      }

      return (Option_real){.val = ZERO};
    }
  }

  return (Option_real){.none = SUNTRUE};
}

int main(int argc, char* argv[])
{
  int test_config;

  TEST_ASSERT(argc > 1)
  char c = argv[1][0];
  TEST_ASSERT((c == 'd') || (c == 'r') || (c == 'c'));
  if (c == 'd') { matrix_type = DENSE; }
  else if (c == 'r') { matrix_type = SPARSE_CSR; }
  else if (c == 'c') { matrix_type = SPARSE_CSC; }
  else { TEST_ASSERT(SUNFALSE) /* Unreachable */ }

  test_config = argc - 2;

  TEST_ASSERT((test_config == 0) || (test_config == 1));

  sunindextype N, M, NNZ_spec;
  sunindextype* NZ_spec;
  uint8_t* spec;

  /* ------------------------------------------------------------------------
   * Lotka-Volterra
   * ------------------------------------------------------------------------ */

  if (test_config == 0)
  {
    st = STCreate(LOTKA_VOLTERRA_N, LOTKA_VOLTERRA_C, LOTKA_VOLTERRA_D, NULL,
                  NULL);
    TEST_ASSERT(st != NULL);

    setup(LOTKA_VOLTERRA_JAC_STRUCTURE_CSR, LOTKA_VOLTERRA_JAC_STRUCTURE_CSC);

    M        = st->st_M;
    N        = st->st_N;
    NNZ_spec = 2;
    NZ_spec  = (sunindextype[]){0, 1};
    spec     = (uint8_t[]){1, 1};

    /* SUNMatZero(A); */

    TEST_ASSERT(DDMatEvalDDJac(DDA, st, NNZ_spec, NZ_spec, spec, TWO) ==
                SUN_SUCCESS);

    for (sunindextype i = 0; i < M; ++i)
    {
      for (sunindextype j = 0; j < N; ++j)
      {
        Option_real Aij = Elem(A, i, j);
        TEST_ASSERT(Aij.none == SUNFALSE);
        TEST_ASSERT((Aij.val == ZERO) || (Aij.val == TWO));
      }
    }

    Option_real A0, A1, A2, A3;

#define LOTKA_VOLTERRA_JAC_ROW(row)                                 \
  A0 = Elem(A, row, 0), A1 = Elem(A, row, 1), A2 = Elem(A, row, 2), \
  A3 = Elem(A, row, 3);                                             \
  TEST_ASSERT(A0.none == SUNFALSE);                                 \
  TEST_ASSERT(A1.none == SUNFALSE);                                 \
  TEST_ASSERT(A2.none == SUNFALSE);                                 \
  TEST_ASSERT(A3.none == SUNFALSE)

    LOTKA_VOLTERRA_JAC_ROW(M);

    TEST_ASSERT(A0.val == -TWO);
    TEST_ASSERT(A1.val == ONE);
    TEST_ASSERT(A2.val == ZERO);
    TEST_ASSERT(A3.val == ZERO);

    LOTKA_VOLTERRA_JAC_ROW(M + 1);

    TEST_ASSERT(A0.val == ZERO);
    TEST_ASSERT(A1.val == ZERO);
    TEST_ASSERT(A2.val == -TWO);
    TEST_ASSERT(A3.val == ONE);

    cleanup();
  }

  /* ------------------------------------------------------------------------
   * Pendulum
   * ------------------------------------------------------------------------ */

  /*   if ((test_config == 0) || (test_config == 1)) */
  /*   { */
  /*     st = STCreate(PENDULUM_N, PENDULUM_C, PENDULUM_D, NULL, NULL); */
  /*     TEST_ASSERT(st != NULL); */

  /*     setup(PENDULUM_JAC_NNZ); */

  /*     M = st->st_M; */
  /*     N = st->st_N; */

  /*     NNZ_spec = 1; */
  /*     if (test_config == 0) /\* y', y'' as DDs *\/ */
  /*     { */
  /*       NZ_spec = (sunindextype[]){0}; */
  /*       spec    = (uint8_t[]){2, 0, 0}; */
  /*     } */
  /*     else if (test_config == 1) /\* x', x'' as DDs *\/ */
  /*     { */
  /*       NZ_spec = (sunindextype[]){1}; */
  /*       spec    = (uint8_t[]){0, 2, 0}; */
  /*     } */
  /*     else { TEST_ASSERT(SUNFALSE); /\* Unreachable *\/ } */

  /*     TEST_ASSERT(DDMatPivotStructure(DDA, st, NNZ_spec, NZ_spec, spec) == */
  /*                 SUN_SUCCESS); */

  /*     SUNMatZero(A); */

  /*     TEST_ASSERT(DDMatEvalDDJac(DDA, st, NNZ_spec, NZ_spec, spec, TWO) == */
  /*                 SUN_SUCCESS); */

  /*     for (sunindextype i = 0; i < M; ++i) */
  /*     { */
  /*       for (sunindextype j = 0; j < N; ++j) */
  /*       { */
  /*         Option_real Aij = Elem(A, i, j); */
  /*         TEST_ASSERT(Aij.none == SUNFALSE); */
  /*         TEST_ASSERT(Aij.val == ZERO); */
  /*       } */
  /*     } */

  /*     Option_real A0, A1, A2, A3, A4, A5, A6; */

  /* #define PENDULUM_JAC_ROW(row)                                       \ */
  /*   A0 = Elem(A, row, 0), A1 = Elem(A, row, 1), A2 = Elem(A, row, 2), \ */
  /*   A3 = Elem(A, row, 3), A4 = Elem(A, row, 4), A5 = Elem(A, row, 5), \ */
  /*   A6 = Elem(A, row, 6);                                             \ */
  /*   TEST_ASSERT(A0.none == SUNFALSE);                                 \ */
  /*   TEST_ASSERT(A1.none == SUNFALSE);                                 \ */
  /*   TEST_ASSERT(A2.none == SUNFALSE);                                 \ */
  /*   TEST_ASSERT(A3.none == SUNFALSE);                                 \ */
  /*   TEST_ASSERT(A4.none == SUNFALSE);                                 \ */
  /*   TEST_ASSERT(A5.none == SUNFALSE);                                 \ */
  /*   TEST_ASSERT(A6.none == SUNFALSE) */

  /*     if (test_config == 0) /\* y', y'' as DDs *\/ */
  /*     { */
  /*       PENDULUM_JAC_ROW(M); */

  /*       TEST_ASSERT(A0.val == -TWO); */
  /*       TEST_ASSERT(A1.val == ONE); */
  /*       TEST_ASSERT(A2.val == ZERO); */
  /*       TEST_ASSERT(A3.val == ZERO); */
  /*       TEST_ASSERT(A4.val == ZERO); */
  /*       TEST_ASSERT(A5.val == ZERO); */
  /*       TEST_ASSERT(A6.val == ZERO); */

  /*       PENDULUM_JAC_ROW(M + 1); */

  /*       TEST_ASSERT(A0.val == ZERO); */
  /*       TEST_ASSERT(A1.val == -TWO); */
  /*       TEST_ASSERT(A2.val == ONE); */
  /*       TEST_ASSERT(A3.val == ZERO); */
  /*       TEST_ASSERT(A4.val == ZERO); */
  /*       TEST_ASSERT(A5.val == ZERO); */
  /*       TEST_ASSERT(A6.val == ZERO); */
  /*     } */
  /*     else if (test_config == 1) /\* x', x'' as DDs *\/ */
  /*     { */
  /*       PENDULUM_JAC_ROW(M); */

  /*       TEST_ASSERT(A0.val == ZERO); */
  /*       TEST_ASSERT(A1.val == ZERO); */
  /*       TEST_ASSERT(A2.val == ZERO); */
  /*       TEST_ASSERT(A3.val == -TWO); */
  /*       TEST_ASSERT(A4.val == ONE); */
  /*       TEST_ASSERT(A5.val == ZERO); */
  /*       TEST_ASSERT(A6.val == ZERO); */

  /*       PENDULUM_JAC_ROW(M + 1); */

  /*       TEST_ASSERT(A0.val == ZERO); */
  /*       TEST_ASSERT(A1.val == ZERO); */
  /*       TEST_ASSERT(A2.val == ZERO); */
  /*       TEST_ASSERT(A3.val == ZERO); */
  /*       TEST_ASSERT(A4.val == -TWO); */
  /*       TEST_ASSERT(A5.val == ONE); */
  /*       TEST_ASSERT(A6.val == ZERO); */
  /*     } */
  /*     else { TEST_ASSERT(SUNFALSE); /\* Unreachable *\/ } */

  /*     cleanup(); */
  /*   } */

  return EXIT_SUCCESS;
}
