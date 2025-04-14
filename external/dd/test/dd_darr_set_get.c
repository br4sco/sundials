#include <stdbool.h>
#include <stdlib.h>

#include "darr.h"
#include "test.h"

DD_DEFINE_DYNARR(int, int)

int main(void)
{
  TEST_ASSERT(DynArrGet_int(NULL, 0).none);

  const size_t initial_capacity = 1;

  DynArr_int* arr = DynArrCreate_int(initial_capacity);
  TEST_ASSERT(DynArrGet_int(arr, 0).none);
  TEST_ASSERT(DynArrPushBack_int(arr, 0));
  TEST_ASSERT(DynArrPushBack_int(arr, 1));
  TEST_ASSERT(DynArrPushBack_int(arr, 2));
  Option_int val = DynArrGet_int(arr, 0);
  TEST_ASSERT(val.none == false);
  TEST_ASSERT(val.val == 0);
  val = DynArrGet_int(arr, 1);
  TEST_ASSERT(val.none == false);
  TEST_ASSERT(val.val == 1);
  val = DynArrGet_int(arr, 2);
  TEST_ASSERT(val.none == false);
  TEST_ASSERT(val.val == 2);
  TEST_ASSERT(DynArrGet_int(arr, 3).none);
  TEST_ASSERT(DynArrSet_int(arr, 3, 3) == false);
  TEST_ASSERT(DynArrSet_int(arr, 0, 2));
  TEST_ASSERT(DynArrSet_int(arr, 1, 1));
  TEST_ASSERT(DynArrSet_int(arr, 2, 0));
  val = DynArrGet_int(arr, 0);
  TEST_ASSERT(val.none == false);
  TEST_ASSERT(val.val == 2);
  val = DynArrGet_int(arr, 1);
  TEST_ASSERT(val.none == false);
  TEST_ASSERT(val.val == 1);
  val = DynArrGet_int(arr, 2);
  TEST_ASSERT(val.none == false);
  TEST_ASSERT(val.val == 0);
  TEST_ASSERT(DynArrGet_int(arr, 3).none);
  DynArrDestroy_int(arr);

  return EXIT_SUCCESS;
}
