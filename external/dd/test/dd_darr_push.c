#include <stdbool.h>
#include <stdlib.h>

#include "darr.h"
#include "test.h"

DD_DEFINE_DYNARR(int, int)

int main(void)
{
  TEST_ASSERT(DynArrLength_int(NULL) == 0);
  TEST_ASSERT(DynArrPushBack_int(NULL, 0) == false);

  const size_t initial_capacity = 1;

  DynArr_int arr = DynArrCreate_int(initial_capacity);
  TEST_ASSERT(arr != NULL);
  TEST_ASSERT(DynArrLength_int(arr) == 0);
  TEST_ASSERT(DynArrPushBack_int(arr, 0));
  TEST_ASSERT(DynArrLength_int(arr) == 1);
  TEST_ASSERT(DynArrPushBack_int(arr, 0));
  TEST_ASSERT(DynArrLength_int(arr) == 2);
  DynArrDestroy_int(arr);

  return EXIT_SUCCESS;
}
