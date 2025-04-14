#include <stdbool.h>
#include <stdlib.h>

#include "darr.h"
#include "test.h"

DD_DEFINE_DYNARR(int, int)

int main(void)
{
  const size_t initial_capacity = 100;

  DynArr_int* arr = DynArrCreate_int(initial_capacity);
  TEST_ASSERT(arr != NULL);
  DynArrDestroy_int(arr);

  return EXIT_SUCCESS;
}
