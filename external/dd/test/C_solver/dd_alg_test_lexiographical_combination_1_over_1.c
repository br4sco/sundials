#include <stdbool.h>
#include <stdlib.h>

#include "alg.h"
#include "test.h"

int main(void)
{
  const size_t n = 1, k = 1;
  size_t comb[] = {0};
  TEST_ASSERT(dd_next_lexiographical_combination(n, k, comb) == false);

  return EXIT_SUCCESS;
}
