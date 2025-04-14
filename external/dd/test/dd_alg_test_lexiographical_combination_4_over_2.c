#include <stdbool.h>
#include <stdlib.h>

#include "alg.h"
#include "test.h"

int main(void)
{
  const size_t n = 4, k = 2;
  size_t comb[] = {0, 1};
  TEST_ASSERT(dd_next_lexiographical_combination(n, k, comb));
  TEST_ASSERT(comb[0] == 0);
  TEST_ASSERT(comb[1] == 2);
  TEST_ASSERT(dd_next_lexiographical_combination(n, k, comb));
  TEST_ASSERT(comb[0] == 0);
  TEST_ASSERT(comb[1] == 3);
  TEST_ASSERT(dd_next_lexiographical_combination(n, k, comb));
  TEST_ASSERT(comb[0] == 1);
  TEST_ASSERT(comb[1] == 2);
  TEST_ASSERT(dd_next_lexiographical_combination(n, k, comb));
  TEST_ASSERT(comb[0] == 1);
  TEST_ASSERT(comb[1] == 3);
  TEST_ASSERT(dd_next_lexiographical_combination(n, k, comb));
  TEST_ASSERT(comb[0] == 2);
  TEST_ASSERT(comb[1] == 3);
  TEST_ASSERT(dd_next_lexiographical_combination(n, k, comb) == false);
}
