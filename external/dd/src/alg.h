#ifndef _ALG_H
#define _ALG_H

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>

/* ==========================================================================
 * Lexiographical Combinations
 * ========================================================================== */

/*
 * @brief Computes the next combination of `n` over `k`, zero indexed and in
 * lexiographical order.
 *
 * @param[in] n The size of the underlying set, the set is assumed to be:
 *  0,1,..,(n-1).
 * @param[in] k The number of elements to take from the set (length of `comb`).
 *  Assumes `k <= n`.
 * @param[inout] `comb` The current combination as input and the next
 *  combination as output. Elements in `comb` are assumed unique, increasing,
 *  and less than `n`.
 * @return `true` if there was a next combination, otherwise `false`.
 */
static inline bool dd_next_lexiographical_combination(size_t n,
                                                      size_t k,
                                                      size_t comb[static k])
{
  /*
     * This code is adapted from
     * https://codereview.stackexchange.com/a/184616
     */
  assert(n >= k);

  /*
     * If the last value in the combination is less than the maximum value in
     * the set, we can just increment the last value in the combination and
     * then we are done.
     */

  if (comb[k - 1] < n - 1)
  {
    comb[k - 1]++;
    return true;
  }

  for (size_t i = k - 1; i; --i)
  {
    if (comb[i - 1] < comb[i] - 1)
    {
      comb[i - 1]++;

      for (size_t j = i; j < k; ++j) { comb[j] = comb[j - 1] + 1; }

      return true;
    }
  }

  return false;
}

#endif
