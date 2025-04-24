#ifndef _TEST_H
#define _TEST_H

#include <stdio.h>
#include <stdlib.h>

#define TEST_ASSERT(expr)                                              \
  if (!(expr))                                                         \
  {                                                                    \
    fprintf(stderr, "%s:%d:%s: Test assertion (%s) failed.", __FILE__, \
            __LINE__, __func__, #expr);                                \
    return EXIT_FAILURE;                                               \
  }

#endif
