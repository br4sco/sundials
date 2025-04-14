#ifndef _TEST_H
#define _TEST_H

#include <stdio.h>
#include <stdlib.h>

#define TEST_ASSERT(test)                                                   \
  if (!(test))                                                              \
  {                                                                         \
    fprintf(stderr, "%s:%d:%s: Test assertion failed.", __FILE__, __LINE__, \
            __func__);                                                      \
    return EXIT_FAILURE;                                                    \
  }

#endif
