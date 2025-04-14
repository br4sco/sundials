#ifndef _DD_MACROS_H
#define _DD_MACROS_H

#include <stdbool.h>
#include <stddef.h>
#include <sundials/sundials_core.h>

/* ==========================================================================
 * Type Macros
 * ========================================================================== */

/** @brief Defines a result type with a result of type `T`, errors with type
 * `E`. */
#define DD_DEFINE_RESULT(NAME, T, E) \
  typedef struct                     \
  {                                  \
    bool is_error : 1;               \
    union                            \
    {                                \
      T result;                      \
      E error;                       \
    };                               \
  } Result_##NAME

/** @brief Defines an option type of type `T`. */
#define DD_DEFINE_OPTION(NAME, T) \
  typedef struct                  \
  {                               \
    bool none : 1;                \
    T val;                        \
  } Option_##NAME

/* ==========================================================================
 * Constants
 * ========================================================================== */

/** @brief Scales potential pivoting value before deciding if to pivot */
#define DD_PIVOT_SCALE SUN_RCONST(1.125)

#endif
