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
    sunbooleantype is_error;         \
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
    sunbooleantype none;          \
    T val;                        \
  } Option_##NAME

/** @brief Defines a product of the types `U` and `V`. */
#define DD_DEFINE_PAIR(NAME, U, V) \
  typedef struct                   \
  {                                \
    U fst;                         \
    V snd;                         \
  } Pair_##NAME

/* ==========================================================================
 * Constants
 * ========================================================================== */

/** @brief Scales potential pivoting value before deciding if to pivot */
#define DD_PIVOT_SCALE SUN_RCONST(1.125)

#endif
