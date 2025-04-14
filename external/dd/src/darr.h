#ifndef _DARR_H
#define _DARR_H

#include <stdbool.h>
#include "macros.h"

/* ==========================================================================
 * Dynamically growing array data structure
 * ========================================================================== */

/* Based on https://iafisher.com/blog/2020/06/type-safe-generics-in-c */

/** @brief Defines a dynamic array. */
#define DD_DEFINE_DYNARR(NAME, T)                                         \
  typedef struct                                                          \
  {                                                                       \
    size_t capacity;                                                      \
    size_t len;                                                           \
    T* data;                                                              \
  } DynArr_##NAME;                                                        \
                                                                          \
  DD_DEFINE_OPTION(NAME, T);                                              \
                                                                          \
  DynArr_##NAME* DynArrCreate_##NAME(size_t initial_capacity)             \
  {                                                                       \
    if (initial_capacity == 0) { return NULL; }                           \
                                                                          \
    DynArr_##NAME* arr = malloc(sizeof(*arr));                            \
    if (!arr) { return NULL; }                                            \
                                                                          \
    arr->data = malloc(initial_capacity * sizeof(T));                     \
    if (!arr->data)                                                       \
    {                                                                     \
      free(arr);                                                          \
      return NULL;                                                        \
    }                                                                     \
                                                                          \
    arr->capacity = initial_capacity;                                     \
    arr->len      = 0;                                                    \
    return arr;                                                           \
  }                                                                       \
                                                                          \
  size_t DynArrLength_##NAME(const DynArr_##NAME* arr)                    \
  {                                                                       \
    return arr ? arr->len : 0;                                            \
  }                                                                       \
                                                                          \
  Option_##NAME DynArrGet_##NAME(DynArr_##NAME* arr, size_t i)            \
  {                                                                       \
    if (!arr || i >= arr->len) { return (Option_##NAME){.none = true}; }  \
                                                                          \
    return (Option_##NAME){.val = arr->data[i]};                          \
  }                                                                       \
                                                                          \
  bool DynArrSet_##NAME(DynArr_##NAME* arr, size_t i, T val)              \
  {                                                                       \
    if (!arr || i >= arr->len) { return false; }                          \
                                                                          \
    arr->data[i] = val;                                                   \
    return true;                                                          \
  }                                                                       \
                                                                          \
  void DynArrDestroy_##NAME(DynArr_##NAME* arr)                           \
  {                                                                       \
    if (arr)                                                              \
    {                                                                     \
      if (arr->data) { free(arr->data); }                                 \
      arr->data = NULL;                                                   \
    }                                                                     \
    free(arr);                                                            \
  }                                                                       \
                                                                          \
  bool DynArrPushBack_##NAME(DynArr_##NAME* arr, T val)                   \
  {                                                                       \
    if (!arr) { return false; }                                           \
                                                                          \
    if (arr->len >= arr->capacity)                                        \
    {                                                                     \
      size_t new_capacity = arr->capacity * 2;                            \
      T* new_data         = realloc(arr->data, new_capacity * sizeof(T)); \
                                                                          \
      if (!new_data) { return false; }                                    \
                                                                          \
      arr->capacity = new_capacity;                                       \
      arr->data     = new_data;                                           \
    }                                                                     \
                                                                          \
    arr->data[arr->len] = val;                                            \
    arr->len += 1;                                                        \
    return true;                                                          \
  }                                                                       \
                                                                          \
  Option_##NAME DynArrPopBack_##NAME(DynArr_##NAME* arr)                  \
  {                                                                       \
    if (!arr || arr->len == 0) { return (Option_##NAME){.none = true}; }  \
                                                                          \
    arr->len -= 1;                                                        \
    return (Option_##NAME){.val = arr->data[arr->len]};                   \
  }

/** @brief A reference to the i'th element in a dynamic array. */
#define DA_Ith(arr, i) (arr->data[i])

/** @brief The length of a dynamic array. */
#define DA_LENGTH(arr) (arr ? arr->len : 0)

#endif
