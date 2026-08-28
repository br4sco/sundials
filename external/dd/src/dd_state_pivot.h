#ifndef _DD_STATE_PIVOT_H
#define _DD_STATE_PIVOT_H

#include <stdint.h>
#include <sundials/sundials_nvector.h>
#include <sundials/sundials_types.h>

/** @brief Updates a dummy derivative specification. */
typedef struct _generic_DDStatePivot* DDStatePivot;

/** @brief Pointer to a table of DDStatePivot operations. */
typedef struct _generic_DDStatePivot_Ops* DDStatePivot_Ops;

/** @brief Table of operations implemented by a concrete DDStatePivot. */
struct _generic_DDStatePivot_Ops
{
  /** @brief Proposes an updated spec: mutate `spec` in place and set
      `*spec_changed`. Returns 0 on success, negative on failure. */
  int (*update)(DDStatePivot self,
                sunrealtype t,
                N_Vector Y,
                void* user_data,
                uint8_t* spec,
                sunbooleantype* spec_changed);

  /** @brief Destroys `self`. May be NULL. */
  void (*destroy)(DDStatePivot self);
};

/** @brief Implementation-owned content and the operations table. */
struct _generic_DDStatePivot
{
  void* content;
  DDStatePivot_Ops ops;
};

/** @brief Allocates an empty DDStatePivot: the shell plus a zeroed ops table.
    A concrete implementation fills `content` and `ops->update`/`ops->destroy`. */
DDStatePivot DDSPNewEmpty(void);

/** @brief Frees the shell and ops table only; does not dispatch `ops->destroy`.
    No-op if `self` is NULL. */
void DDSPFreeEmpty(DDStatePivot self);

/** @brief Dispatches to `self->ops->update()`, passing `user_data` through. */
int DDSPUpdate(DDStatePivot self,
               sunrealtype t,
               N_Vector Y,
               void* user_data,
               uint8_t* spec,
               sunbooleantype* spec_changed);

/** @brief Dispatches to `self->ops->destroy()`. No-op if `self` is NULL. */
void DDSPDestroy(DDStatePivot self);

#endif
