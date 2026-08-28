#include <stdlib.h>

#include "dd_state_pivot.h"

DDStatePivot DDSPNewEmpty(void)
{
  DDStatePivot self = malloc(sizeof(*self));
  if (self == NULL) { return NULL; }

  DDStatePivot_Ops ops = calloc(1, sizeof(*ops));
  if (ops == NULL)
  {
    free(self);
    return NULL;
  }

  self->content = NULL;
  self->ops     = ops;

  return self;
}

void DDSPFreeEmpty(DDStatePivot self)
{
  if (self == NULL) { return; }
  free(self->ops);
  free(self);
}

int DDSPUpdate(DDStatePivot self,
               sunrealtype t,
               N_Vector Y,
               void* user_data,
               uint8_t* spec,
               sunbooleantype* spec_changed)
{
  return self->ops->update(self, t, Y, user_data, spec, spec_changed);
}

void DDSPDestroy(DDStatePivot self)
{
  if (self == NULL) { return; }
  if (self->ops != NULL && self->ops->destroy != NULL)
  {
    self->ops->destroy(self);
  }
}
