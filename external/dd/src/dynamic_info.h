#ifndef _DYNAMIC_INFO_H
#define _DYNAMIC_INFO_H

#include <sundials/sundials_types.h>

#include "pivot.h"
#include "structure.h"

/* ==========================================================================
 * DD State Memory
 * ========================================================================== */

struct DDstateMemRec
{
  Pair_sunindextype* diff_var_aliases;
  sunindextype* yy_diff_alias_row;
  sunindextype* yp_diff_alias_row;
};

typedef struct DDstateMemRec* DDstateMem;

void DDstateDestroy(DDstateMem*);

DDstateMem DDstateCreate(DAEStruct);

DDstateMem DDstateClone(DAEStruct, DDstateMem);

SUNErrCode DDstateUpdate(DAEStruct, uint8_t*, DDstateMem);

#endif
