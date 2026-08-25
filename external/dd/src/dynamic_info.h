#ifndef _DYNAMIC_INFO_H
#define _DYNAMIC_INFO_H

#include <sundials/sundials_types.h>

#include "macros.h"
#include "static_info.h"

/* ==========================================================================
 * DD State Memory
 * ========================================================================== */

DD_DEFINE_PAIR(sunindextype, sunindextype, sunindextype);

struct DDDAEStateRec
{
  SUNContext sunctx;
  DDStaticInfo si;
  Pair_sunindextype* diff_var_aliases;
  sunindextype* yy_diff_alias_row;
  sunindextype* yp_diff_alias_row;
};

typedef struct DDDAEStateRec* DDDAEState;

void DDDAEStateDestroy(DDDAEState*);

DDDAEState DDDAEStateCreate(SUNContext, DDStaticInfo);

DDDAEState DDDAEStateClone(DDDAEState);

void DDDAEStateCopy(DDDAEState, DDDAEState);

SUNErrCode DDDAEStateUpdate(DDDAEState, uint8_t*);

#endif
