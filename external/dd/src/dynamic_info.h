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
  Pair_sunindextype* diff_var_aliases;
  sunindextype* yy_diff_alias_row;
  sunindextype* yp_diff_alias_row;
};

typedef struct DDDAEStateRec* DDDAEState;

void DDDAEStateDestroy(DDDAEState*);

DDDAEState DDDAEStateCreate(DDStaticInfo);

DDDAEState DDDAEStateClone(DDStaticInfo, DDDAEState);

void DDDAEStateCopy(DDStaticInfo, DDDAEState, DDDAEState);

SUNErrCode DDDAEStateUpdate(DDStaticInfo, uint8_t*, DDDAEState);

#endif
