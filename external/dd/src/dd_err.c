#include <sundials/sundials_errors.h>

#include "dd_err.h"

const char* DDGetErrMsg(int code)
{
#define DD_EXPAND_TO_CASES(name, description) \
  case name: return description;

  switch (code)
  {
    DD_ERR_CODE_LIST(DD_EXPAND_TO_CASES)
  default:
    if (SUN_ERR_MINIMUM < code && code < SUN_ERR_MAXIMUM)
    {
      return SUNGetErrMsg(code);
    }

    return "unknown error";
  }
}
