#include "dd_err.h"

const char* DDGetErrMsg(int code)
{
#define DD_EXPAND_TO_CASES(name, description) \
  case name: return description; break;

  switch (code)
  {
    DD_ERR_CODE_LIST(DD_EXPAND_TO_CASES)
  default: return "unknown error";
  }
}
