#ifndef _DD_ERR_H
#define _DD_ERR_H

#include <sundials/priv/sundials_errors_impl.h>

/* ----------------------------------------------------------------------------
 * Error code definitions
 * ---------------------------------------------------------------------------*/

#define DD_ERR_CODE_LIST(ENTRY)                                       \
  ENTRY(DD_ERR_NULL_SUNCTX, "SUNCtx is NULL")                         \
  ENTRY(DD_ERR_DD_MEM_NULL, "DD memory is NULL")                      \
  ENTRY(DD_ERR_ARG_CORRUPT, "argument provided is NULL or corrupted") \
  ENTRY(DD_ERR_MEM_FAIL, "a memory request failed")                   \
  ENTRY(DD_ERR_MALLOC_FAIL, "malloc returned NULL")                   \
  ENTRY(DD_ERR_IDA_ERR, "an IDA error occured")                       \
  ENTRY(DD_ERR_OP_FAIL, "an operation failed")                        \
  ENTRY(DD_ERR_GENERIC, "an error occurred")                          \
  ENTRY(DD_ERR_UNKNOWN, "Unknown error occured")

#define DD_EXPAND_TO_ENUM(name, description) name,

/* clang-format off */
/** @brief DD error codes. The range of error codes is [-300,-200] to avoid
    conflicts with IDA return codes. */
enum
{
  DD_ERR_MINIMUM                                                = -300,
  DD_ERR_CODE_LIST(DD_EXPAND_TO_ENUM)
  DD_ERR_MAXIUMUM                                               = -200,
  DD_SUCCESS                                                    = 0
};

/* clang-format on */

/* ----------------------------------------------------------------------------
 * Error functions
 * ---------------------------------------------------------------------------*/

const char* DDGetErrMsg(int);

/* --------------------------------------------------------------------------
 * DD Error handling macros wrapping the Sundials error handlers
 * -------------------------------------------------------------------------- */

#define DDHandleErrWithMsg(msg, code, sunctx) \
  SUNHandleErrWithMsg(__LINE__, __func__, __FILE__, msg, code, sunctx)

#define DDHandleErrWithFmtMsg(msg, code, sunctx, ...)                     \
  SUNHandleErrWithFmtMsg(__LINE__, __func__, __FILE__, msg, code, sunctx, \
                         __VA_ARGS__)

#define DDHandleErr(code, sunctx) \
  DDHandleErrWithMsg(DDGetErrMsg(code), code, sunctx)

#define DDFunctionBegin(sunctx) \
  SUNContext SUNCTX_ = sunctx;  \
  (void)SUNCTX_

#define DDCheck(expr, code)                                       \
  do {                                                            \
    if (SUNHintFalse(!(expr)))                                    \
    {                                                             \
      DDHandleErrWithFmtMsg("expected %s", code, SUNCTX_, #expr); \
      return code;                                                \
    }                                                             \
  }                                                               \
  while (0)

#define DDCheckNull(expr, code)                                   \
  do {                                                            \
    if (SUNHintFalse(!(expr)))                                    \
    {                                                             \
      DDHandleErrWithFmtMsg("expected %s", code, SUNCTX_, #expr); \
      return NULL;                                                \
    }                                                             \
  }                                                               \
  while (0)

#define DDCheckCall(call)                          \
  do {                                             \
    int dd_chk_call_err_code_ = call;              \
    if (SUNHintFalse(dd_chk_call_err_code_ < 0))   \
    {                                              \
      DDHandleErr(dd_chk_call_err_code_, SUNCTX_); \
      return dd_chk_call_err_code_;                \
    }                                              \
  }                                                \
  while (0)

#endif
