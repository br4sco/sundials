#ifndef _DD_ERR_H
#define _DD_ERR_H

#include <sundials/priv/sundials_errors_impl.h>

/* ----------------------------------------------------------------------------
 * Error code definitions
 * ---------------------------------------------------------------------------*/

#define DD_ERR_CODE_LIST(ENTRY)                  \
  ENTRY(DD_ERR_NULL_SUNCTX, "SUNCtx is NULL")    \
  ENTRY(DD_ERR_DD_MEM_NULL, "DD memory is NULL") \
  ENTRY(DD_ERR_IDA_ERR, "an IDA error occured")  \
  ENTRY(DD_ERR_GENERIC, "an error occured")

#define DD_EXPAND_TO_ENUM(name, description) name,

/* clang-format off */
/** @brief DD error codes. The range of error codes is [-300,-200] to avoid
    conflicts with IDA return codes and SUNDIALS errors. */
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

#define DDHandleErrWithMsgAndCtx(msg, code, sunctx) \
  SUNHandleErrWithMsg(__LINE__, __func__, __FILE__, msg, code, sunctx)

#define DDHandleErrWithFmtMsgAndCtx(msg, code, sunctx, ...)               \
  SUNHandleErrWithFmtMsg(__LINE__, __func__, __FILE__, msg, code, sunctx, \
                         __VA_ARGS__)

#define DDHandleErrWithCtx(code, sunctx) \
  DDHandleErrWithMsgAndCtx(DDGetErrMsg(code), code, sunctx)

#define DDHandleErrWithMsg(msg, code) \
  DDHandleErrWithMsgAndCtx(msg, code, SUNCTX_)

#define DDHandleErrWithFmtMsg(msg, code, ...) \
  DDHandleErrWithFmtMsgAndCtx(msg, code, SUNCTX_, __VA_ARGS__)

#define DDHandleErr(code) DDHandleErrWithCtx(code, SUNCTX_)

#if defined(SUNDIALS_ENABLE_ERROR_CHECKS)
#define DDAssertWithCtx(expr, code, sunctx)                               \
  do {                                                                    \
    if (SUNHintFalse(!(expr)))                                            \
    {                                                                     \
      SUNHandleErrWithFmtMsg(__LINE__, __func__, __FILE__, "expected %s", \
                             code, sunctx, #expr);                        \
      return code;                                                        \
    }                                                                     \
  }                                                                       \
  while (0)
#else
#define DDAssertWithCtx(expr, code, sunctx)
#endif

#endif
