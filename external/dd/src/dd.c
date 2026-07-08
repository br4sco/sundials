#include <idas/idas.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sundials/priv/sundials_errors_impl.h>
#include <sundials/sundials_core.h>
#include <sundials/sundials_macros.h>
#include <sundials/sundials_math.h>

#include "darr.h"
#include "dd.h"
#include "dd_err.h"
#include "dynamic_info.h"
#include "idas/idas_impl.h"
#include "idas/idas_ls.h"
#include "static_info.h"
#include "sundials/sundials_errors.h"
#include "sundials/sundials_matrix.h"
#include "sundials/sundials_nvector.h"
#include "sundials/sundials_types.h"
#include "sunmatrix/sunmatrix_dense.h"
#include "sunmatrix/sunmatrix_sparse.h"

/* ==========================================================================
 * Types and Macros
 * ========================================================================== */

#define INITIAL_DYN_ARR_CAPACITY 20
#define ZERO                     SUN_RCONST(0.0)
#define ONE                      SUN_RCONST(1.0)

/* --------------------------------------------------------------------------
 * DD Checkpoints
 * -------------------------------------------------------------------------- */

struct DDckpntMemRec
{
  sunrealtype ck_t0;
  sunrealtype ck_t1;
  sunrealtype ck_h0u;
  DDDAEState ck_state;
  IDAckpntMem ida_ck_mem;
  struct DDckpntMemRec* ck_next;
};

typedef struct DDckpntMemRec* DDckpntMem;

static void DDckpntDestroy(DDckpntMem* ck_mem_ptr)
{
  if (ck_mem_ptr == NULL || *ck_mem_ptr == NULL) { return; }

  DDckpntMem ck = *ck_mem_ptr;
  DDDAEStateDestroy(&ck->ck_state);

  /* NOTE(oerikss, 2025-04-15): We delegate deleting IDA checkpoints to
       `IDAAdjFree` in `DDAdjFree`. */

  free(ck);
  *ck_mem_ptr = NULL;
}

static DDckpntMem DDckpntCreate(DDStaticInfo si, sunrealtype t, DDDAEState state)
{
  DDckpntMem ck_mem = calloc(1, sizeof(*ck_mem));
  if (ck_mem == NULL) { return NULL; }

  ck_mem->ck_state = DDDAEStateClone(si, state);
  if (ck_mem->ck_state == NULL)
  {
    DDckpntDestroy(&ck_mem);
    return NULL;
  }

  ck_mem->ck_t0 = t;
  ck_mem->ck_t1 = t;

  return ck_mem;
}

/* --------------------------------------------------------------------------
 * Backwards Problems
 * -------------------------------------------------------------------------- */

typedef struct
{
  DDResFnB* db_resB;
  DDLsJacFnB* db_jacB;
  DDQuadRhsFnB* db_quadB;
  void* db_user_data;
} DataB;

typedef struct
{
  int pb_which;
  DataB* pb_data;
} ProbB;

static void ProbBDestroy(ProbB pb)
{
  if (pb.pb_data != NULL) { free(pb.pb_data); }
}

/* --------------------------------------------------------------------------
 * DD Session Memory
 * -------------------------------------------------------------------------- */

DD_DEFINE_DYNARR(ProbB, ProbB)

struct DDMemRec
{
  SUNContext sunctx;

  /* Static DAE Info */

  DDStaticInfo dd_si;

  /* Dynamic State */

  DDDAEState dd_state;
  uint8_t* dd_spec;

  /* IDA Memory */

  IDAMem ida_mem;

  /* Forward Problem */

  SUNMatrix dd_J;
  DDResFn* dd_res;
  DDLsJacFn1* dd_jacfn1;
  DDLsJacFn2* dd_jacfn2;
  sunrealtype dd_t0;
  N_Vector dd_yy;
  N_Vector dd_yp;
  N_Vector dd_id;
  void* dd_user_data;

  /* Forward Sensitivities */

  N_Vector* dd_yyS;
  N_Vector* dd_ypS;
  DDSensResFn* dd_resS;

  /* Backwards Problem */

  sunrealtype dd_tinitial;
  DynArr_ProbB dd_probBs;
  DDckpntMem ck_mem;
  DDckpntMem ck_mem_cur;
};

/* --------------------------------------------------------------------------
 * Private Function Prototypes
 * -------------------------------------------------------------------------- */

static int DDResWrapper(sunrealtype, N_Vector, N_Vector, N_Vector, void*);

static int DDLsJacFnWrapper1_Dense(sunrealtype,
                                   sunrealtype,
                                   N_Vector,
                                   N_Vector,
                                   N_Vector,
                                   SUNMatrix,
                                   void*,
                                   N_Vector,
                                   N_Vector,
                                   N_Vector);

static int DDLsJacFnWrapper1_CSR(sunrealtype,
                                 sunrealtype,
                                 N_Vector,
                                 N_Vector,
                                 N_Vector,
                                 SUNMatrix,
                                 void*,
                                 N_Vector,
                                 N_Vector,
                                 N_Vector);

static int DDLsJacFnWrapper2(sunrealtype,
                             sunrealtype,
                             N_Vector,
                             N_Vector,
                             N_Vector,
                             SUNMatrix,
                             void*,
                             N_Vector,
                             N_Vector,
                             N_Vector);

static int DDResSWrapper(int Ns,
                         sunrealtype,
                         N_Vector,
                         N_Vector,
                         N_Vector,
                         N_Vector[static Ns],
                         N_Vector[static Ns],
                         N_Vector[static Ns],
                         void*,
                         N_Vector,
                         N_Vector,
                         N_Vector);

static int DDResBWrapper(sunrealtype,
                         N_Vector,
                         N_Vector,
                         N_Vector,
                         N_Vector,
                         N_Vector,
                         void*);

static int DDLsJacFnBWrapper(sunrealtype,
                             sunrealtype,
                             N_Vector,
                             N_Vector,
                             N_Vector,
                             N_Vector,
                             N_Vector,
                             SUNMatrix,
                             void*,
                             N_Vector,
                             N_Vector,
                             N_Vector);

static int DDQuadRhsFnBWrapper(sunrealtype,
                               N_Vector,
                               N_Vector,
                               N_Vector,
                               N_Vector,
                               N_Vector,
                               void*);

static void DDSetYpFromY(DDStaticInfo, DDDAEState, N_Vector, N_Vector);
static void DDSetId(DDStaticInfo, DDDAEState, N_Vector);
static void DDAdjCleanup(DDMem dd_mem);
static void DDSensCleanup(DDMem dd_mem);

/* --------------------------------------------------------------------------
 * Debug helpers
 * -------------------------------------------------------------------------- */
/* static void debug_print_residuals(DD_Session s, sunrealtype t, FILE* file) */
/* { */
/*   N_Vector rr   = NULL; */
/*   N_Vector* rrS = NULL; */

/*   if (!s) */
/*   { */
/*     fprintf(file, "NULL session\n"); */
/*     return; */
/*   } */

/*   if (!s->resfn) */
/*   { */
/*     fprintf(file, "NULL residual\n"); */
/*     return; */
/*   } */

/*   rr = N_VClone(s->yy); */
/*   if (s->structure && s->structure->eqn_names) */
/*   { */
/*     fprintf(file, "=== Residual names ===\n"); */
/*     sunindextype ofs = 0; */
/*     for (sunindextype i = 0; i < s->structure->dae_size; ++i) */
/*     { */
/*       for (uint8_t l = 0; l <= s->structure->eqn_ofs[i]; ++l) */
/*       { */
/*         if (l == 0) */
/*         { */
/*           fprintf(file, "[%ld]%s ", ofs, s->structure->eqn_names[i]); */
/*         } */
/*         else */
/*         { */
/*           fprintf(file, "[%ld]d%d%s ", ofs, l, s->structure->eqn_names[i]); */
/*         } */
/*         ofs += 1; */
/*       } */
/*     } */
/*     fprintf(file, "\n\n"); */
/*   } */
/*   ida_resfn(t, s->yy, s->yp, rr, s); */
/*   fprintf(file, "=== DAE residual ===\n"); */
/*   N_VPrintFile(rr, file); */

/*   if (s->init_forward_sens) */
/*   { */
/*     if (!s->resfn) */
/*     { */
/*       fprintf(file, "NULL forward sensitivity residual\n"); */
/*       goto cleanup; */
/*     } */

/*     N_Vector* rrS = N_VCloneVectorArray(s->Ns, rr); */
/*     ida_resfnS(s->Ns, t, s->yy, s->yp, rr, s->yyS, s->ypS, rrS, s, NULL, NULL, */
/*                NULL); */
/*     fprintf(file, "=== DAE forward sensitivity residual ===\n"); */
/*     for (int i = 0; i < s->Ns; ++i) { N_VPrintFile(rrS[i], file); } */
/*   } */

/* cleanup: */
/*   if (rr) { N_VDestroy(rr); } */
/*   if (rrS) { N_VDestroyVectorArray(rrS, s->Ns); } */
/* } */

/* ==========================================================================
 * Interface implementation
 * ========================================================================== */

/* --------------------------------------------------------------------------
 * DDCreate
 * -------------------------------------------------------------------------- */

DDMem DDCreate(SUNContext sunctx)

{
  SUNFunctionBegin(sunctx);

  DDMem dd_mem = calloc(1, sizeof(*dd_mem));
  if (dd_mem == NULL)
  {
    DDHandleErr(SUN_ERR_MEM_FAIL);
    return NULL;
  }

  dd_mem->ida_mem = IDACreate(sunctx);
  if (dd_mem->ida_mem == NULL)
  {
    free(dd_mem);
    DDHandleErr(DD_ERR_IDA_ERR);
    return NULL;
  }

  dd_mem->sunctx = sunctx;

  return dd_mem;
}

/* --------------------------------------------------------------------------
 * DDFree
 * -------------------------------------------------------------------------- */

void DDFree(DDMem* dd_mem_ptr)
{
  if (dd_mem_ptr == NULL || *dd_mem_ptr == NULL) { return; }

  DDMem dd_mem = *dd_mem_ptr;

  DDSensCleanup(dd_mem);
  DDAdjCleanup(dd_mem);
  DDDAEStateDestroy(&dd_mem->dd_state);
  free(dd_mem->dd_spec);

  if (dd_mem->ida_mem != NULL)
  {
    IDAFree((void**)&dd_mem->ida_mem);
    dd_mem->ida_mem = NULL;
  }

  N_VDestroy(dd_mem->dd_yy);
  N_VDestroy(dd_mem->dd_yp);
  N_VDestroy(dd_mem->dd_id);

  free(dd_mem);

  *dd_mem_ptr = NULL;
}

/* --------------------------------------------------------------------------
 * DDInit
 * -------------------------------------------------------------------------- */

int DDInit(DDMem dd_mem,
           DDStaticInfo si,
           DDResFn res,
           uint8_t* spec,
           sunrealtype t0,
           N_Vector Y0)
{
  if (dd_mem == NULL)
  {
    DDHandleErrWithCtx(DD_ERR_DD_MEM_NULL, NULL);
    return DD_ERR_DD_MEM_NULL;
  }

  SUNContext sunctx = dd_mem->sunctx;

  SUNFunctionBegin(sunctx);

  if (si == NULL)
  {
    DDHandleErr(SUN_ERR_ARG_CORRUPT);
    return SUN_ERR_ARG_CORRUPT;
  }

  if (res == NULL)
  {
    DDHandleErr(SUN_ERR_ARG_CORRUPT);
    return SUN_ERR_ARG_CORRUPT;
  }

  if (spec == NULL)
  {
    DDHandleErr(SUN_ERR_ARG_CORRUPT);
    return SUN_ERR_ARG_CORRUPT;
  }

  if (Y0 == NULL)
  {
    DDHandleErr(SUN_ERR_ARG_CORRUPT);
    return SUN_ERR_ARG_CORRUPT;
  }

  dd_mem->dd_si = si;
  dd_mem->dd_t0 = t0;

  dd_mem->dd_spec = malloc(si->N * sizeof(*dd_mem->dd_spec));
  if (dd_mem->dd_spec == NULL)
  {
    DDHandleErr(SUN_ERR_MEM_FAIL);
    return SUN_ERR_MEM_FAIL;
  }
  memcpy(dd_mem->dd_spec, spec, si->N * sizeof(*dd_mem->dd_spec));

  DDDAEState state = DDDAEStateCreate(si);
  if (state == NULL)
  {
    DDHandleErr(SUN_ERR_MEM_FAIL);
    return SUN_ERR_MEM_FAIL;
  }
  dd_mem->dd_state = state;

  if (DDDAEStateUpdate(si, spec, state) < 0)
  {
    DDHandleErr(SUN_ERR_OP_FAIL);
    return SUN_ERR_OP_FAIL;
  }

  dd_mem->dd_res = res;

  dd_mem->dd_yy = N_VClone(Y0);
  dd_mem->dd_yp = N_VClone(Y0);
  if ((dd_mem->dd_yy == NULL) || (dd_mem->dd_yp == NULL))
  {
    DDHandleErr(SUN_ERR_MEM_FAIL);
    return SUN_ERR_MEM_FAIL;
  }
  DDSetYpFromY(si, state, Y0, dd_mem->dd_yp);

  dd_mem->dd_id = N_VClone(Y0);
  if (dd_mem->dd_id == NULL)
  {
    DDHandleErr(SUN_ERR_MEM_FAIL);
    return SUN_ERR_MEM_FAIL;
  }
  DDSetId(si, state, dd_mem->dd_id);

  if (IDAInit(dd_mem->ida_mem, DDResWrapper, t0, Y0, dd_mem->dd_yp) < 0)
  {
    DDHandleErr(DD_ERR_IDA_ERR);
    return DD_ERR_IDA_ERR;
  }

  if ((IDASetId(dd_mem->ida_mem, dd_mem->dd_id) < 0) ||
      (IDASetUserData(dd_mem->ida_mem, dd_mem) < 0))
  {
    DDHandleErr(DD_ERR_IDA_ERR);
    return DD_ERR_IDA_ERR;
  }

  return DD_SUCCESS;
}

static int DDResWrapper(sunrealtype t,
                        N_Vector yy,
                        N_Vector yp,
                        N_Vector rr,
                        void* user_data)
{
  const DDMem dd_mem     = (DDMem)user_data;
  const DDStaticInfo si  = dd_mem->dd_si;
  const DDDAEState state = dd_mem->dd_state;

  /* Evaluate differential equations residual. */

  const sunindextype N_diff = si->N_diff;
  const sunindextype rr_ofs = N_VGetLength(rr) - N_diff;

  const sunrealtype *yy_arr = N_VGetArrayPointer(yy),
                    *yp_arr = N_VGetArrayPointer(yp);

  sunrealtype* rr_arr = N_VGetArrayPointer(rr);
  for (sunindextype i = 0; i < N_diff; ++i)
  {
    const Pair_sunindextype p = state->diff_var_aliases[i];
    rr_arr[rr_ofs + i]        = yy_arr[p.snd] - yp_arr[p.fst];
  }

  /* Evaluate user supplied residual function. */

  int flag = dd_mem->dd_res(t, yy, rr, dd_mem->dd_user_data);

  return flag;
}

static void DDSetYpFromY(DDStaticInfo si, DDDAEState state, N_Vector yy, N_Vector yp)
{
  N_VConst(ZERO, yp);
  const sunrealtype* yy_arr = N_VGetArrayPointer(yy);
  sunrealtype* yp_arr       = N_VGetArrayPointer(yp);

  for (sunindextype i = 0; i < si->N_diff; ++i)
  {
    const Pair_sunindextype p = state->diff_var_aliases[i];
    yp_arr[p.fst]             = yy_arr[p.snd];
  }
}

static void DDSetId(DDStaticInfo si, DDDAEState state, N_Vector id)
{
  N_VConst(ZERO, id);
  sunrealtype* id_arr = N_VGetArrayPointer(id);
  for (sunindextype i = 0; i < si->N_diff; ++i)
  {
    const Pair_sunindextype p = state->diff_var_aliases[i];
    id_arr[p.fst]             = ONE;
  }
}

/* --------------------------------------------------------------------------
 * DDReInit
 * -------------------------------------------------------------------------- */

int DDReInit(DDMem dd_mem, sunrealtype t0, N_Vector Y0)
{
  if (dd_mem == NULL)
  {
    DDHandleErrWithCtx(DD_ERR_DD_MEM_NULL, NULL);
    return DD_ERR_DD_MEM_NULL;
  }

  SUNFunctionBegin(dd_mem->sunctx);

  DDSetYpFromY(dd_mem->dd_si, dd_mem->dd_state, Y0, dd_mem->dd_yp);

  if (IDAReInit(dd_mem->ida_mem, t0, Y0, dd_mem->dd_yp) < 0)
  {
    DDHandleErr(DD_ERR_IDA_ERR);
    return DD_ERR_IDA_ERR;
  }

  return DD_SUCCESS;
}

/* --------------------------------------------------------------------------
 * DDSolve
 * -------------------------------------------------------------------------- */

int DDSolve(DDMem dd_mem,
            sunrealtype tout,
            sunrealtype tret[static 1],
            N_Vector Y,
            int itask)
{
  if (dd_mem == NULL)
  {
    DDHandleErrWithCtx(DD_ERR_DD_MEM_NULL, NULL);
    return DD_ERR_DD_MEM_NULL;
  }

  SUNFunctionBegin(dd_mem->sunctx);

  if (Y == NULL)
  {
    DDHandleErr(SUN_ERR_ARG_CORRUPT);
    return SUN_ERR_ARG_CORRUPT;
  }

  int flag = IDASolve(dd_mem->ida_mem, tout, tret, Y, dd_mem->dd_yp, itask);
  if (flag < 0)
  {
    DDHandleErr(DD_ERR_IDA_ERR);
    return DD_ERR_IDA_ERR;
  }

  return flag;
}

/* --------------------------------------------------------------------------
 * DDSetSpec
 * -------------------------------------------------------------------------- */

int DDSetSpec(DDMem dd_mem, uint8_t* spec)
{
  if (dd_mem == NULL)
  {
    DDHandleErrWithCtx(DD_ERR_DD_MEM_NULL, NULL);
    return DD_ERR_DD_MEM_NULL;
  }

  SUNFunctionBegin(dd_mem->sunctx);

  DDStaticInfo si = dd_mem->dd_si;

  if (memcmp(spec, dd_mem->dd_spec, si->N * sizeof(*spec)) == 0)
  {
    return SUN_SUCCESS;
  }
  memcpy(dd_mem->dd_spec, spec, si->N * sizeof(*dd_mem->dd_spec));

  DDDAEState state = dd_mem->dd_state;

  if (DDDAEStateUpdate(si, spec, state) < 0)
  {
    DDHandleErr(SUN_ERR_OP_FAIL);
    return SUN_ERR_OP_FAIL;
  }

  IDAMem ida_mem = dd_mem->ida_mem;

  N_Vector yy = dd_mem->dd_yy, yp = dd_mem->dd_yp, id = dd_mem->dd_id;
  const sunrealtype tn = ida_mem->ida_tn;

  if (IDAGetDky(ida_mem, tn, 0, yy) < 0)
  {
    DDHandleErr(DD_ERR_IDA_ERR);
    return DD_ERR_IDA_ERR;
  }

  DDSetId(si, state, id);

  if (IDASetId(ida_mem, id) < 0)
  {
    DDHandleErr(DD_ERR_IDA_ERR);
    return DD_ERR_IDA_ERR;
  }

  DDSetYpFromY(si, state, yy, yp);

  if (IDAReInit(ida_mem, tn, yy, yp) < 0)
  {
    DDHandleErr(DD_ERR_IDA_ERR);
    return DD_ERR_IDA_ERR;
  }

  if (ida_mem->ida_sensi)
  {
    N_Vector *yyS = dd_mem->dd_yyS, *ypS = dd_mem->dd_ypS;

    if (IDAGetSensDky(ida_mem, tn, 0, yyS) < 0)
    {
      DDHandleErr(DD_ERR_IDA_ERR);
      return DD_ERR_IDA_ERR;
    }

    for (int i = 0; i < ida_mem->ida_Ns; ++i)
    {
      DDSetYpFromY(si, state, yyS[i], ypS[i]);
    }

    if (IDASensReInit(ida_mem, ida_mem->ida_ism, yyS, ypS) < 0)
    {
      DDHandleErr(DD_ERR_IDA_ERR);
      return DD_ERR_IDA_ERR;
    }
  }

  if (ida_mem->ida_adj)
  {
    IDAadjMem ida_adj_mem = ida_mem->ida_adj_mem;

    if (ida_adj_mem->ck_mem == NULL)
    {
      /* No IDA checkpoints were accumulated since the last IDAAdjReInit, so
         DDSetSpec is being called again before any DDSolveF. Update the
         current DD checkpoint's state in-place rather than prepending a new
         empty checkpoint. */
      if (DDDAEStateUpdate(si, spec, dd_mem->ck_mem->ck_state) < 0)
      {
        DDHandleErr(SUN_ERR_OP_FAIL);
        return SUN_ERR_OP_FAIL;
      }
    }
    else
    {
      /* Normal case: IDA checkpoints were accumulated since the last pivot.
         Store them in the current DD checkpoint and prepend a fresh one.

         NOTE(oerikss, 2025-04-11): We need to set the internal checkpoint
         field to NULL to prevent `IDAAdjReInit` from deleting all accumulated
         checkpoints. We restore these as we integrate the backwards problem
         over the DD checkpoints. We later take care of deleting the
         checkpoints in `DDAdjFree`. */

      DDckpntMem ck_next  = dd_mem->ck_mem;
      ck_next->ck_t1      = tn;
      ck_next->ida_ck_mem = ida_adj_mem->ck_mem;

      DDckpntMem ck_mem = DDckpntCreate(si, tn, state);
      if (ck_mem == NULL)
      {
        DDHandleErr(SUN_ERR_MEM_FAIL);
        return DD_ERR_IDA_ERR;
      }

      ck_mem->ck_next = ck_next;
      dd_mem->ck_mem  = ck_mem;

      ida_adj_mem->ck_mem = NULL;
    }

    if (IDAAdjReInit(ida_mem) < 0)
    {
      DDHandleErr(DD_ERR_IDA_ERR);
      return DD_ERR_IDA_ERR;
    }
  }

  return SUN_SUCCESS;
}

/* --------------------------------------------------------------------------
 * DDSensFree
 * -------------------------------------------------------------------------- */

static void DDSensCleanup(DDMem dd_mem)
{
  IDAMem ida_mem = dd_mem->ida_mem;

  if (dd_mem->dd_yyS != NULL)
  {
    for (int i = 0; i < ida_mem->ida_Ns; ++i)
    {
      N_VDestroy(dd_mem->dd_yyS[i]);
      dd_mem->dd_yyS[i] = NULL;
    }

    dd_mem->dd_yyS = NULL;
  }

  if (dd_mem->dd_ypS != NULL)
  {
    for (int i = 0; i < ida_mem->ida_Ns; ++i)
    {
      N_VDestroy(dd_mem->dd_ypS[i]);
      dd_mem->dd_ypS[i] = NULL;
    }

    dd_mem->dd_ypS = NULL;
  }
}

void DDSensFree(DDMem dd_mem)
{
  if (dd_mem == NULL) { return; }

  DDSensCleanup(dd_mem);
  IDASensFree(dd_mem->ida_mem);
}

/* --------------------------------------------------------------------------
 * DDSensInit
 * -------------------------------------------------------------------------- */

int DDSensInit(DDMem dd_mem,
               int Ns,
               int ism,
               DDSensResFn resfnS,
               N_Vector YS0[static Ns])
{
  if (dd_mem == NULL)
  {
    DDHandleErrWithCtx(DD_ERR_DD_MEM_NULL, NULL);
    return DD_ERR_GENERIC;
  }

  SUNFunctionBegin(dd_mem->sunctx);

  if (Ns < 1)
  {
    DDHandleErr(SUN_ERR_ARG_OUTOFRANGE);
    return SUN_ERR_ARG_OUTOFRANGE;
  }

  if ((ism != IDA_STAGGERED) && (ism != IDA_SIMULTANEOUS))
  {
    DDHandleErr(SUN_ERR_ARG_OUTOFRANGE);
    return SUN_ERR_ARG_OUTOFRANGE;
  }

  N_Vector* yS = N_VCloneVectorArray(Ns, YS0[0]);
  if (yS == NULL)
  {
    DDHandleErr(SUN_ERR_MEM_FAIL);
    return SUN_ERR_MEM_FAIL;
  }
  dd_mem->dd_yyS = yS;

  N_Vector* ypS = N_VCloneVectorArray(Ns, YS0[0]);
  if (ypS == NULL)
  {
    DDHandleErr(SUN_ERR_MEM_FAIL);
    return SUN_ERR_MEM_FAIL;
  }
  dd_mem->dd_ypS = ypS;

  for (int i = 0; i < Ns; ++i)
  {
    DDSetYpFromY(dd_mem->dd_si, dd_mem->dd_state, YS0[i], ypS[i]);
  }

  if (IDASensInit(dd_mem->ida_mem, Ns, ism, resfnS ? DDResSWrapper : NULL, YS0, ypS) <
      0)
  {
    DDHandleErr(DD_ERR_IDA_ERR);
    return DD_ERR_IDA_ERR;
  }

  dd_mem->dd_resS = resfnS;

  return DD_SUCCESS;
}

static int DDResSWrapper(int Ns,
                         sunrealtype t,
                         N_Vector yy,
                         SUNDIALS_MAYBE_UNUSED N_Vector yp,
                         N_Vector rr,
                         N_Vector yyS[static Ns],
                         N_Vector ypS[static Ns],
                         N_Vector rrS[static Ns],
                         void* user_data,
                         N_Vector tmp1,
                         N_Vector tmp2,
                         N_Vector tmp3)
{
  const DDMem dd_mem     = (DDMem)user_data;
  const DDStaticInfo si  = dd_mem->dd_si;
  const DDDAEState state = dd_mem->dd_state;

  /* Compute sensitivites of differential residuals. */

  const sunindextype N_diff = si->N_diff;
  const sunindextype rr_ofs = N_VGetLength(rrS[0]) - N_diff;
  for (int i = 0; i < Ns; ++i)
  {
    const sunrealtype *yyS_arr = N_VGetArrayPointer(yyS[i]),
                      *ypS_arr = N_VGetArrayPointer(ypS[i]);

    sunrealtype* rrS_arr = N_VGetArrayPointer(rrS[i]);
    for (sunindextype j = 0; j < N_diff; ++j)
    {
      const Pair_sunindextype p = state->diff_var_aliases[j];
      rrS_arr[rr_ofs + j]       = yyS_arr[p.snd] - ypS_arr[p.fst];
    }
  }

  /* Evaluate user supplied sensitivity residual function. */

  const int flag =
    dd_mem->dd_resS(Ns, t, yy, rr, yyS, rrS, dd_mem->dd_user_data, tmp1, tmp2, tmp3);

  return flag;
}

/* --------------------------------------------------------------------------
 * DDSensReInit
 * -------------------------------------------------------------------------- */

int DDSensReInit(DDMem dd_mem, int ism, N_Vector* YS0)
{
  if (dd_mem == NULL)
  {
    DDHandleErrWithCtx(DD_ERR_DD_MEM_NULL, NULL);
    return DD_ERR_GENERIC;
  }

  SUNFunctionBegin(dd_mem->sunctx);

  if (YS0 == NULL)
  {
    DDHandleErr(SUN_ERR_ARG_CORRUPT);
    return SUN_ERR_ARG_CORRUPT;
  }

  IDAMem ida_mem = dd_mem->ida_mem;
  N_Vector* ypS  = dd_mem->dd_ypS;

  for (int i = 0; i < ida_mem->ida_Ns; ++i)
  {
    DDSetYpFromY(dd_mem->dd_si, dd_mem->dd_state, YS0[i], ypS[i]);
  }

  if (IDASensReInit(ida_mem, ism, YS0, ypS) < 0)
  {
    DDHandleErr(DD_ERR_IDA_ERR);
    return DD_ERR_IDA_ERR;
  }

  return DD_SUCCESS;
}

/* --------------------------------------------------------------------------
 * DDSolveF
 * -------------------------------------------------------------------------- */

int DDSolveF(DDMem dd_mem,
             sunrealtype tout,
             sunrealtype tret[static 1],
             N_Vector Y,
             int itask,
             int ncheck[static 1])
{
  if (dd_mem == NULL)
  {
    DDHandleErrWithCtx(DD_ERR_DD_MEM_NULL, NULL);
    return DD_ERR_GENERIC;
  }

  SUNFunctionBegin(dd_mem->sunctx);

  if (Y == NULL)
  {
    DDHandleErr(SUN_ERR_ARG_CORRUPT);
    return SUN_ERR_ARG_CORRUPT;
  }

  IDAMem ida_mem = dd_mem->ida_mem;
  N_Vector yp    = dd_mem->dd_yp;

  int flag = IDASolveF(ida_mem, tout, tret, Y, yp, itask, ncheck);
  if (flag < 0)
  {
    DDHandleErr(DD_ERR_IDA_ERR);
    return DD_ERR_IDA_ERR;
  }

  /* Update the current time for the current checkpoint. */

  dd_mem->ck_mem->ck_t1 = ida_mem->ida_tn;

  /* Update the initial time step for this checkpoint here beacuse it is set
     after the first call to `IDASolveF` and because it will be overwritten when
     we call `IDAReInit` after a pivot. */

  dd_mem->ck_mem->ck_h0u = ida_mem->ida_h0u;

  return flag;
}

/* --------------------------------------------------------------------------
 * DDCalcIC
 * -------------------------------------------------------------------------- */

int DDCalcIC(DDMem dd_mem, int iocopt, sunrealtype tout1)
{
  if (dd_mem == NULL)
  {
    DDHandleErrWithCtx(DD_ERR_DD_MEM_NULL, NULL);
    return DD_ERR_GENERIC;
  }

  SUNFunctionBegin(dd_mem->sunctx);

  if (IDACalcIC(dd_mem->ida_mem, iocopt, tout1) < 0)
  {
    DDHandleErr(DD_ERR_IDA_ERR);
    return DD_ERR_IDA_ERR;
  }

  return DD_SUCCESS;
}

/* --------------------------------------------------------------------------
 * DDAdjInit
 * -------------------------------------------------------------------------- */

int DDAdjInit(DDMem dd_mem, long Nd, int interpType)
{
  if (dd_mem == NULL)
  {
    DDHandleErrWithCtx(DD_ERR_DD_MEM_NULL, NULL);
    return DD_ERR_GENERIC;
  }

  SUNFunctionBegin(dd_mem->sunctx);

  IDAMem ida_mem = dd_mem->ida_mem;

  if (IDAAdjInit(ida_mem, Nd, interpType) < 0)
  {
    DDHandleErr(DD_ERR_IDA_ERR);
    return DD_ERR_IDA_ERR;
  }

  dd_mem->dd_probBs = DynArrCreate_ProbB(INITIAL_DYN_ARR_CAPACITY);
  if (dd_mem->dd_probBs == NULL)
  {
    DDHandleErr(SUN_ERR_MEM_FAIL);
    return SUN_ERR_MEM_FAIL;
  }

  dd_mem->ck_mem = DDckpntCreate(dd_mem->dd_si, dd_mem->dd_t0, dd_mem->dd_state);
  if (dd_mem->ck_mem == NULL)
  {
    DDHandleErr(SUN_ERR_MEM_FAIL);
    return SUN_ERR_MEM_FAIL;
  }

  dd_mem->dd_tinitial = ida_mem->ida_tn;

  return DD_SUCCESS;
}

/* --------------------------------------------------------------------------
 * DDAdjFree
 * -------------------------------------------------------------------------- */

static void DDAdjCleanup(DDMem dd_mem)
{
  if (dd_mem->dd_probBs != NULL)
  {
    for (size_t i = 0; i < DA_LENGTH(dd_mem->dd_probBs); ++i)
    {
      ProbBDestroy(DA_Ith(dd_mem->dd_probBs, i));
    }
    DynArrDestroy_ProbB(dd_mem->dd_probBs);
  }
  dd_mem->dd_probBs = NULL;

  /* Collect IDA checkpoints from DD checkpoints and attach them to the IDA
     Adjoint memory before calling to make sure that they are all free'ed when
     calling `IDAAdjFree`. */

  IDAMem ida_mem = dd_mem->ida_mem;

  if (ida_mem->ida_adj)
  {
    /* Link IDA checkpoints in each DD checkpoint. */

    for (DDckpntMem ck = dd_mem->ck_mem; ck->ck_next != NULL; ck = ck->ck_next)
    {
      IDAckpntMem ida_ck = ck->ida_ck_mem;

      /* ida_ck is set to non-NULL only on a pivot so the last DD checkpoint
         will have NULL in this field. */

      if (ida_ck != NULL)
      {
        while (ida_ck->ck_next != NULL) { ida_ck = ida_ck->ck_next; }
        ida_ck->ck_next = ck->ck_next->ida_ck_mem;
      }
    }

    /* Attach these IDA checkpoints to the end of the checkpoints in the IDA
       adjoint memory. */

    DDckpntMem ck_mem     = dd_mem->ck_mem;
    IDAadjMem ida_adj_mem = ida_mem->ida_adj_mem;

    if (ida_adj_mem->ck_mem == NULL)
    {
      ida_adj_mem->ck_mem = ck_mem->ida_ck_mem;
    }
    else
    {
      IDAckpntMem ck = ida_adj_mem->ck_mem;

      while (ck->ck_next != NULL) { ck = ck->ck_next; }

      ck->ck_next = ck_mem->ida_ck_mem;
    }
  }

  DDckpntMem ck_mem = dd_mem->ck_mem;
  while (ck_mem != NULL)
  {
    DDckpntMem ck_mem_next = ck_mem->ck_next;
    DDckpntDestroy(&ck_mem);
    ck_mem = ck_mem_next;
  }

  dd_mem->ck_mem     = NULL;
  dd_mem->ck_mem_cur = NULL;
}

void DDAdjFree(DDMem dd_mem)
{
  if (dd_mem == NULL) { return; }

  DDAdjCleanup(dd_mem);
  IDAAdjFree(dd_mem->ida_mem);
}

/* --------------------------------------------------------------------------
 * DDCreateB
 * -------------------------------------------------------------------------- */

int DDCreateB(DDMem dd_mem, int which[static 1])
{
  if (dd_mem == NULL)
  {
    DDHandleErrWithCtx(DD_ERR_DD_MEM_NULL, NULL);
    return DD_ERR_GENERIC;
  }

  SUNFunctionBegin(dd_mem->sunctx);

  if (IDACreateB(dd_mem->ida_mem, which) < 0)
  {
    DDHandleErr(DD_ERR_IDA_ERR);
    return DD_ERR_IDA_ERR;
  }

  return DD_SUCCESS;
}

/* --------------------------------------------------------------------------
 * DDInitB
 * -------------------------------------------------------------------------- */

int DDInitB(DDMem dd_mem,
            int which,
            DDResFnB resB,
            sunrealtype tB0,
            N_Vector yyB0,
            N_Vector ypB0)
{
  if (dd_mem == NULL)
  {
    DDHandleErrWithCtx(DD_ERR_DD_MEM_NULL, NULL);
    return DD_ERR_GENERIC;
  }

  SUNFunctionBegin(dd_mem->sunctx);

  if (resB == NULL)
  {
    DDHandleErr(SUN_ERR_ARG_CORRUPT);
    return SUN_ERR_ARG_CORRUPT;
  }

  IDAMem ida_mem = dd_mem->ida_mem;

  sunrealtype ti = dd_mem->dd_tinitial, tn = ida_mem->ida_tn;

  /* Direction of the forward problem */
  int sign = (tn - ti > ZERO) ? 1 : -1;

  if ((sign * (tB0 - ti) < ZERO) || (sign * (tn - tB0) < ZERO))
  {
    DDHandleErr(SUN_ERR_ARG_OUTOFRANGE);
    return SUN_ERR_ARG_OUTOFRANGE;
  }

  if (yyB0 == NULL)
  {
    DDHandleErr(SUN_ERR_ARG_CORRUPT);
    return SUN_ERR_ARG_CORRUPT;
  }

  if (ypB0 == NULL)
  {
    DDHandleErr(SUN_ERR_ARG_CORRUPT);
    return SUN_ERR_ARG_CORRUPT;
  }

  ProbB pb = {.pb_which = which};

  pb.pb_data = calloc(1, sizeof(*pb.pb_data));
  if (pb.pb_data == NULL)
  {
    DDHandleErr(SUN_ERR_MEM_FAIL);
    return SUN_ERR_MEM_FAIL;
  }

  pb.pb_data->db_resB = resB;

  if (IDAInitB(ida_mem, which, DDResBWrapper, tB0, yyB0, ypB0) < 0)
  {
    ProbBDestroy(pb);
    DDHandleErr(DD_ERR_IDA_ERR);
    return DD_ERR_IDA_ERR;
  }

  if (IDASetUserDataB(ida_mem, which, pb.pb_data) < 0)
  {
    ProbBDestroy(pb);
    DDHandleErr(DD_ERR_IDA_ERR);
    return DD_ERR_IDA_ERR;
  }

  if (!DynArrPushBack_ProbB(dd_mem->dd_probBs, pb))
  {
    ProbBDestroy(pb);
    DDHandleErr(SUN_ERR_OP_FAIL);
    return SUN_ERR_OP_FAIL;
  }

  /* We re-init the IDA adjoint problem at each pivot, which changes
     `ia_tinitial`, so we need to re-set the true value for tinitial (which is
     the time when we called `DDAdjInit`). */
  ida_mem->ida_adj_mem->ia_tinitial = ti;

  dd_mem->ck_mem_cur = dd_mem->ck_mem;

  return DD_SUCCESS;
}

static int DDResBWrapper(sunrealtype t,
                         N_Vector yy,
                         SUNDIALS_MAYBE_UNUSED N_Vector yp,
                         N_Vector yyB,
                         N_Vector ypB,
                         N_Vector rrB,
                         void* user_dataB)
{
  DataB* data = (DataB*)user_dataB;

  /* Evaluate user supplied residual function. */
  return data->db_resB(t, yy, yyB, ypB, rrB, data->db_user_data);
}

static int DDLsJacFnBWrapper(sunrealtype t,
                             sunrealtype cj,
                             N_Vector yy,
                             SUNDIALS_MAYBE_UNUSED N_Vector yp,
                             N_Vector yyB,
                             N_Vector ypB,
                             N_Vector rrB,
                             SUNMatrix JB,
                             void* user_dataB,
                             N_Vector tmp1,
                             N_Vector tmp2,
                             N_Vector tmp3)
{
  DataB* data = (DataB*)user_dataB;

  /* Evaluate user supplied jacobian function. */
  return data
    ->db_jacB(t, cj, yy, yyB, ypB, rrB, JB, data->db_user_data, tmp1, tmp2, tmp3);
}

static int DDQuadRhsFnBWrapper(sunrealtype t,
                               N_Vector yy,
                               SUNDIALS_MAYBE_UNUSED N_Vector yp,
                               N_Vector yyB,
                               N_Vector ypB,
                               N_Vector rhsBQ,
                               void* user_dataB)
{
  DataB* data = (DataB*)user_dataB;

  /* Evaluate user supplied quadrature function. */
  return data->db_quadB(t, yy, yyB, ypB, rhsBQ, data->db_user_data);
}

/* --------------------------------------------------------------------------
 * DDSolveB
 * -------------------------------------------------------------------------- */

int DDSolveB(DDMem dd_mem, sunrealtype tBout, int itaskB)
{
  if (dd_mem == NULL)
  {
    DDHandleErrWithCtx(DD_ERR_DD_MEM_NULL, NULL);
    return DD_ERR_GENERIC;
  }

  SUNFunctionBegin(dd_mem->sunctx);

  IDAMem ida_mem        = dd_mem->ida_mem;
  IDAadjMem ida_adj_mem = ida_mem->ida_adj_mem;

  /* Direction of the forward problem. */

  int sign = (ida_adj_mem->ia_tfinal - dd_mem->dd_tinitial > ZERO) ? 1 : -1;

  if (sign * (tBout - dd_mem->dd_tinitial) < ZERO)
  {
    DDHandleErr(SUN_ERR_ARG_OUTOFRANGE);
    return SUN_ERR_ARG_OUTOFRANGE;
  }

  if ((itaskB != IDA_ONE_STEP) && (itaskB != IDA_NORMAL))
  {
    DDHandleErr(SUN_ERR_ARG_OUTOFRANGE);
    return SUN_ERR_ARG_OUTOFRANGE;
  }

  /* Starting from the current pivot checkpoint, loop through checkpoints until
     the current time of any of the backwards problems comes after the start
     time of the checkpoint (in the forward direction).  */

  /* This code is code from `IDASolveB` re-purposed to find the first relevant
     DD checkpoint. */

  DDckpntMem ck_mem = NULL;

  for (ck_mem = dd_mem->ck_mem_cur; ck_mem != NULL; ck_mem = ck_mem->ck_next)
  {
    sunbooleantype got_ckpnt = SUNFALSE;

    for (IDABMem b = ida_adj_mem->IDAB_mem; b != NULL; b = b->ida_next)
    {
      sunrealtype tBn = b->IDA_mem->ida_tn;

      if (sign * (ck_mem->ck_t0 - tBn) < ZERO)
      {
        got_ckpnt = SUNTRUE;
        break;
      }

      if ((itaskB == IDA_NORMAL) && (tBn == ck_mem->ck_t0) &&
          (sign * (tBout - ck_mem->ck_t0) >= ZERO))
      {
        got_ckpnt = SUNTRUE;
        break;
      }
    }

    if (got_ckpnt) { break; }
  }

  SUNAssert(ck_mem != NULL, SUN_ERR_OP_FAIL);

  int flag = SUN_ERR_UNREACHABLE;

  while (SUNTRUE)
  {
    /* If we found a new checkpoint we need to update the state. */

    if (ck_mem != dd_mem->ck_mem_cur)
    {
      DDDAEState state = ck_mem->ck_state;
      ck_mem->ck_state =
        dd_mem->dd_state; /* this state will be destroyed when we destroy the
        checkpoints. */
      dd_mem->dd_state = state;

      /* Set the initial time step for the IDA solver becase it is resetted on
       each call to `IDASolve` after a call to `IDAReInit`. This field is used
       by `IDASolveB` when reading the IDA checkpoint at a DD pivot.  */

      ida_mem->ida_h0u = ck_mem->ck_h0u;

      /* Set the IDA checkpoints to the IDA checkpoints associated with this
         particular DD checkpoint. */

      ida_adj_mem->ck_mem = ck_mem->ida_ck_mem;

      dd_mem->ck_mem_cur = ck_mem;
    }

    if (itaskB == IDA_NORMAL)
    {
      /* Solve to the next checkpoint or `tBout`, whichever comes first. */

      sunrealtype ck_t0 = ck_mem->ck_t0;
      sunrealtype t     = sign * (ck_t0 - tBout) <= ZERO ? tBout : ck_t0;

      flag = IDASolveB(ida_mem, t, IDA_NORMAL);
      if (flag < 0)
      {
        DDHandleErr(DD_ERR_IDA_ERR);
        return DD_ERR_IDA_ERR;
      }

      /* If there was an error, we reached `tBout`, or we ran out of checkpoints
         we are done. */

      if ((flag < 0) || (t == tBout) || (ck_mem->ck_next == NULL)) { break; }

      /* If we come this far we still have work to do to reach `tBout`. Change
         to the next checkpoint and continue. */

      ck_mem = ck_mem->ck_next;
    }
    else if (itaskB == IDA_ONE_STEP)
    {
      flag = IDASolveB(ida_mem, tBout, IDA_ONE_STEP);
      if (flag < 0)
      {
        DDHandleErr(DD_ERR_IDA_ERR);
        return DD_ERR_IDA_ERR;
      }

      break;
    }
    else
    {
      return SUN_ERR_UNREACHABLE;
    }
  }

  return flag;
}

/* --------------------------------------------------------------------------
 * DDCalcICB
 * -------------------------------------------------------------------------- */

int DDCalcICB(DDMem dd_mem, int which, sunrealtype tBout1, N_Vector Y)
{
  if (dd_mem == NULL)
  {
    DDHandleErrWithCtx(DD_ERR_DD_MEM_NULL, NULL);
    return DD_ERR_GENERIC;
  }

  SUNFunctionBegin(dd_mem->sunctx);

  if (Y == NULL)
  {
    DDHandleErr(SUN_ERR_ARG_CORRUPT);
    return SUN_ERR_ARG_CORRUPT;
  }

  if (IDACalcICB(dd_mem->ida_mem,
                 which,
                 tBout1,
                 Y,
                 /* Not used by the residual function. */ Y) < 0)
  {
    DDHandleErr(DD_ERR_IDA_ERR);
    return DD_ERR_IDA_ERR;
  }

  return DD_SUCCESS;
}

/* --------------------------------------------------------------------------
 * DDSetJacFn
 * -------------------------------------------------------------------------- */

int DDSetJacFn(DDMem dd_mem, DDLsJacFn jacfn)
{
  if (dd_mem == NULL)
  {
    DDHandleErrWithCtx(DD_ERR_DD_MEM_NULL, NULL);
    return DD_ERR_DD_MEM_NULL;
  }

  SUNFunctionBegin(dd_mem->sunctx);

  if ((jacfn.id != DD_JAC_1) && (jacfn.id != DD_JAC_2))
  {
    DDHandleErr(SUN_ERR_ARG_OUTOFRANGE);
    return SUN_ERR_ARG_OUTOFRANGE;
  }

  IDAMem ida_mem = dd_mem->ida_mem;

  if (jacfn.id == DD_JAC_1)
  {
    SUNMatrix J = dd_mem->dd_J;

    if (J != NULL)
    {
      int flag              = SUN_SUCCESS;
      const SUNMatrix_ID id = SUNMatGetID(J);

      if (id == SUNMATRIX_DENSE)
      {
        flag = IDASetJacFn(ida_mem, DDLsJacFnWrapper1_Dense);
      }
      else if ((id == SUNMATRIX_SPARSE) && (SM_SPARSETYPE_S(J) == CSR_MAT))
      {
        flag = IDASetJacFn(ida_mem, DDLsJacFnWrapper1_CSR);
      }
      else
      {
        DDHandleErr(SUN_ERR_ARG_WRONGTYPE);
        return SUN_ERR_ARG_WRONGTYPE;
      }

      if (flag < 0)
      {
        DDHandleErr(DD_ERR_IDA_ERR);
        return DD_ERR_IDA_ERR;
      }
    }

    dd_mem->dd_jacfn1 = jacfn.fn.jacfn1;
    dd_mem->dd_jacfn2 = NULL;
  }
  else if (jacfn.id == DD_JAC_2)
  {
    if (IDASetJacFn(ida_mem, DDLsJacFnWrapper2) < 0)
    {
      DDHandleErr(DD_ERR_IDA_ERR);
      return DD_ERR_IDA_ERR;
    }

    dd_mem->dd_jacfn1 = NULL;
    dd_mem->dd_jacfn2 = jacfn.fn.jacfn2;
  }
  else
  {
    return SUN_ERR_UNREACHABLE;
  }

  return DD_SUCCESS;
}

static int DDLsJacFnWrapper1_Dense(sunrealtype t,
                                   sunrealtype cj,
                                   N_Vector yy,
                                   SUNDIALS_MAYBE_UNUSED N_Vector yp,
                                   N_Vector rr,
                                   SUNMatrix J,
                                   void* user_data,
                                   N_Vector tmp1,
                                   N_Vector tmp2,
                                   N_Vector tmp3)
{
  const DDMem dd_mem     = (DDMem)user_data;
  const DDStaticInfo si  = dd_mem->dd_si;
  const DDDAEState state = dd_mem->dd_state;

  /* Compute lower part of the Jacobian containing rows for the differential
     equations. */

  const sunindextype N_diff  = si->N_diff;
  const sunindextype row_ofs = SM_ROWS_D(J) - N_diff;

  for (sunindextype i = 0; i < N_diff; ++i)
  {
    const Pair_sunindextype p           = state->diff_var_aliases[i];
    SM_ELEMENT_D(J, row_ofs + i, p.fst) = -cj;
    SM_ELEMENT_D(J, row_ofs + i, p.snd) = ONE;
  }

  /* Call user supplied Jacobian callback function. */

  void* ud = dd_mem->dd_user_data;
  int flag = dd_mem->dd_jacfn1(t, yy, rr, J, ud, tmp1, tmp2, tmp3);

  return flag;
}

static int DDLsJacFnWrapper1_CSR(sunrealtype t,
                                 sunrealtype cj,
                                 N_Vector yy,
                                 SUNDIALS_MAYBE_UNUSED N_Vector yp,
                                 N_Vector rr,
                                 SUNMatrix J,
                                 void* user_data,
                                 N_Vector tmp1,
                                 N_Vector tmp2,
                                 N_Vector tmp3)
{
  const DDMem dd_mem     = (DDMem)user_data;
  const DDStaticInfo si  = dd_mem->dd_si;
  const DDDAEState state = dd_mem->dd_state;

  /* Compute lower part of the Jacobian containing rows for the differential
     equations. */

  const sunindextype N_diff = si->N_diff;

  sunindextype row = SM_ROWS_S(J) - N_diff, nnz = SM_NNZ_S(J) - 2 * N_diff;
  for (sunindextype i = 0; i < N_diff; ++i)
  {
    const Pair_sunindextype p  = state->diff_var_aliases[i];
    SM_INDEXVALS_S(J)[nnz]     = p.fst;
    SM_DATA_S(J)[nnz]          = -cj;
    SM_INDEXVALS_S(J)[nnz + 1] = p.snd;
    SM_DATA_S(J)[nnz + 1]      = ONE;
    SM_INDEXPTRS_S(J)[row]     = nnz;

    row += 1;
    nnz += 2;
  }

  SM_INDEXPTRS_S(J)[row] = nnz;

  /* Call user supplied Jacobian callback function. */

  void* ud = dd_mem->dd_user_data;
  int flag = dd_mem->dd_jacfn1(t, yy, rr, J, ud, tmp1, tmp2, tmp3);

  return flag;
}

static int DDLsJacFnWrapper2(sunrealtype t,
                             sunrealtype cj,
                             N_Vector yy,
                             SUNDIALS_MAYBE_UNUSED N_Vector yp,
                             N_Vector rr,
                             SUNMatrix J,
                             void* user_data,
                             N_Vector tmp1,
                             N_Vector tmp2,
                             N_Vector tmp3)
{
  const DDMem dd_mem     = (DDMem)user_data;
  const DDDAEState state = dd_mem->dd_state;

  /* Call user supplied Jacobian callback function. */

  void* ud = dd_mem->dd_user_data;
  int flag = dd_mem->dd_jacfn2(state->yy_diff_alias_row,
                               state->yp_diff_alias_row,
                               t,
                               cj,
                               yy,
                               rr,
                               J,
                               ud,
                               tmp1,
                               tmp2,
                               tmp3);

  return flag;
}

int DDJacFn_CSC(sunindextype M,
                DDLsJacColFn_CSC* fn,
                const sunindextype yy_diff_alias_row[static 1],
                const sunindextype yp_diff_alias_row[static 1],
                sunrealtype t,
                sunrealtype cj,
                N_Vector Y,
                N_Vector R,
                SUNMatrix J,
                void* user_data,
                N_Vector tmp1,
                N_Vector tmp2,
                N_Vector tmp3)
{
  SUNFunctionBegin(J->sunctx);

  SUNCheck(SUNMatGetID(J) == SUNMATRIX_SPARSE, SUN_ERR_ARG_WRONGTYPE);
  SUNCheck(SM_SPARSETYPE_S(J) == CSC_MAT, SUN_ERR_ARG_OUTOFRANGE);
  SUNCheck((0 < M) && (M < SM_ROWS_S(J)), SUN_ERR_ARG_OUTOFRANGE);

  const sunindextype N = SM_COLUMNS_S(J);

  SUNCheck(M < N, SUN_ERR_ARG_OUTOFRANGE);

  sunindextype nnz = 0;
  for (sunindextype j = 0; j < N; ++j)
  {
    SM_INDEXPTRS_S(J)[j] = nnz;
    int flag             = fn(j, t, Y, R, J, &nnz, user_data, tmp1, tmp2, tmp3);
    if (flag != 0) { return flag; }

    sunindextype yy_alias_row = yy_diff_alias_row[j];
    if (yy_alias_row >= 0)
    {
      SM_DATA_S(J)[nnz]      = -cj;
      SM_INDEXVALS_S(J)[nnz] = yy_alias_row;
      nnz += 1;
    }

    sunindextype yp_alias_row = yp_diff_alias_row[j];
    if (yp_alias_row >= 0)
    {
      SM_DATA_S(J)[nnz]      = ONE;
      SM_INDEXVALS_S(J)[nnz] = yp_alias_row;
      nnz += 1;
    }
  }

  SM_INDEXPTRS_S(J)[SM_NP_S(J)] = nnz;

  return SUN_SUCCESS;
}

/* --------------------------------------------------------------------------
 * Quadrature Functions
 * -------------------------------------------------------------------------- */

int DDQuadInitB(DDMem dd_mem, int indexB, DDQuadRhsFnB rhsQB, N_Vector yQB0)
{
  if (dd_mem == NULL)
  {
    DDHandleErrWithCtx(DD_ERR_DD_MEM_NULL, NULL);
    return DD_ERR_DD_MEM_NULL;
  }

  SUNFunctionBegin(dd_mem->sunctx);

  SUNAssert(rhsQB != NULL, SUN_ERR_ARG_CORRUPT);
  SUNAssert(yQB0 != NULL, SUN_ERR_ARG_CORRUPT);

  DynArr_ProbB pbs = dd_mem->dd_probBs;

  int flag = SUN_ERR_ARG_OUTOFRANGE;

  for (size_t i = 0; i < DA_LENGTH(pbs); ++i)
  {
    if (DA_Ith(pbs, i).pb_which == indexB)
    {
      DA_Ith(pbs, i).pb_data->db_quadB = rhsQB;

      flag = SUN_SUCCESS;

      break;
    }
  }

  SUNAssert(flag < 0, flag);

  if (IDAQuadInitB(dd_mem->ida_mem, indexB, DDQuadRhsFnBWrapper, yQB0) < 0)
  {
    DDHandleErr(DD_ERR_IDA_ERR);
    flag = DD_ERR_IDA_ERR;
  }

  return flag;
}

int DDGetQuadB(DDMem dd_mem, int indexB, sunrealtype* tret, N_Vector yQB)
{
  if (dd_mem == NULL)
  {
    DDHandleErrWithCtx(DD_ERR_DD_MEM_NULL, NULL);
    return DD_ERR_DD_MEM_NULL;
  }

  SUNFunctionBegin(dd_mem->sunctx);

  SUNAssert(yQB != NULL, SUN_ERR_ARG_CORRUPT);

  if (IDAGetQuadB(dd_mem->ida_mem, indexB, tret, yQB) < 0)
  {
    DDHandleErr(DD_ERR_IDA_ERR);
    return DD_ERR_IDA_ERR;
  }

  return SUN_SUCCESS;
}

/* --------------------------------------------------------------------------
 * Remaining Setters and Getters
 * -------------------------------------------------------------------------- */

void* DDGetIDAMem(DDMem dd_mem)
{
  if (dd_mem == NULL)
  {
    DDHandleErrWithCtx(DD_ERR_DD_MEM_NULL, NULL);
    return NULL;
  }

  return dd_mem->ida_mem;
}

const uint8_t* DDGetSpec(DDMem dd_mem)
{
  if (dd_mem == NULL)
  {
    DDHandleErrWithCtx(DD_ERR_DD_MEM_NULL, NULL);
    return NULL;
  }

  return dd_mem->dd_spec;
}

N_Vector DDGetId(DDMem dd_mem)
{
  if (dd_mem == NULL)
  {
    DDHandleErrWithCtx(DD_ERR_DD_MEM_NULL, NULL);
    return NULL;
  }

  return dd_mem->dd_id;
}

int DDSetLinearSolver(DDMem dd_mem, SUNLinearSolver LS, SUNMatrix A)
{
  if (dd_mem == NULL)
  {
    DDHandleErrWithCtx(DD_ERR_DD_MEM_NULL, NULL);
    return DD_ERR_GENERIC;
  }

  SUNFunctionBegin(dd_mem->sunctx);

  if (IDASetLinearSolver(dd_mem->ida_mem, LS, A) < 0)
  {
    DDHandleErr(DD_ERR_IDA_ERR);
    return DD_ERR_IDA_ERR;
  }

  dd_mem->dd_J = A;

  return DD_SUCCESS;
}

int DDSSTolerances(DDMem dd_mem, sunrealtype reltol, sunrealtype abstol)
{
  if (dd_mem == NULL)
  {
    DDHandleErrWithCtx(DD_ERR_DD_MEM_NULL, NULL);
    return DD_ERR_GENERIC;
  }

  SUNFunctionBegin(dd_mem->sunctx);

  if (IDASStolerances(dd_mem->ida_mem, reltol, abstol) < 0)
  {
    DDHandleErr(DD_ERR_IDA_ERR);
    return DD_ERR_IDA_ERR;
  }

  return DD_SUCCESS;
}

int DDSetStopTime(DDMem dd_mem, sunrealtype tstop)
{
  if (dd_mem == NULL)
  {
    DDHandleErrWithCtx(DD_ERR_DD_MEM_NULL, NULL);
    return DD_ERR_GENERIC;
  }

  SUNFunctionBegin(dd_mem->sunctx);

  if (IDASetStopTime(dd_mem->ida_mem, tstop) < 0)
  {
    DDHandleErr(DD_ERR_IDA_ERR);
    return DD_ERR_IDA_ERR;
  }

  return DD_SUCCESS;
}

int DDSetUserData(DDMem dd_mem, void* user_data)
{
  if (dd_mem == NULL)
  {
    DDHandleErrWithCtx(DD_ERR_DD_MEM_NULL, NULL);
    return DD_ERR_GENERIC;
  }

  dd_mem->dd_user_data = user_data;

  return DD_SUCCESS;
}

int DDGetSens(DDMem dd_mem, sunrealtype* tret, N_Vector* YS)
{
  if (dd_mem == NULL)
  {
    DDHandleErrWithCtx(DD_ERR_DD_MEM_NULL, NULL);
    return DD_ERR_GENERIC;
  }

  SUNFunctionBegin(dd_mem->sunctx);

  if (tret == NULL)
  {
    DDHandleErr(SUN_ERR_ARG_CORRUPT);
    return SUN_ERR_ARG_CORRUPT;
  }

  if (YS == NULL)
  {
    DDHandleErr(SUN_ERR_ARG_CORRUPT);
    return SUN_ERR_ARG_CORRUPT;
  }

  if (IDAGetSens(dd_mem->ida_mem, tret, YS) < 0)
  {
    DDHandleErr(DD_ERR_IDA_ERR);
    return DD_ERR_IDA_ERR;
  }

  return DD_SUCCESS;
}

char* DDGetReturnFlagName(long int flag)
{
#define DD_ERR_EXPAND_TO_CASE(name, description) \
  case name: sprintf(buf, "name"); break;

  char* buf = malloc(24 * sizeof(*buf));

  /* clang-format off */
  switch (flag)
  {
    DD_ERR_CODE_LIST(DD_ERR_EXPAND_TO_CASE)
    SUN_ERR_CODE_LIST(DD_ERR_EXPAND_TO_CASE)
    default: free(buf); return IDAGetReturnFlagName(flag);
  }

  /* clang-format on */
  return buf;
}

int DDGetConsistentIC(DDMem dd_mem, N_Vector Y)
{
  if (dd_mem == NULL)
  {
    DDHandleErrWithCtx(DD_ERR_DD_MEM_NULL, NULL);
    return DD_ERR_GENERIC;
  }

  SUNFunctionBegin(dd_mem->sunctx);

  if (Y == NULL)
  {
    DDHandleErr(SUN_ERR_ARG_CORRUPT);
    return SUN_ERR_ARG_CORRUPT;
  }

  if (IDAGetConsistentIC(dd_mem->ida_mem, Y, NULL) < 0)
  {
    DDHandleErr(DD_ERR_IDA_ERR);
    return DD_ERR_IDA_ERR;
  }

  return DD_SUCCESS;
}

int DDGetSensConsistentIC(DDMem dd_mem, N_Vector* YS)
{
  if (dd_mem == NULL)
  {
    DDHandleErrWithCtx(DD_ERR_DD_MEM_NULL, NULL);
    return DD_ERR_GENERIC;
  }

  SUNFunctionBegin(dd_mem->sunctx);

  if (YS == NULL)
  {
    DDHandleErr(SUN_ERR_ARG_CORRUPT);
    return SUN_ERR_ARG_CORRUPT;
  }

  if (IDAGetSensConsistentIC(dd_mem->ida_mem, YS, NULL) < 0)
  {
    DDHandleErr(DD_ERR_IDA_ERR);
    return DD_ERR_IDA_ERR;
  }

  return DD_SUCCESS;
}

int DDSetSensParams(DDMem dd_mem, sunrealtype* p, sunrealtype* pbar, int* plist)
{
  if (dd_mem == NULL)
  {
    DDHandleErrWithCtx(DD_ERR_DD_MEM_NULL, NULL);
    return DD_ERR_GENERIC;
  }

  SUNFunctionBegin(dd_mem->sunctx);

  if (p == NULL)
  {
    DDHandleErr(SUN_ERR_ARG_CORRUPT);
    return SUN_ERR_ARG_CORRUPT;
  }

  if (IDASetSensParams(dd_mem->ida_mem, p, pbar, plist) < 0)
  {
    DDHandleErr(DD_ERR_IDA_ERR);
    return DD_ERR_IDA_ERR;
  }

  return DD_SUCCESS;
}

int DDSensEEtolerances(DDMem dd_mem)
{
  if (dd_mem == NULL)
  {
    DDHandleErrWithCtx(DD_ERR_DD_MEM_NULL, NULL);
    return DD_ERR_GENERIC;
  }

  SUNFunctionBegin(dd_mem->sunctx);

  if (IDASensEEtolerances(dd_mem->ida_mem) < 0)
  {
    DDHandleErr(DD_ERR_IDA_ERR);
    return DD_ERR_IDA_ERR;
  }

  return DD_SUCCESS;
}

int DDSetLinearSolverB(DDMem dd_mem, int indexB, SUNLinearSolver LS, SUNMatrix A)
{
  if (dd_mem == NULL)
  {
    DDHandleErrWithCtx(DD_ERR_DD_MEM_NULL, NULL);
    return DD_ERR_GENERIC;
  }

  SUNFunctionBegin(dd_mem->sunctx);

  if (LS == NULL)
  {
    DDHandleErr(SUN_ERR_ARG_CORRUPT);
    return SUN_ERR_ARG_CORRUPT;
  }

  if (A == NULL)
  {
    DDHandleErr(SUN_ERR_ARG_CORRUPT);
    return SUN_ERR_ARG_CORRUPT;
  }

  if (IDASetLinearSolverB(dd_mem->ida_mem, indexB, LS, A) < 0)
  {
    DDHandleErr(DD_ERR_IDA_ERR);
    return DD_ERR_IDA_ERR;
  }

  return DD_SUCCESS;
}

int DDSetJacFnB(DDMem dd_mem, int indexB, DDLsJacFnB jacfn)
{
  if (dd_mem == NULL)
  {
    DDHandleErrWithCtx(DD_ERR_DD_MEM_NULL, NULL);
    return DD_ERR_GENERIC;
  }

  SUNFunctionBegin(dd_mem->sunctx);

  SUNAssert(jacfn != NULL, SUN_ERR_ARG_CORRUPT);

  DynArr_ProbB pbs = dd_mem->dd_probBs;

  int flag = SUN_ERR_ARG_OUTOFRANGE;

  for (size_t i = 0; i < DA_LENGTH(pbs); ++i)
  {
    if (DA_Ith(pbs, i).pb_which == indexB)
    {
      DA_Ith(pbs, i).pb_data->db_jacB = jacfn;

      flag = SUN_SUCCESS;

      break;
    }
  }

  SUNAssert(flag < 0, flag);

  if (IDASetJacFnB(dd_mem->ida_mem, indexB, DDLsJacFnBWrapper) < 0)
  {
    DDHandleErr(DD_ERR_IDA_ERR);
    flag = DD_ERR_IDA_ERR;
  }

  return flag;
}

int DDSStolerancesB(DDMem dd_mem, int indexB, sunrealtype reltolB, sunrealtype abstolB)
{
  if (dd_mem == NULL)
  {
    DDHandleErrWithCtx(DD_ERR_DD_MEM_NULL, NULL);
    return DD_ERR_GENERIC;
  }

  SUNFunctionBegin(dd_mem->sunctx);

  if (IDASStolerancesB(dd_mem->ida_mem, indexB, reltolB, abstolB) < 0)
  {
    DDHandleErr(DD_ERR_IDA_ERR);
    return DD_ERR_IDA_ERR;
  }

  return DD_SUCCESS;
}

int DDSetUserDataB(DDMem dd_mem, int indexB, void* user_dataB)
{
  if (dd_mem == NULL)
  {
    DDHandleErrWithCtx(DD_ERR_DD_MEM_NULL, NULL);
    return DD_ERR_GENERIC;
  }

  SUNFunctionBegin(dd_mem->sunctx);

  DynArr_ProbB pbs = dd_mem->dd_probBs;

  int flag = SUN_ERR_ARG_OUTOFRANGE;
  for (size_t i = 0; i < DA_LENGTH(pbs); ++i)
  {
    if (DA_Ith(pbs, i).pb_which == indexB)
    {
      DA_Ith(pbs, i).pb_data->db_user_data = user_dataB;

      flag = SUN_SUCCESS;

      break;
    }
  }

  if (flag < 0) { DDHandleErr(flag); }

  return flag;
}

int DDSetIdB(DDMem dd_mem, int which, N_Vector idB)
{
  if (dd_mem == NULL)
  {
    DDHandleErrWithCtx(DD_ERR_DD_MEM_NULL, NULL);
    return DD_ERR_GENERIC;
  }

  SUNFunctionBegin(dd_mem->sunctx);

  if (idB == NULL)
  {
    DDHandleErr(SUN_ERR_ARG_CORRUPT);
    return SUN_ERR_ARG_CORRUPT;
  }

  if (IDASetIdB(dd_mem->ida_mem, which, idB) < 0)
  {
    DDHandleErr(DD_ERR_IDA_ERR);
    return DD_ERR_IDA_ERR;
  }

  return DD_SUCCESS;
}

int DDGetB(DDMem dd_mem,
           int which,
           sunrealtype tret[static 1],
           N_Vector yyB,
           N_Vector ypB)
{
  if (dd_mem == NULL)
  {
    DDHandleErrWithCtx(DD_ERR_DD_MEM_NULL, NULL);
    return DD_ERR_GENERIC;
  }

  SUNFunctionBegin(dd_mem->sunctx);

  if (IDAGetB(dd_mem->ida_mem, which, tret, yyB, ypB) < 0)
  {
    DDHandleErr(DD_ERR_IDA_ERR);
    return DD_ERR_IDA_ERR;
  }

  return DD_SUCCESS;
}

int DDGetConsistentICB(DDMem dd_mem, int which, N_Vector yyB0, N_Vector ypB0)
{
  if (dd_mem == NULL)
  {
    DDHandleErrWithCtx(DD_ERR_DD_MEM_NULL, NULL);
    return DD_ERR_GENERIC;
  }

  SUNFunctionBegin(dd_mem->sunctx);

  if (IDAGetConsistentICB(dd_mem->ida_mem, which, yyB0, ypB0) < 0)
  {
    DDHandleErr(DD_ERR_IDA_ERR);
    return DD_ERR_IDA_ERR;
  }

  return DD_SUCCESS;
}
