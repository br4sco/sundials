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
  DDDAEState ck_state;
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

static DDckpntMem DDckpntCreate(sunrealtype t, DDDAEState state)
{
  DDckpntMem ck_mem = calloc(1, sizeof(*ck_mem));
  if (ck_mem == NULL) { return NULL; }

  ck_mem->ck_state = DDDAEStateClone(state);
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
 * DDIDAAckpntAllocVectors, DDIDAAckpntInit, DDIDAAckpntCopyVectors,
 * DDAIDAckpntNew
 *
 * These functions are copies of the internal (`static`, not exported outside
 * src/idas/idaa.c) SUNDIALS functions `IDAAckpntAllocVectors`,
 * `IDAAckpntCopyVectors`, `DDIDAAckpntInit`, and `IDAAckpntNew`, respectively.
 * -------------------------------------------------------------------------- */

extern int IDAGetSolution(void* ida_mem,
                          sunrealtype t,
                          N_Vector yret,
                          N_Vector ypret);

static sunbooleantype DDIDAAckpntAllocVectors(IDAMem ida_mem, IDAckpntMem ck_mem)
{
  int j, jj;

  for (j = 0; j < ck_mem->ck_phi_alloc; j++)
  {
    ck_mem->ck_phi[j] = N_VClone(ida_mem->ida_tempv1);
    if (ck_mem->ck_phi[j] == NULL)
    {
      for (jj = 0; jj < j; jj++) { N_VDestroy(ck_mem->ck_phi[jj]); }
      return SUNFALSE;
    }
  }

  if (ck_mem->ck_quadr)
  {
    for (j = 0; j < ck_mem->ck_phi_alloc; j++)
    {
      ck_mem->ck_phiQ[j] = N_VClone(ida_mem->ida_eeQ);
      if (ck_mem->ck_phiQ[j] == NULL)
      {
        for (jj = 0; jj < j; jj++) { N_VDestroy(ck_mem->ck_phiQ[jj]); }
        for (jj = 0; jj < ck_mem->ck_phi_alloc; jj++)
        {
          N_VDestroy(ck_mem->ck_phi[jj]);
        }
        return SUNFALSE;
      }
    }
  }

  if (ck_mem->ck_sensi)
  {
    for (j = 0; j < ck_mem->ck_phi_alloc; j++)
    {
      ck_mem->ck_phiS[j] = N_VCloneVectorArray(ida_mem->ida_Ns,
                                               ida_mem->ida_tempv1);
      if (ck_mem->ck_phiS[j] == NULL)
      {
        for (jj = 0; jj < j; jj++)
        {
          N_VDestroyVectorArray(ck_mem->ck_phiS[jj], ida_mem->ida_Ns);
        }
        if (ck_mem->ck_quadr)
        {
          for (jj = 0; jj < ck_mem->ck_phi_alloc; jj++)
          {
            N_VDestroy(ck_mem->ck_phiQ[jj]);
          }
        }
        for (jj = 0; jj < ck_mem->ck_phi_alloc; jj++)
        {
          N_VDestroy(ck_mem->ck_phi[jj]);
        }
        return SUNFALSE;
      }
    }
  }

  if (ck_mem->ck_quadr_sensi)
  {
    for (j = 0; j < ck_mem->ck_phi_alloc; j++)
    {
      ck_mem->ck_phiQS[j] = N_VCloneVectorArray(ida_mem->ida_Ns,
                                                ida_mem->ida_eeQ);
      if (ck_mem->ck_phiQS[j] == NULL)
      {
        for (jj = 0; jj < j; jj++)
        {
          N_VDestroyVectorArray(ck_mem->ck_phiQS[jj], ida_mem->ida_Ns);
        }
        for (jj = 0; jj < ck_mem->ck_phi_alloc; jj++)
        {
          N_VDestroyVectorArray(ck_mem->ck_phiS[jj], ida_mem->ida_Ns);
        }
        if (ck_mem->ck_quadr)
        {
          for (jj = 0; jj < ck_mem->ck_phi_alloc; jj++)
          {
            N_VDestroy(ck_mem->ck_phiQ[jj]);
          }
        }
        for (jj = 0; jj < ck_mem->ck_phi_alloc; jj++)
        {
          N_VDestroy(ck_mem->ck_phi[jj]);
        }
        return SUNFALSE;
      }
    }
  }

  return SUNTRUE;
}

static void DDIDAAckpntCopyVectors(IDAMem ida_mem, IDAckpntMem ck_mem)
{
  int j, is;

  for (j = 0; j < ck_mem->ck_phi_alloc; j++) { ida_mem->ida_cvals[j] = ONE; }

  (void)N_VScaleVectorArray(ck_mem->ck_phi_alloc,
                            ida_mem->ida_cvals,
                            ida_mem->ida_phi,
                            ck_mem->ck_phi);

  if (ck_mem->ck_quadr)
  {
    (void)N_VScaleVectorArray(ck_mem->ck_phi_alloc,
                              ida_mem->ida_cvals,
                              ida_mem->ida_phiQ,
                              ck_mem->ck_phiQ);
  }

  if (ck_mem->ck_sensi || ck_mem->ck_quadr_sensi)
  {
    for (j = 0; j < ck_mem->ck_phi_alloc; j++)
    {
      for (is = 0; is < ida_mem->ida_Ns; is++)
      {
        ida_mem->ida_cvals[j * ida_mem->ida_Ns + is] = ONE;
      }
    }
  }

  if (ck_mem->ck_sensi)
  {
    for (j = 0; j < ck_mem->ck_phi_alloc; j++)
    {
      for (is = 0; is < ida_mem->ida_Ns; is++)
      {
        ida_mem->ida_Xvecs[j * ida_mem->ida_Ns + is] = ida_mem->ida_phiS[j][is];
        ida_mem->ida_Zvecs[j * ida_mem->ida_Ns + is] = ck_mem->ck_phiS[j][is];
      }
    }

    (void)N_VScaleVectorArray(ck_mem->ck_phi_alloc * ida_mem->ida_Ns,
                              ida_mem->ida_cvals,
                              ida_mem->ida_Xvecs,
                              ida_mem->ida_Zvecs);
  }

  if (ck_mem->ck_quadr_sensi)
  {
    for (j = 0; j < ck_mem->ck_phi_alloc; j++)
    {
      for (is = 0; is < ida_mem->ida_Ns; is++)
      {
        ida_mem->ida_Xvecs[j * ida_mem->ida_Ns + is] = ida_mem->ida_phiQS[j][is];
        ida_mem->ida_Zvecs[j * ida_mem->ida_Ns + is] = ck_mem->ck_phiQS[j][is];
      }
    }

    (void)N_VScaleVectorArray(ck_mem->ck_phi_alloc * ida_mem->ida_Ns,
                              ida_mem->ida_cvals,
                              ida_mem->ida_Xvecs,
                              ida_mem->ida_Zvecs);
  }
}

static IDAckpntMem DDIDAAckpntInit(IDAMem IDA_mem)
{
  IDAckpntMem ck_mem;

  /* Allocate space for ckdata */
  ck_mem = (IDAckpntMem)malloc(sizeof(struct IDAckpntMemRec));
  if (NULL == ck_mem) { return (NULL); }

  ck_mem->ck_t0  = IDA_mem->ida_tn;
  ck_mem->ck_nst = 0;
  ck_mem->ck_kk  = 1;
  ck_mem->ck_hh  = ZERO;

  /* Test if we need to carry quadratures */
  ck_mem->ck_quadr = IDA_mem->ida_quadr && IDA_mem->ida_errconQ;

  /* Test if we need to carry sensitivities */
  ck_mem->ck_sensi = IDA_mem->ida_sensi;
  if (ck_mem->ck_sensi) { ck_mem->ck_Ns = IDA_mem->ida_Ns; }

  /* Test if we need to carry quadrature sensitivities */
  ck_mem->ck_quadr_sensi = IDA_mem->ida_quadr_sensi && IDA_mem->ida_errconQS;

  /* Alloc 3: current order, i.e. 1,  +   2. */
  ck_mem->ck_phi_alloc = 3;

  if (!DDIDAAckpntAllocVectors(IDA_mem, ck_mem))
  {
    free(ck_mem);
    ck_mem = NULL;
    return (NULL);
  }
  /* Save phi* vectors from IDA_mem to ck_mem. */
  DDIDAAckpntCopyVectors(IDA_mem, ck_mem);

  /* Next in list */
  ck_mem->ck_next = NULL;

  return (ck_mem);
}

static IDAckpntMem DDIDAAckpntNew(IDAMem ida_mem)
{
  IDAckpntMem ck_mem = (IDAckpntMem)malloc(sizeof(struct IDAckpntMemRec));
  if (ck_mem == NULL) { return NULL; }

  ck_mem->ck_nst = ida_mem->ida_nst;

  /* Deliberately NOT ida_mem->ida_tretlast: upstream's checkpoints are only
     ever created inside IDASolveF's own IDA_ONE_STEP-mode stepping loop
     (idaa.c:569), where tn == tretlast always holds right after a step, so
     copying tretlast verbatim is a no-op there. DD drives the forward
     solve in IDA_NORMAL mode (per-tstep outputs), where IDA can land
     internally past the requested output time before interpolating back --
     so tn and tretlast legitimately diverge at the moment a pivot happens.
     If we captured the stale tretlast here, then restoring it via
     IDAAckpntGet() and immediately replaying with IDA_ONE_STEP (as
     IDAAdataStore() does) would hit IDAStopTest1()'s "tn already past
     tretlast" case (idas.c:5663) and return the CURRENT point again
     without taking a real step -- producing a duplicate/degenerate first
     replay point and, eventually, step-size collapse. Keep tn/tretlast
     consistent in the checkpoint, matching the invariant vanilla
     checkpoints always have. */

  ck_mem->ck_tretlast = ida_mem->ida_tn;
  ck_mem->ck_kk       = ida_mem->ida_kk;
  ck_mem->ck_kused    = ida_mem->ida_kused;
  ck_mem->ck_knew     = ida_mem->ida_knew;
  ck_mem->ck_phase    = ida_mem->ida_phase;
  ck_mem->ck_ns       = ida_mem->ida_ns;
  ck_mem->ck_hh       = ida_mem->ida_hh;
  ck_mem->ck_hused    = ida_mem->ida_hused;
  ck_mem->ck_eta      = ida_mem->ida_eta;
  ck_mem->ck_cj       = ida_mem->ida_cj;
  ck_mem->ck_cjlast   = ida_mem->ida_cjlast;
  ck_mem->ck_cjold    = ida_mem->ida_cjold;
  ck_mem->ck_cjratio  = ida_mem->ida_cjratio;
  ck_mem->ck_ss       = ida_mem->ida_ss;
  ck_mem->ck_ssS      = ida_mem->ida_ssS;
  ck_mem->ck_t0       = ida_mem->ida_tn;

  for (int j = 0; j < MXORDP1; j++)
  {
    ck_mem->ck_psi[j]   = ida_mem->ida_psi[j];
    ck_mem->ck_alpha[j] = ida_mem->ida_alpha[j];
    ck_mem->ck_beta[j]  = ida_mem->ida_beta[j];
    ck_mem->ck_sigma[j] = ida_mem->ida_sigma[j];
    ck_mem->ck_gamma[j] = ida_mem->ida_gamma[j];
  }

  ck_mem->ck_quadr = ida_mem->ida_quadr && ida_mem->ida_errconQ;

  ck_mem->ck_sensi = ida_mem->ida_sensi;
  if (ck_mem->ck_sensi) { ck_mem->ck_Ns = ida_mem->ida_Ns; }

  ck_mem->ck_quadr_sensi = ida_mem->ida_quadr_sensi && ida_mem->ida_errconQS;

  /* Unlike IDAAckpntInit (hardcoded order 1, alloc 3), size storage for the
     integrator's ACTUAL current order -- this is the whole point. */
  ck_mem->ck_phi_alloc = (ida_mem->ida_kk + 2 < MXORDP1) ? ida_mem->ida_kk + 2
                                                         : MXORDP1;

  if (!DDIDAAckpntAllocVectors(ida_mem, ck_mem))
  {
    free(ck_mem);
    return NULL;
  }

  DDIDAAckpntCopyVectors(ida_mem, ck_mem);

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

DD_DEFINE_DYNARR(ProbB, ProbB)

static DataB* ProbBFind(DynArr_ProbB pbs, int which)
{
  for (size_t i = 0; i < DA_LENGTH(pbs); ++i)
  {
    if (DA_Ith(pbs, i).pb_which == which) { return DA_Ith(pbs, i).pb_data; }
  }
  return NULL;
}

/* --------------------------------------------------------------------------
 * DD Session Memory
 * -------------------------------------------------------------------------- */

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
  DDQuadRhsFn* dd_quad;
  DDLsJacFn1* dd_jacfn1;
  DDLsJacFn2* dd_jacfn2;
  sunrealtype dd_t0;
  N_Vector dd_yy;
  N_Vector dd_yp;
  N_Vector dd_id;
  N_Vector dd_Q;
  void* dd_user_data;

  /* Forward Sensitivities */

  N_Vector* dd_yyS;
  N_Vector* dd_ypS;
  DDSensResFn* dd_resS;

  /* Backwards Problem */

  DynArr_ProbB dd_probBs;
  DDckpntMem ck_mem;
  DDckpntMem ck_mem_cur;
};

/* --------------------------------------------------------------------------
 * Private Function Prototypes
 * -------------------------------------------------------------------------- */

static int DDResFnWrapper(sunrealtype, N_Vector, N_Vector, N_Vector, void*);

static void DDSetYpFromY(DDStaticInfo, DDDAEState, N_Vector, N_Vector);
static void DDSetId(DDStaticInfo, DDDAEState, N_Vector);

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

static int DDResFnSWrapper(int Ns,
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

static void DDSensCleanup(DDMem dd_mem);

static int DDQuadRhsFnWrapper(sunrealtype, N_Vector, N_Vector, N_Vector, void*);

static void DDAdjCleanupCheckpoints(DDMem dd_mem);

static int DDResFnBWrapper(sunrealtype,
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
 * Forward Solution
 * -------------------------------------------------------------------------- */

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
  DDAdjCleanupCheckpoints(dd_mem);
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

  DDDAEState state = DDDAEStateCreate(sunctx, si);
  if (state == NULL)
  {
    DDHandleErr(SUN_ERR_MEM_FAIL);
    return SUN_ERR_MEM_FAIL;
  }
  dd_mem->dd_state = state;

  if (DDDAEStateUpdate(state, spec) < 0)
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

  if (IDAInit(dd_mem->ida_mem, DDResFnWrapper, t0, Y0, dd_mem->dd_yp) < 0)
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

static int DDResFnWrapper(sunrealtype t,
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

int DDReInit(DDMem dd_mem, uint8_t* spec, sunrealtype t0, N_Vector Y0)
{
  if (dd_mem == NULL)
  {
    DDHandleErrWithCtx(DD_ERR_DD_MEM_NULL, NULL);
    return DD_ERR_DD_MEM_NULL;
  }

  SUNFunctionBegin(dd_mem->sunctx);

  SUNAssert(spec != NULL, SUN_ERR_ARG_CORRUPT);
  SUNAssert(Y0 != NULL, SUN_ERR_ARG_CORRUPT);

  DDStaticInfo si  = dd_mem->dd_si;
  DDDAEState state = dd_mem->dd_state;

  memcpy(dd_mem->dd_spec, spec, si->N * sizeof(*dd_mem->dd_spec));

  SUNCheckCall(DDDAEStateUpdate(state, spec));

  DDSetYpFromY(si, state, Y0, dd_mem->dd_yp);
  DDSetId(si, state, dd_mem->dd_id);

  if (IDAReInit(dd_mem->ida_mem, t0, Y0, dd_mem->dd_yp) < 0)
  {
    DDHandleErr(DD_ERR_IDA_ERR);
    return DD_ERR_IDA_ERR;
  }

  if (IDASetId(dd_mem->ida_mem, dd_mem->dd_id) < 0)
  {
    DDHandleErr(DD_ERR_IDA_ERR);
    return DD_ERR_IDA_ERR;
  }

  dd_mem->dd_t0 = t0;

  return SUN_SUCCESS;
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

  if (DDDAEStateUpdate(state, spec) < 0)
  {
    DDHandleErr(SUN_ERR_OP_FAIL);
    return SUN_ERR_OP_FAIL;
  }

  IDAMem ida_mem = dd_mem->ida_mem;

  N_Vector id          = dd_mem->dd_id;
  const sunrealtype tn = ida_mem->ida_tn;

  DDSetId(si, state, id);

  if (IDASetId(ida_mem, id) < 0)
  {
    DDHandleErr(DD_ERR_IDA_ERR);
    return DD_ERR_IDA_ERR;
  }

  /* Since a pivot changes the residual structure we must make sure that
     anything that depends on the Jacobian is re-computed.  */

  if (ida_mem->ida_linit != NULL)
  {
    if (ida_mem->ida_linit(ida_mem) != 0)
    {
      DDHandleErr(DD_ERR_IDA_ERR);
      return DD_ERR_IDA_ERR;
    }
  }

  ida_mem->ida_forceSetup = SUNTRUE;

  if (ida_mem->ida_adj)
  {
    IDAadjMem ida_adj_mem = ida_mem->ida_adj_mem;

    if (ida_adj_mem->ck_mem == NULL)
    {
      /* No IDA checkpoints were accumulated since the last IDAAdjReInit, so
         DDSetSpec is being called again before any DDSolveF. Update the
         current DD checkpoint's state in-place rather than prepending a new
         empty checkpoint. */
      if (DDDAEStateUpdate(dd_mem->ck_mem->ck_state, spec) < 0)
      {
        DDHandleErr(SUN_ERR_OP_FAIL);
        return SUN_ERR_OP_FAIL;
      }
    }
    else
    {
      /* Normal case: IDA checkpoints were accumulated since the last pivot.
         Store them in the current DD checkpoint and prepend a fresh one. */

      DDckpntMem ck_next = dd_mem->ck_mem;
      ck_next->ck_t1     = tn;

      DDckpntMem ck_mem = DDckpntCreate(tn, state);
      if (ck_mem == NULL)
      {
        DDHandleErr(SUN_ERR_MEM_FAIL);
        return DD_ERR_IDA_ERR;
      }

      ck_mem->ck_next = ck_next;
      dd_mem->ck_mem  = ck_mem;

      /* Prepend a new IDA-level checkpoint capturing the integrator's
         ACTUAL current order/step history (DDAIDAckpntNew). */

      IDAckpntMem tmp = DDIDAAckpntNew(ida_mem);
      if (tmp == NULL)
      {
        DDHandleErr(SUN_ERR_MEM_FAIL);
        return DD_ERR_IDA_ERR;
      }

      tmp->ck_next = ida_adj_mem->ck_mem;

      ida_adj_mem->ck_mem = tmp;
      ida_adj_mem->ia_nckpnts++;

      ida_adj_mem->dt_mem[0]->t = tmp->ck_t0;
      ida_adj_mem->ia_storePnt(ida_mem, ida_adj_mem->dt_mem[0]);
    }
  }

  return SUN_SUCCESS;
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
 * DDSetLinearSolver
 * -------------------------------------------------------------------------- */

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

/* --------------------------------------------------------------------------
 * DDJacFn_CSC
 * -------------------------------------------------------------------------- */

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
 * DDSSTolerances
 * -------------------------------------------------------------------------- */

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

/* --------------------------------------------------------------------------
 * DDSetStopTime
 * -------------------------------------------------------------------------- */

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

/* --------------------------------------------------------------------------
 * DDSetUserData
 * -------------------------------------------------------------------------- */

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

/* --------------------------------------------------------------------------
 * DDGetSpec
 * -------------------------------------------------------------------------- */

const uint8_t* DDGetSpec(DDMem dd_mem)
{
  if (dd_mem == NULL)
  {
    DDHandleErrWithCtx(DD_ERR_DD_MEM_NULL, NULL);
    return NULL;
  }

  return dd_mem->dd_spec;
}

/* --------------------------------------------------------------------------
 * DDGetId
 * -------------------------------------------------------------------------- */

N_Vector DDGetId(DDMem dd_mem)
{
  if (dd_mem == NULL)
  {
    DDHandleErrWithCtx(DD_ERR_DD_MEM_NULL, NULL);
    return NULL;
  }

  return dd_mem->dd_id;
}

/* --------------------------------------------------------------------------
 * DDGetConsistentIC
 * -------------------------------------------------------------------------- */

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

/* --------------------------------------------------------------------------
 * Forward Sensitivity
 * -------------------------------------------------------------------------- */

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

  if (IDASensInit(dd_mem->ida_mem, Ns, ism, resfnS ? DDResFnSWrapper : NULL, YS0, ypS) <
      0)
  {
    DDHandleErr(DD_ERR_IDA_ERR);
    return DD_ERR_IDA_ERR;
  }

  dd_mem->dd_resS = resfnS;

  return DD_SUCCESS;
}

static int DDResFnSWrapper(int Ns,
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
 * DDGetSens
 * -------------------------------------------------------------------------- */

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

/* --------------------------------------------------------------------------
 * DDSetSensParams
 * -------------------------------------------------------------------------- */

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

/* --------------------------------------------------------------------------
 * DDSensEEtolerances
 * -------------------------------------------------------------------------- */

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

/* --------------------------------------------------------------------------
 * DDGetSensConsistentIC
 * -------------------------------------------------------------------------- */

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

/* --------------------------------------------------------------------------
 * Forward Quadrature
 * -------------------------------------------------------------------------- */

/* --------------------------------------------------------------------------
 * DDQuadInit
 * -------------------------------------------------------------------------- */

int DDQuadInit(DDMem dd_mem, DDQuadRhsFn rhsQ, N_Vector yQ0)
{
  if (dd_mem == NULL)
  {
    DDHandleErrWithCtx(DD_ERR_DD_MEM_NULL, NULL);
    return DD_ERR_DD_MEM_NULL;
  }

  SUNFunctionBegin(dd_mem->sunctx);

  dd_mem->dd_quad = rhsQ;

  if (IDAQuadInit(dd_mem->ida_mem, DDQuadRhsFnWrapper, yQ0) < 0)
  {
    dd_mem->dd_quad = NULL;
    DDHandleErr(DD_ERR_IDA_ERR);
    return DD_ERR_IDA_ERR;
  }

  dd_mem->dd_Q = N_VClone(yQ0);

  return SUN_SUCCESS;
}

static int DDQuadRhsFnWrapper(sunrealtype t,
                              N_Vector yy,
                              SUNDIALS_MAYBE_UNUSED N_Vector yp,
                              N_Vector rrQ,
                              void* user_data)
{
  const DDMem dd_mem = (DDMem)user_data;
  return dd_mem->dd_quad(t, yy, rrQ, dd_mem->dd_user_data);
}

/* --------------------------------------------------------------------------
 * DDQuadReInit
 * -------------------------------------------------------------------------- */

int DDQuadReInit(DDMem dd_mem, N_Vector yQ0)
{
  if (dd_mem == NULL)
  {
    DDHandleErrWithCtx(DD_ERR_DD_MEM_NULL, NULL);
    return DD_ERR_DD_MEM_NULL;
  }

  SUNFunctionBegin(dd_mem->sunctx);

  if (IDAQuadReInit(dd_mem->ida_mem, yQ0) < 0)
  {
    DDHandleErr(DD_ERR_IDA_ERR);
    return DD_ERR_IDA_ERR;
  }

  return SUN_SUCCESS;
}

/* --------------------------------------------------------------------------
 * DDQuadFree
 * -------------------------------------------------------------------------- */

void DDQuadFree(DDMem dd_mem)
{
  if (dd_mem == NULL) { return; }

  N_VDestroy(dd_mem->dd_Q);
  dd_mem->dd_Q = NULL;

  IDAQuadFree(dd_mem->ida_mem);
}

/* --------------------------------------------------------------------------
 * DDGetQuad
 * -------------------------------------------------------------------------- */

int DDGetQuad(DDMem dd_mem, sunrealtype* tret, N_Vector yQ)
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

  if (yQ == NULL)
  {
    DDHandleErr(SUN_ERR_ARG_CORRUPT);
    return SUN_ERR_ARG_CORRUPT;
  }

  if (IDAGetQuad(dd_mem->ida_mem, tret, yQ) < 0)
  {
    DDHandleErr(DD_ERR_IDA_ERR);
    return DD_ERR_IDA_ERR;
  }

  return DD_SUCCESS;
}

/* --------------------------------------------------------------------------
 * Adjoint
 * -------------------------------------------------------------------------- */

/* --------------------------------------------------------------------------
 * DDSolveF
 * -------------------------------------------------------------------------- */

/* This is a copy of `IDASolveF` from src/idas/idaa.c modified to not assume a
   fixed number of solver steps between each checkpoint. */
static int DDIDASolveF(void* ida_mem,
                       sunrealtype tout,
                       sunrealtype* tret,
                       N_Vector yret,
                       N_Vector ypret,
                       int itask,
                       int* ncheckPtr)
{
  IDAadjMem IDAADJ_mem;
  IDAMem IDA_mem;
  IDAckpntMem tmp;
  IDAdtpntMem* dt_mem;
  long int nstloc;
  int flag, i;
  sunbooleantype allocOK, earlyret;
  sunrealtype ttest;

  /* Is the mem OK? */
  if (ida_mem == NULL)
  {
    IDAProcessError(NULL,
                    IDA_MEM_NULL,
                    __LINE__,
                    __func__,
                    __FILE__,
                    MSGAM_NULL_IDAMEM);
    return (IDA_MEM_NULL);
  }
  IDA_mem = (IDAMem)ida_mem;

  SUNDIALS_MARK_FUNCTION_BEGIN(IDA_PROFILER);

  /* Is ASA initialized ? */
  if (IDA_mem->ida_adjMallocDone == SUNFALSE)
  {
    IDAProcessError(IDA_mem, IDA_NO_ADJ, __LINE__, __func__, __FILE__, MSGAM_NO_ADJ);
    SUNDIALS_MARK_FUNCTION_END(IDA_PROFILER);
    return (IDA_NO_ADJ);
  }
  IDAADJ_mem = IDA_mem->ida_adj_mem;

  /* Check for yret != NULL */
  if (yret == NULL)
  {
    IDAProcessError(IDA_mem,
                    IDA_ILL_INPUT,
                    __LINE__,
                    __func__,
                    __FILE__,
                    MSG_YRET_NULL);
    SUNDIALS_MARK_FUNCTION_END(IDA_PROFILER);
    return (IDA_ILL_INPUT);
  }

  /* Check for ypret != NULL */
  if (ypret == NULL)
  {
    IDAProcessError(IDA_mem,
                    IDA_ILL_INPUT,
                    __LINE__,
                    __func__,
                    __FILE__,
                    MSG_YPRET_NULL);
    SUNDIALS_MARK_FUNCTION_END(IDA_PROFILER);
    return (IDA_ILL_INPUT);
  }
  /* Check for tret != NULL */
  if (tret == NULL)
  {
    IDAProcessError(IDA_mem,
                    IDA_ILL_INPUT,
                    __LINE__,
                    __func__,
                    __FILE__,
                    MSG_TRET_NULL);
    SUNDIALS_MARK_FUNCTION_END(IDA_PROFILER);
    return (IDA_ILL_INPUT);
  }

  /* Check for valid itask */
  if ((itask != IDA_NORMAL) && (itask != IDA_ONE_STEP))
  {
    IDAProcessError(IDA_mem,
                    IDA_ILL_INPUT,
                    __LINE__,
                    __func__,
                    __FILE__,
                    MSG_BAD_ITASK);
    SUNDIALS_MARK_FUNCTION_END(IDA_PROFILER);
    return (IDA_ILL_INPUT);
  }

  /* All memory checks done, proceed ... */

  dt_mem = IDAADJ_mem->dt_mem;

  /* If tstop is enabled, store some info */
  if (IDA_mem->ida_tstopset)
  {
    IDAADJ_mem->ia_tstopIDAFcall = SUNTRUE;
    IDAADJ_mem->ia_tstopIDAF     = IDA_mem->ida_tstop;
  }

  /* On the first step:
   *   - set tinitial
   *   - initialize list of check points
   *   - if needed, initialize the interpolation module
   *   - load dt_mem[0]
   * On subsequent steps, test if taking a new step is necessary.
   */
  if (IDAADJ_mem->ia_firstIDAFcall)
  {
    IDAADJ_mem->ia_tinitial = IDA_mem->ida_tn;
    IDAADJ_mem->ck_mem      = DDIDAAckpntInit(IDA_mem);
    if (IDAADJ_mem->ck_mem == NULL)
    {
      IDAProcessError(IDA_mem,
                      IDA_MEM_FAIL,
                      __LINE__,
                      __func__,
                      __FILE__,
                      MSG_MEM_FAIL);
      SUNDIALS_MARK_FUNCTION_END(IDA_PROFILER);
      return (IDA_MEM_FAIL);
    }

    if (!IDAADJ_mem->ia_mallocDone)
    {
      /* Do we need to store sensitivities? */
      if (!IDA_mem->ida_sensi) { IDAADJ_mem->ia_storeSensi = SUNFALSE; }

      /* Allocate space for interpolation data */
      allocOK = IDAADJ_mem->ia_malloc(IDA_mem);
      if (!allocOK)
      {
        IDAProcessError(IDA_mem,
                        IDA_MEM_FAIL,
                        __LINE__,
                        __func__,
                        __FILE__,
                        MSG_MEM_FAIL);
        SUNDIALS_MARK_FUNCTION_END(IDA_PROFILER);
        return (IDA_MEM_FAIL);
      }

      /* Rename phi and, if needed, phiS for use in interpolation */
      for (i = 0; i < MXORDP1; i++)
      {
        IDAADJ_mem->ia_Y[i] = IDA_mem->ida_phi[i];
      }
      if (IDAADJ_mem->ia_storeSensi)
      {
        for (i = 0; i < MXORDP1; i++)
        {
          IDAADJ_mem->ia_YS[i] = IDA_mem->ida_phiS[i];
        }
      }

      IDAADJ_mem->ia_mallocDone = SUNTRUE;
    }

    dt_mem[0]->t = IDAADJ_mem->ck_mem->ck_t0;
    IDAADJ_mem->ia_storePnt(IDA_mem, dt_mem[0]);

    IDAADJ_mem->ia_firstIDAFcall = SUNFALSE;
  }
  else if (itask == IDA_NORMAL)
  {
    /* When in normal mode, check if tout was passed or if a previous root was
       not reported and return an interpolated solution. No changes to ck_mem
       or dt_mem are needed. */

    /* flag to signal if an early return is needed */
    earlyret = SUNFALSE;

    /* if a root needs to be reported compare tout to troot otherwise compare
       to the rent time tn */
    ttest = (IDAADJ_mem->ia_rootret) ? IDAADJ_mem->ia_troot : IDA_mem->ida_tn;

    if ((ttest - tout) * IDA_mem->ida_hh >= ZERO)
    {
      /* ttest is after tout, interpolate to tout */
      *tret    = tout;
      flag     = IDAGetSolution(IDA_mem, tout, yret, ypret);
      earlyret = SUNTRUE;
    }
    else if (IDAADJ_mem->ia_rootret)
    {
      /* tout is after troot, interpolate to troot */
      *tret = IDAADJ_mem->ia_troot;
      flag  = IDAGetSolution(IDA_mem, IDAADJ_mem->ia_troot, yret, ypret);
      flag  = IDA_ROOT_RETURN;
      IDAADJ_mem->ia_rootret = SUNFALSE;
      earlyret               = SUNTRUE;
    }

    /* return if necessary */
    if (earlyret)
    {
      *ncheckPtr               = IDAADJ_mem->ia_nckpnts;
      IDAADJ_mem->ia_newData   = SUNTRUE;
      IDAADJ_mem->ia_ckpntData = IDAADJ_mem->ck_mem;
      /* Steps since the *current* checkpoint (periodic or pivot-created),
         not `nst % ia_nsteps`: pivots create extra checkpoints at nst
         values that aren't multiples of ia_nsteps, so the ring-buffer
         position must be tracked relative to ck_mem->ck_nst instead of a
         fixed global grid. */
      IDAADJ_mem->ia_np = IDA_mem->ida_nst - IDAADJ_mem->ck_mem->ck_nst + 1;
      SUNDIALS_MARK_FUNCTION_END(IDA_PROFILER);
      return (flag);
    }
  }

  /* Integrate to tout (in IDA_ONE_STEP mode) while loading check points */
  nstloc = 0;
  for (;;)
  {
    /* Check for too many steps */

    if ((IDA_mem->ida_mxstep > 0) && (nstloc >= IDA_mem->ida_mxstep))
    {
      IDAProcessError(IDA_mem,
                      IDA_TOO_MUCH_WORK,
                      __LINE__,
                      __func__,
                      __FILE__,
                      MSG_MAX_STEPS,
                      IDA_mem->ida_tn);
      flag = IDA_TOO_MUCH_WORK;
      break;
    }

    /* Perform one step of the integration */

    flag = IDASolve(IDA_mem, tout, tret, yret, ypret, IDA_ONE_STEP);
    if (flag < 0) { break; }

    nstloc++;

    /* Test if a new check point is needed.

       This is deliberately NOT `nst % ia_nsteps == 0`: that assumes every
       checkpoint sits on a fixed global grid, which only holds if
       checkpoints are ever created periodically. DDSetSpec() also creates
       checkpoints at pivots, at arbitrary nst values -- so "periodic"
       spacing has to be measured relative to whichever checkpoint is
       currently active (ck_mem->ck_nst), not to a global multiple of
       ia_nsteps. This also means a pivot restarts the periodic countdown,
       instead of leaving a short, misaligned remainder until the next
       global-grid boundary. */

    long int ia_nlocal = IDA_mem->ida_nst - IDAADJ_mem->ck_mem->ck_nst;

    if (ia_nlocal == IDAADJ_mem->ia_nsteps)
    {
      IDAADJ_mem->ck_mem->ck_t1 = IDA_mem->ida_tn;

      /* Create a new check point, load it, and append it to the list */
      tmp = DDIDAAckpntNew(IDA_mem);
      if (tmp == NULL)
      {
        flag = IDA_MEM_FAIL;
        break;
      }

      tmp->ck_next       = IDAADJ_mem->ck_mem;
      IDAADJ_mem->ck_mem = tmp;
      IDAADJ_mem->ia_nckpnts++;

      IDA_mem->ida_forceSetup = SUNTRUE;

      /* Reset i=0 and load dt_mem[0] */
      dt_mem[0]->t = IDAADJ_mem->ck_mem->ck_t0;
      IDAADJ_mem->ia_storePnt(IDA_mem, dt_mem[0]);
    }
    else
    {
      /* Load next point in dt_mem, indexed by steps since the current
         checkpoint (see above), not nst % ia_nsteps. */
      dt_mem[ia_nlocal]->t = IDA_mem->ida_tn;
      IDAADJ_mem->ia_storePnt(IDA_mem, dt_mem[ia_nlocal]);
    }

    /* Set t1 field of the current check point structure
       for the case in which there will be no future
       check points */
    IDAADJ_mem->ck_mem->ck_t1 = IDA_mem->ida_tn;

    /* tfinal is now set to tn */
    IDAADJ_mem->ia_tfinal = IDA_mem->ida_tn;

    /* Return if in IDA_ONE_STEP mode */
    if (itask == IDA_ONE_STEP) { break; }

    /* IDA_NORMAL_STEP returns */

    /* Return if tout reached */
    if ((*tret - tout) * IDA_mem->ida_hh >= ZERO)
    {
      /* If this was a root return, save the root time to return later */
      if (flag == IDA_ROOT_RETURN)
      {
        IDAADJ_mem->ia_rootret = SUNTRUE;
        IDAADJ_mem->ia_troot   = *tret;
      }

      /* Get solution value at tout to return now */
      *tret = tout;
      flag  = IDAGetSolution(IDA_mem, tout, yret, ypret);

      /* Reset tretlast in IDA_mem so that IDAGetQuad and IDAGetSens
       * evaluate quadratures and/or sensitivities at the proper time */
      IDA_mem->ida_tretlast = tout;

      break;
    }

    /* Return if tstop or a root was found */
    if ((flag == IDA_TSTOP_RETURN) || (flag == IDA_ROOT_RETURN)) { break; }

  } /* end of for(;;) */

  /* Get ncheck from IDAADJ_mem */
  *ncheckPtr = IDAADJ_mem->ia_nckpnts;

  /* Data is available for the last interval */
  IDAADJ_mem->ia_newData   = SUNTRUE;
  IDAADJ_mem->ia_ckpntData = IDAADJ_mem->ck_mem;
  IDAADJ_mem->ia_np        = IDA_mem->ida_nst - IDAADJ_mem->ck_mem->ck_nst + 1;

  SUNDIALS_MARK_FUNCTION_END(IDA_PROFILER);
  return (flag);
}

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

  int flag = DDIDASolveF(ida_mem, tout, tret, Y, yp, itask, ncheck);
  if (flag < 0)
  {
    DDHandleErr(DD_ERR_IDA_ERR);
    return DD_ERR_IDA_ERR;
  }

  /* Update the current time for the current checkpoint and forward solution. */

  dd_mem->ck_mem->ck_t1 = ida_mem->ida_tn;

  return flag;
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

  dd_mem->ck_mem = DDckpntCreate(dd_mem->dd_t0, dd_mem->dd_state);
  if (dd_mem->ck_mem == NULL)
  {
    DDHandleErr(SUN_ERR_MEM_FAIL);
    return SUN_ERR_MEM_FAIL;
  }

  return DD_SUCCESS;
}

/* --------------------------------------------------------------------------
 * DDAdjFree
 * -------------------------------------------------------------------------- */

static void DDAdjCleanupProblems(DDMem dd_mem)
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
}

static void DDAdjCleanupCheckpoints(DDMem dd_mem)
{
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

  DDAdjCleanupProblems(dd_mem);
  DDAdjCleanupCheckpoints(dd_mem);
  IDAAdjFree(dd_mem->ida_mem);
}

/* --------------------------------------------------------------------------
 * DDAdjReInit
 * -------------------------------------------------------------------------- */

int DDAdjReInit(DDMem dd_mem)
{
  if (dd_mem == NULL)
  {
    DDHandleErrWithCtx(DD_ERR_DD_MEM_NULL, NULL);
    return DD_ERR_GENERIC;
  }

  SUNFunctionBegin(dd_mem->sunctx);

  DDAdjCleanupCheckpoints(dd_mem);

  if (IDAAdjReInit(dd_mem->ida_mem) < 0)
  {
    DDHandleErr(DD_ERR_IDA_ERR);
    return DD_ERR_IDA_ERR;
  }

  dd_mem->ck_mem = DDckpntCreate(dd_mem->dd_t0, dd_mem->dd_state);
  SUNAssert(dd_mem->ck_mem != NULL, SUN_ERR_MEM_FAIL);

  return SUN_SUCCESS;
}

/* --------------------------------------------------------------------------
 * Backwards
 * -------------------------------------------------------------------------- */

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
            int indexB,
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

  ProbB pb = {.pb_which = indexB};

  pb.pb_data = calloc(1, sizeof(*pb.pb_data));
  if (pb.pb_data == NULL)
  {
    DDHandleErr(SUN_ERR_MEM_FAIL);
    return SUN_ERR_MEM_FAIL;
  }

  pb.pb_data->db_resB = resB;

  IDAMem ida_mem = dd_mem->ida_mem;

  if (IDAInitB(ida_mem, indexB, DDResFnBWrapper, tB0, yyB0, ypB0) < 0)
  {
    ProbBDestroy(pb);
    DDHandleErr(DD_ERR_IDA_ERR);
    return DD_ERR_IDA_ERR;
  }

  if (IDASetUserDataB(ida_mem, indexB, pb.pb_data) < 0)
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

  dd_mem->ck_mem_cur = NULL;

  return DD_SUCCESS;
}

static int DDResFnBWrapper(sunrealtype t,
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

/* --------------------------------------------------------------------------
 * DDReInitB
 * -------------------------------------------------------------------------- */

int DDReInitB(DDMem dd_mem, int indexB, sunrealtype tB0, N_Vector yyB0, N_Vector ypB0)
{
  if (dd_mem == NULL)
  {
    DDHandleErrWithCtx(DD_ERR_DD_MEM_NULL, NULL);
    return DD_ERR_GENERIC;
  }

  SUNFunctionBegin(dd_mem->sunctx);

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

  IDAMem ida_mem = dd_mem->ida_mem;

  if (IDAReInitB(ida_mem, indexB, tB0, yyB0, ypB0) < 0)
  {
    DDHandleErr(DD_ERR_IDA_ERR);
    return DD_ERR_IDA_ERR;
  }

  dd_mem->ck_mem_cur = NULL;

  return DD_SUCCESS;
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

  int sign = (ida_adj_mem->ia_tfinal - ida_adj_mem->ia_tinitial > ZERO) ? 1 : -1;

  if (sign * (tBout - ida_adj_mem->ia_tinitial) < ZERO)
  {
    DDHandleErr(SUN_ERR_ARG_OUTOFRANGE);
    return SUN_ERR_ARG_OUTOFRANGE;
  }

  if ((itaskB != IDA_ONE_STEP) && (itaskB != IDA_NORMAL))
  {
    DDHandleErr(SUN_ERR_ARG_OUTOFRANGE);
    return SUN_ERR_ARG_OUTOFRANGE;
  }

  /* Starting from the right-most checkpoint, loop through checkpoints until
     the current time of any of the backwards problems comes after the start
     time of the checkpoint (in the forward direction).  */

  /* This code is code from `IDASolveB` re-purposed to find the first relevant
     DD checkpoint. */

  DDckpntMem ck_mem = NULL;

  for (ck_mem = dd_mem->ck_mem; ck_mem != NULL; ck_mem = ck_mem->ck_next)
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
      DDDAEStateCopy(ck_mem->ck_state, dd_mem->dd_state);
      dd_mem->ck_mem_cur = ck_mem;

      /* Since a pivot changes the residual structure we must make sure that
         anything that depends on the Jacobian is re-computed.  */

      if (ida_mem->ida_linit != NULL)
      {
        if (ida_mem->ida_linit(ida_mem) != 0)
        {
          DDHandleErr(DD_ERR_IDA_ERR);
          return DD_ERR_IDA_ERR;
        }
      }

      ida_mem->ida_forceSetup = SUNTRUE;
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
 * DDSetLinearSolverB
 * -------------------------------------------------------------------------- */

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

/* --------------------------------------------------------------------------
 * DDSetJacFnB
 * -------------------------------------------------------------------------- */

int DDSetJacFnB(DDMem dd_mem, int indexB, DDLsJacFnB jacfn)
{
  if (dd_mem == NULL)
  {
    DDHandleErrWithCtx(DD_ERR_DD_MEM_NULL, NULL);
    return DD_ERR_GENERIC;
  }

  SUNFunctionBegin(dd_mem->sunctx);

  SUNAssert(jacfn != NULL, SUN_ERR_ARG_CORRUPT);

  DataB* db = ProbBFind(dd_mem->dd_probBs, indexB);
  SUNAssert(db != NULL, SUN_ERR_ARG_OUTOFRANGE);

  db->db_jacB = jacfn;

  if (IDASetJacFnB(dd_mem->ida_mem, indexB, DDLsJacFnBWrapper) < 0)
  {
    db->db_jacB = NULL;
    DDHandleErr(DD_ERR_IDA_ERR);
    return DD_ERR_IDA_ERR;
  }

  return SUN_SUCCESS;
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

/* --------------------------------------------------------------------------
 * DDSStolerancesB
 * -------------------------------------------------------------------------- */

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

/* --------------------------------------------------------------------------
 * DDSetUserDataB
 * -------------------------------------------------------------------------- */

int DDSetUserDataB(DDMem dd_mem, int indexB, void* user_dataB)
{
  if (dd_mem == NULL)
  {
    DDHandleErrWithCtx(DD_ERR_DD_MEM_NULL, NULL);
    return DD_ERR_GENERIC;
  }

  SUNFunctionBegin(dd_mem->sunctx);

  DataB* db = ProbBFind(dd_mem->dd_probBs, indexB);
  SUNAssert(db != NULL, SUN_ERR_ARG_OUTOFRANGE);

  db->db_user_data = user_dataB;

  return SUN_SUCCESS;
}

/* --------------------------------------------------------------------------
 * DDSetIdB
 * -------------------------------------------------------------------------- */

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

/* --------------------------------------------------------------------------
 * DDGetB
 * -------------------------------------------------------------------------- */

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

/* --------------------------------------------------------------------------
 * DDGetConsistentICB
 * -------------------------------------------------------------------------- */

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

/* --------------------------------------------------------------------------
 * Backwards Quadrature
 * -------------------------------------------------------------------------- */

/* --------------------------------------------------------------------------
 * DDQuadInitB
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

  DataB* db = ProbBFind(dd_mem->dd_probBs, indexB);
  SUNAssert(db != NULL, SUN_ERR_ARG_OUTOFRANGE);

  db->db_quadB = rhsQB;

  if (IDAQuadInitB(dd_mem->ida_mem, indexB, DDQuadRhsFnBWrapper, yQB0) < 0)
  {
    db->db_quadB = NULL;
    DDHandleErr(DD_ERR_IDA_ERR);
    return DD_ERR_IDA_ERR;
  }

  return SUN_SUCCESS;
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
 * DDQuadReInitB
 * -------------------------------------------------------------------------- */

int DDQuadReInitB(DDMem dd_mem, int indexB, N_Vector yQB0)
{
  if (dd_mem == NULL)
  {
    DDHandleErrWithCtx(DD_ERR_DD_MEM_NULL, NULL);
    return DD_ERR_DD_MEM_NULL;
  }

  SUNFunctionBegin(dd_mem->sunctx);

  SUNAssert(yQB0 != NULL, SUN_ERR_ARG_CORRUPT);

  if (IDAQuadReInitB(dd_mem->ida_mem, indexB, yQB0) < 0)
  {
    DDHandleErr(DD_ERR_IDA_ERR);
    return DD_ERR_IDA_ERR;
  }

  return SUN_SUCCESS;
}

/* --------------------------------------------------------------------------
 * DDGetQuadB
 * -------------------------------------------------------------------------- */

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
 * General Setters and Getters
 * -------------------------------------------------------------------------- */

/* --------------------------------------------------------------------------
 * DDGetIDAMem
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

/* --------------------------------------------------------------------------
 * DDGetReturnFlagName
 * -------------------------------------------------------------------------- */

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
