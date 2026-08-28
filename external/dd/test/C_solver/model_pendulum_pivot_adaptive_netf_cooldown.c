#include <idas/idas.h>
#include <nvector/nvector_serial.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <sundials/sundials_core.h>
#include <sundials/sundials_matrix.h>
#include <sundials/sundials_nvector.h>
#include <sunlinsol/sunlinsol_dense.h>
#include <sunlinsol/sunlinsol_klu.h>
#include <sunmatrix/sunmatrix_dense.h>
#include <sunmatrix/sunmatrix_sparse.h>

#include "dd.h"
#include "dd_math.h"
#include "dd_staged_pivot.h"
#include "dd_staged_pivot_matrix.h"
#include "models.h"
#include "static_info.h"
#include "sundials/sundials_types.h"
#include "test.h"

#define ZERO SUN_RCONST(0.0)
#define ONE  SUN_RCONST(1.0)
#define TWO  SUN_RCONST(2.0)
#define FOUR SUN_RCONST(4.0)
#define FIVE SUN_RCONST(5.0)

/* All logging goes through these so a DD_BENCH build (see the *_bench
 * CMake targets) can compile it out entirely, isolating the cost of the
 * pivoting logic itself from file I/O for hyperfine benchmarking. */
#ifdef DD_BENCH
#define DD_LOG(...)                 ((void)0)
#define DD_PRINT_ALL_STATS(m, f, t) IDA_SUCCESS
#else
#define DD_LOG(...)                 fprintf(__VA_ARGS__)
#define DD_PRINT_ALL_STATS(m, f, t) IDAPrintAllStats(m, f, t)
#endif

/*
 * Experiment driver: no pivot-check schedule at all. DDSPPivotStaged is only
 * called reactively: every output step, it checks whether IDA's error test
 * fail counter (netf) grew since the last check. If so, it calls DDSPPivotStaged
 * right away, on the theory that a growing failure count means the solver
 * is fighting a stale (pre-pivot) DAE structure. If netf didn't grow, no
 * check happens at all that step.
 *
 * This is a variant of model_pendulum_pivot_adaptive_netf.c that adds a
 * cooldown: after a successful pivot (spec_changed == true), no further
 * check happens until dt time units later, even if netf keeps growing in
 * the meantime. The idea is that netf often keeps ticking up for a few
 * steps right after a real pivot while the solver settles into the new
 * spec, producing checks that are very unlikely to find another spec
 * change -- false positives this cooldown is meant to suppress. netf is
 * still sampled every step during the cooldown (so growth isn't silently
 * missed once the cooldown ends), only the DDSPPivotStaged call itself is
 * skipped. Monitoring only happens once per tstep (0.1), since DDSolve
 * advances directly to the next output time under IDA_NORMAL.
 */

/*
 * Wraps DDSPStaged() with a netf-reactive trigger policy plus a
 * cooldown: after a successful pivot, no further check happens until dt
 * time units later, even if netf keeps growing in the meantime.
 */
typedef struct
{
  DDStatePivot inner;
  void* ida_mem;
  FILE* stats_file;
  sunrealtype dt;
  sunrealtype cooldown_until;
  long int prev_netf;
  int pivot_num, check_num, piv_call_count, suppressed_count;
} NetfCooldownSpecContent;

static int NetfCooldownSpecUpdate(DDStatePivot self,
                                  sunrealtype t,
                                  N_Vector Y,
                                  void* user_data,
                                  uint8_t* spec,
                                  sunbooleantype* spec_changed)
{
  NetfCooldownSpecContent* c = (NetfCooldownSpecContent*)self->content;
  void* ida_mem              = c->ida_mem;

  *spec_changed = SUNFALSE;

  long int netf;
  if (IDAGetNumErrTestFails(ida_mem, &netf) < 0) { return -1; }
  sunbooleantype trouble = netf > c->prev_netf;
  c->prev_netf           = netf;

  if (trouble && t < c->cooldown_until)
  {
    c->suppressed_count++;
    return 0;
  }

  if (!trouble) { return 0; }

  c->check_num++;
  c->piv_call_count++;

  if (DDSPUpdate(c->inner, t, Y, user_data, spec, spec_changed) < 0)
  {
    return -1;
  }

  DD_LOG(c->stats_file,
         "# check %d at t=%.6f (netf=%ld, spec_changed=%d)\n",
         c->check_num,
         (double)t,
         netf,
         *spec_changed ? 1 : 0);

  if (*spec_changed)
  {
    c->pivot_num++;
    c->cooldown_until = t + c->dt;
    DD_LOG(c->stats_file, "# pivot %d at t=%.6f\n", c->pivot_num, (double)t);
    if (DD_PRINT_ALL_STATS(ida_mem, c->stats_file, SUN_OUTPUTFORMAT_TABLE) !=
        IDA_SUCCESS)
    {
      return -1;
    }
  }

  return 0;
}

static void NetfCooldownSpecDestroy(DDStatePivot self)
{
  NetfCooldownSpecContent* c = (NetfCooldownSpecContent*)self->content;

  DD_LOG(c->stats_file, "# total DDSPPivotStaged calls: %d\n", c->piv_call_count);
  DD_LOG(c->stats_file, "# total pivots: %d\n", c->pivot_num);
  DD_LOG(c->stats_file,
         "# checks suppressed by cooldown: %d\n",
         c->suppressed_count);
  DD_LOG(c->stats_file,
         "# precision (pivots / checks): %.1f%%\n",
         c->piv_call_count > 0
           ? 100.0 * (double)c->pivot_num / (double)c->piv_call_count
           : 0.0);

  DDSPDestroy(c->inner);
  free(c);
  free(self);
}

static DDStatePivot NetfCooldownSpecCreate(DDStatePivot inner,
                                           void* ida_mem,
                                           FILE* stats_file,
                                           sunrealtype t0,
                                           sunrealtype dt)
{
  NetfCooldownSpecContent* content = malloc(sizeof(*content));
  if (content == NULL) { return NULL; }

  content->inner   = inner;
  content->ida_mem = ida_mem;

  content->stats_file       = stats_file;
  content->dt               = dt;
  content->cooldown_until   = t0;
  content->prev_netf        = 0;
  content->pivot_num        = 0;
  content->check_num        = 0;
  content->piv_call_count   = 0;
  content->suppressed_count = 0;

  DDStatePivot sp = DDSPNewEmpty();
  if (sp == NULL)
  {
    DDSPDestroy(content->inner);
    free(content);
    return NULL;
  }

  sp->content      = content;
  sp->ops->update  = NetfCooldownSpecUpdate;
  sp->ops->destroy = NetfCooldownSpecDestroy;

  return sp;
}

int main(int argc, char* argv[])
{
  enum
  {
    NONE  = 'n',
    DENSE = 'd',
    CSR   = 'r',
    CSC   = 'c'
  } mat_type;

  TEST_ASSERT(argc > 2);
  switch (argv[1][0])
  {
  case NONE: mat_type = NONE; break;
  case DENSE: mat_type = DENSE; break;
  case CSR: mat_type = CSR; break;
  case CSC: mat_type = CSC; break;
  default: TEST_ASSERT(0);
  }

  const sunrealtype dt = (sunrealtype)strtod(argv[2], NULL);
  TEST_ASSERT(dt >= ZERO);

  const sunrealtype t0    = ZERO;
  const sunrealtype tstep = SUN_RCONST(0.1);
  const sunrealtype tout  = SUN_RCONST(100.0);

  /* Setup Sundials context. */
  SUNContext sunctx;
  TEST_ASSERT(SUNContext_Create(SUN_COMM_NULL, &sunctx) == SUN_SUCCESS);

  /* Compute DAE structure. */
  DDStaticInfo si = DDStaticInfoCreate(PENDULUM_N,
                                       PENDULUM_C,
                                       PENDULUM_D,
                                       PENDULUM_VAR_IDX_MAP,
                                       PENDULUM_EQN_NAMES,
                                       PENDULUM_VAR_NAMES);

  TEST_ASSERT(si);

  const sunindextype N = si->N_all_orders;

  /* Allocate state and Jacobian data. */
  SUNMatrix J0 = SUNDenseMatrix(si->N, si->N, sunctx);
  TEST_ASSERT(J0);
  DDStagedPivotMatrix pJ0 = DDStagedPivotMatWrapDense(J0);
  TEST_ASSERT(pJ0);
  N_Vector Y = N_VNew_Serial(N, sunctx);
  TEST_ASSERT(Y);

  /* Set DAE parameters. */
  const sunrealtype m = SUN_RCONST(1.1), l = SUN_RCONST(1.2), g = SUN_RCONST(1.3);
  PendulumData* data = malloc(sizeof(*data));
  data->m            = m;
  data->param[0]     = l;
  data->param[1]     = g;

  /* Set initial values. */
  sunrealtype theta0 = SUN_RCONST(PI) / FIVE + SUN_RCONST(PI) / TWO;
  PendulumY0(data, theta0, Y);

  /* Create pivot policy and compute initial spec. */
  DDStatePivot inner = DDSPStaged(sunctx, si, pJ0, PendulumJacf0, ZERO);
  TEST_ASSERT(inner != NULL);

  uint8_t spec[si->N];
  sunbooleantype spec_changed;
  TEST_ASSERT(DDSPUpdate(inner, t0, Y, data, spec, &spec_changed) >= 0);

  /* Create solver session. */
  DDMem dd_mem = DDCreate(sunctx);
  TEST_ASSERT(dd_mem);

  TEST_ASSERT(DDInit(dd_mem, si, PendulumRes, spec, t0, Y) == IDA_SUCCESS);

  TEST_ASSERT(DDSetUserData(dd_mem, data) == IDA_SUCCESS);

  TEST_ASSERT(DDSSTolerances(dd_mem, SUN_RCONST(1.0e-9), SUN_RCONST(1.0e-9)) ==
              IDA_SUCCESS);

  /* Setup and set linear solver. */
  SUNMatrix J        = NULL;
  SUNLinearSolver LS = NULL;

  switch (mat_type)
  {
  case NONE:
  case DENSE:
    J = SUNDenseMatrix(N, N, sunctx);
    TEST_ASSERT(J);
    LS = SUNLinSol_Dense(Y, J, sunctx);
    TEST_ASSERT(LS);
    break;
  case CSR:
  case CSC:
    J = SUNSparseMatrix(N,
                        N,
                        PENDULUM_JAC_NNZ + 0 + 4,
                        mat_type == CSR ? CSR_MAT : CSC_MAT,
                        sunctx);
    TEST_ASSERT(J);
    LS = SUNLinSol_KLU(Y, J, sunctx);
    TEST_ASSERT(LS);
    break;
  }

  TEST_ASSERT(DDSetLinearSolver(dd_mem, LS, J) == IDA_SUCCESS);

  switch (mat_type)
  {
  case NONE: break;
  case DENSE:
    TEST_ASSERT(
      DDSetJacFn(dd_mem,
                 (DDLsJacFn){.id = DD_JAC_1, .fn.jacfn1 = PendulumJacfn_Dense}) ==
      IDA_SUCCESS);
    break;
  case CSR:
    TEST_ASSERT(
      DDSetJacFn(dd_mem,
                 (DDLsJacFn){.id = DD_JAC_1, .fn.jacfn1 = PendulumJacfn_CSR}) ==
      IDA_SUCCESS);
    break;
  case CSC:
    TEST_ASSERT(
      DDSetJacFn(dd_mem,
                 (DDLsJacFn){.id = DD_JAC_2, .fn.jacfn2 = PendulumJacfn_CSC}) ==
      IDA_SUCCESS);
    break;
  }

  /* Set stop time */
  TEST_ASSERT(DDSetStopTime(dd_mem, tout) == IDA_SUCCESS);

  /* Set up result file */
  char filename[64];
  sprintf(filename,
          "pendulum_%c_adaptive_netf_cooldown_dt%.3f.csv",
          mat_type,
          (double)dt);
  FILE* file = fopen(filename, "w");
  TEST_ASSERT(file);

  /* Set up pivot-stats log file. */
  char stats_filename[64];
  sprintf(stats_filename,
          "pendulum_stats_%c_adaptive_netf_cooldown_dt%.3f.log",
          mat_type,
          (double)dt);
  FILE* stats_file = fopen(stats_filename, "w");
  TEST_ASSERT(stats_file);

  void* ida_mem = DDGetIDAMem(dd_mem);

  DDStatePivot sp = NetfCooldownSpecCreate(inner, ida_mem, stats_file, t0, dt);
  TEST_ASSERT(sp != NULL);
  TEST_ASSERT(DDSetStatePivot(dd_mem, sp) == SUN_SUCCESS);

  /* Solve and output solution. */
  DD_LOG(file, "t,x,y,λ,ΔL\n"); /* print header */

  int flag      = IDA_SUCCESS;
  sunrealtype t = t0;
  sunrealtype tret;

  while (flag != IDA_TSTOP_RETURN)
  {
    const sunrealtype x = P_Ith(Y, 0, 0), y = P_Ith(Y, 1, 0),
                      lam = P_Ith(Y, 2, 0);

    DD_LOG(file,
           "%.20f,%.20f,%.20f,%.20f,%.20f\n",
           t,
           x,
           y,
           lam,
           x * x + y * y - l * l);

    t += tstep;

    flag = DDSolve(dd_mem, t, &tret, Y, IDA_NORMAL);
    TEST_ASSERT(flag >= 0);
  }

  /* Cleanup */
  DDFree(&dd_mem);
  DDSPDestroy(sp);
  DDStagedPivotMatDestroy(pJ0);
  N_VDestroy(Y);
  DDStaticInfoDestroy(si);
  SUNContext_Free(&sunctx);
  SUNLinSolFree(LS);
  SUNMatDestroy(J);
  SUNMatDestroy(J0);
  fclose(file);
  fclose(stats_file);
  free(data);

  return EXIT_SUCCESS;
}
