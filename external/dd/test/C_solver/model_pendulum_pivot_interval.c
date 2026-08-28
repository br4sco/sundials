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
 * Experiment driver: decouples the pivot-check cadence from the solution
 * output cadence so IDAPrintAllStats can be captured at each successful
 * pivot (a DDSPPivotStaged call where spec_changed == SUNTRUE) as the time between
 * pivot checks (dt_check) is varied. Solution output still happens every
 * tstep, but DDSPPivotStaged is only invoked once t has reached the next scheduled
 * check time. dt_check should be >= tstep since checks can only occur on
 * output-step boundaries.
 */

/*
 * Wraps DDSPStaged() with a fixed-interval trigger policy: only
 * re-pivots once t has reached the next scheduled check time (t_next_check),
 * advancing it by dt_check after each check.
 */
typedef struct
{
  DDStatePivot inner;
  void* ida_mem;
  FILE* stats_file;
  sunrealtype dt_check;
  sunrealtype t_next_check;
  int pivot_num;
  int piv_call_count;
} IntervalSpecContent;

static int IntervalSpecUpdate(DDStatePivot self,
                              sunrealtype t,
                              N_Vector Y,
                              void* user_data,
                              uint8_t* spec,
                              sunbooleantype* spec_changed)
{
  IntervalSpecContent* c = (IntervalSpecContent*)self->content;
  void* ida_mem          = c->ida_mem;

  *spec_changed = SUNFALSE;

  if (t < c->t_next_check) { return 0; }

  c->piv_call_count++;
  c->t_next_check += c->dt_check;

  if (DDSPUpdate(c->inner, t, Y, user_data, spec, spec_changed) < 0)
  {
    return -1;
  }

  if (*spec_changed)
  {
    c->pivot_num++;
    DD_LOG(c->stats_file,
           "# pivot %d at t=%.6f (dt_check=%.6f)\n",
           c->pivot_num,
           (double)t,
           (double)c->dt_check);
    if (DD_PRINT_ALL_STATS(ida_mem, c->stats_file, SUN_OUTPUTFORMAT_TABLE) !=
        IDA_SUCCESS)
    {
      return -1;
    }
  }

  return 0;
}

static void IntervalSpecDestroy(DDStatePivot self)
{
  IntervalSpecContent* c = (IntervalSpecContent*)self->content;

  DD_LOG(c->stats_file, "# total DDSPPivotStaged calls: %d\n", c->piv_call_count);

  DDSPDestroy(c->inner);
  free(c);
  free(self);
}

static DDStatePivot IntervalSpecCreate(DDStatePivot inner,
                                       void* ida_mem,
                                       FILE* stats_file,
                                       sunrealtype t0,
                                       sunrealtype dt_check)
{
  IntervalSpecContent* content = malloc(sizeof(*content));
  if (content == NULL) { return NULL; }

  content->inner   = inner;
  content->ida_mem = ida_mem;

  content->stats_file     = stats_file;
  content->dt_check       = dt_check;
  content->t_next_check   = t0;
  content->pivot_num      = 0;
  content->piv_call_count = 0;

  DDStatePivot sp = DDSPNewEmpty();
  if (sp == NULL)
  {
    DDSPDestroy(content->inner);
    free(content);
    return NULL;
  }

  sp->content      = content;
  sp->ops->update  = IntervalSpecUpdate;
  sp->ops->destroy = IntervalSpecDestroy;

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

  const sunrealtype dt_check = (sunrealtype)strtod(argv[2], NULL);
  TEST_ASSERT(dt_check > ZERO);

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

  void* ida_mem = DDGetIDAMem(dd_mem);

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
  sprintf(filename, "pendulum_%c_dt%.3f.csv", mat_type, (double)dt_check);
  FILE* file = fopen(filename, "w");
  TEST_ASSERT(file);

  /* Set up pivot-stats log file, named after matrix type and check interval
   * so sweep runs (varying dt_check) don't clobber each other. */
  char stats_filename[64];
  sprintf(stats_filename,
          "pendulum_stats_%c_dt%.3f.log",
          mat_type,
          (double)dt_check);
  FILE* stats_file = fopen(stats_filename, "w");
  TEST_ASSERT(stats_file);

  DDStatePivot sp = IntervalSpecCreate(inner, ida_mem, stats_file, t0, dt_check);
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
