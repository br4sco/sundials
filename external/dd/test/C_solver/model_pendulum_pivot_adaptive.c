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

/*
 * Experiment driver: no pivot-check schedule at all. DDSPPivotStaged is only
 * called reactively: every output step, it checks whether IDA's failure
 * counters (Error test fails / NLS step fails / NLS fails) grew since the
 * last check. If so, it calls DDSPPivotStaged right away, on the theory that a
 * growing failure count means the solver is fighting a stale (pre-pivot)
 * DAE structure. If none of the counters grew, no check happens at all
 * that step. Monitoring only happens once per tstep (0.1), since DDSolve
 * advances directly to the next output time under IDA_NORMAL -- trouble
 * that appears and resolves within one 0.1 window is invisible to this
 * heuristic.
 */

/*
 * Wraps DDSPStaged() with the reactive trigger policy above: only
 * re-pivots when one of IDA's failure counters has grown since the last
 * check, and logs check/pivot stats to stats_file.
 */
typedef struct
{
  DDStatePivot inner;
  void* ida_mem;
  FILE* stats_file;
  long int prev_netf, prev_ncfn, prev_nnf;
  int pivot_num, check_num, piv_call_count;
  int checks_by_netf, checks_by_ncfn, checks_by_nnf;
  int pivots_by_netf, pivots_by_ncfn, pivots_by_nnf;
} AdaptiveSpecContent;

static int AdaptiveSpecUpdate(DDStatePivot self,
                              sunrealtype t,
                              N_Vector Y,
                              void* user_data,
                              uint8_t* spec,
                              sunbooleantype* spec_changed)
{
  AdaptiveSpecContent* c = (AdaptiveSpecContent*)self->content;
  void* ida_mem          = c->ida_mem;

  *spec_changed = SUNFALSE;

  long int netf, ncfn, nnf;
  if (IDAGetNumErrTestFails(ida_mem, &netf) < 0) { return -1; }
  if (IDAGetNumStepSolveFails(ida_mem, &ncfn) < 0) { return -1; }
  if (IDAGetNumNonlinSolvConvFails(ida_mem, &nnf) < 0) { return -1; }

  sunbooleantype trig_netf = netf > c->prev_netf;
  sunbooleantype trig_ncfn = ncfn > c->prev_ncfn;
  sunbooleantype trig_nnf  = nnf > c->prev_nnf;
  sunbooleantype trouble   = trig_netf || trig_ncfn || trig_nnf;
  c->prev_netf             = netf;
  c->prev_ncfn             = ncfn;
  c->prev_nnf              = nnf;

  if (!trouble) { return 0; }

  c->check_num++;
  c->piv_call_count++;
  c->checks_by_netf += trig_netf ? 1 : 0;
  c->checks_by_ncfn += trig_ncfn ? 1 : 0;
  c->checks_by_nnf += trig_nnf ? 1 : 0;

  if (DDSPUpdate(c->inner, t, Y, user_data, spec, spec_changed) < 0)
  {
    return -1;
  }

  fprintf(c->stats_file,
          "# check %d at t=%.6f (trigger: netf=%d ncfn=%d nnf=%d, "
          "spec_changed=%d)\n",
          c->check_num,
          (double)t,
          trig_netf ? 1 : 0,
          trig_ncfn ? 1 : 0,
          trig_nnf ? 1 : 0,
          *spec_changed ? 1 : 0);

  if (*spec_changed)
  {
    c->pivot_num++;
    c->pivots_by_netf += trig_netf ? 1 : 0;
    c->pivots_by_ncfn += trig_ncfn ? 1 : 0;
    c->pivots_by_nnf += trig_nnf ? 1 : 0;
    fprintf(c->stats_file, "# pivot %d at t=%.6f\n", c->pivot_num, (double)t);
    if (IDAPrintAllStats(ida_mem, c->stats_file, SUN_OUTPUTFORMAT_TABLE) < 0)
    {
      return -1;
    }
  }

  return 0;
}

static void AdaptiveSpecDestroy(DDStatePivot self)
{
  AdaptiveSpecContent* c = (AdaptiveSpecContent*)self->content;

  fprintf(c->stats_file, "# total DDSPPivotStaged calls: %d\n", c->piv_call_count);
  fprintf(c->stats_file, "# total pivots: %d\n", c->pivot_num);
  fprintf(c->stats_file, "# trigger breakdown (checks -> pivots, precision):\n");
  fprintf(c->stats_file,
          "#   netf: %d -> %d (%.1f%%)\n",
          c->checks_by_netf,
          c->pivots_by_netf,
          c->checks_by_netf > 0
            ? 100.0 * (double)c->pivots_by_netf / (double)c->checks_by_netf
            : 0.0);
  fprintf(c->stats_file,
          "#   ncfn: %d -> %d (%.1f%%)\n",
          c->checks_by_ncfn,
          c->pivots_by_ncfn,
          c->checks_by_ncfn > 0
            ? 100.0 * (double)c->pivots_by_ncfn / (double)c->checks_by_ncfn
            : 0.0);
  fprintf(c->stats_file,
          "#   nnf : %d -> %d (%.1f%%)\n",
          c->checks_by_nnf,
          c->pivots_by_nnf,
          c->checks_by_nnf > 0
            ? 100.0 * (double)c->pivots_by_nnf / (double)c->checks_by_nnf
            : 0.0);

  DDSPDestroy(c->inner);
  free(c);
  free(self);
}

static DDStatePivot AdaptiveSpecCreate(DDStatePivot inner,
                                       void* ida_mem,
                                       FILE* stats_file)
{
  AdaptiveSpecContent* content = malloc(sizeof(*content));
  if (content == NULL) { return NULL; }

  content->inner   = inner;
  content->ida_mem = ida_mem;

  content->stats_file     = stats_file;
  content->prev_netf      = 0;
  content->prev_ncfn      = 0;
  content->prev_nnf       = 0;
  content->pivot_num      = 0;
  content->check_num      = 0;
  content->piv_call_count = 0;
  content->checks_by_netf = 0;
  content->checks_by_ncfn = 0;
  content->checks_by_nnf  = 0;
  content->pivots_by_netf = 0;
  content->pivots_by_ncfn = 0;
  content->pivots_by_nnf  = 0;

  DDStatePivot sp = DDSPNewEmpty();
  if (sp == NULL)
  {
    DDSPDestroy(content->inner);
    free(content);
    return NULL;
  }

  sp->content      = content;
  sp->ops->update  = AdaptiveSpecUpdate;
  sp->ops->destroy = AdaptiveSpecDestroy;

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

  TEST_ASSERT(argc > 1);
  switch (argv[1][0])
  {
  case NONE: mat_type = NONE; break;
  case DENSE: mat_type = DENSE; break;
  case CSR: mat_type = CSR; break;
  case CSC: mat_type = CSC; break;
  default: TEST_ASSERT(0);
  }

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
  sprintf(filename, "pendulum_%c_adaptive.csv", mat_type);
  FILE* file = fopen(filename, "w");
  TEST_ASSERT(file);

  /* Set up pivot-stats log file. */
  char stats_filename[64];
  sprintf(stats_filename, "pendulum_stats_%c_adaptive.log", mat_type);
  FILE* stats_file = fopen(stats_filename, "w");
  TEST_ASSERT(stats_file);

  /* Solve and output solution. */
  fprintf(file, "t,x,y,λ,ΔL\n"); /* print header */

  void* ida_mem = DDGetIDAMem(dd_mem);

  DDStatePivot sp = AdaptiveSpecCreate(inner, ida_mem, stats_file);
  TEST_ASSERT(sp != NULL);
  TEST_ASSERT(DDSetStatePivot(dd_mem, sp) == SUN_SUCCESS);

  int flag      = IDA_SUCCESS;
  sunrealtype t = t0;
  sunrealtype tret;

  while (flag != IDA_TSTOP_RETURN)
  {
    const sunrealtype x = P_Ith(Y, 0, 0), y = P_Ith(Y, 1, 0),
                      lam = P_Ith(Y, 2, 0);

    fprintf(file,
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
