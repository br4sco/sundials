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
#include "matrix.h"
#include "models.h"
#include "pivot.h"
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
 * Experiment driver: no pivot-check schedule at all. PIVPivot is only
 * called reactively: every output step, it checks whether IDA's error test
 * fail counter (netf) grew since the last check. If so, it calls PIVPivot
 * right away, on the theory that a growing failure count means the solver
 * is fighting a stale (pre-pivot) DAE structure. If netf didn't grow, no
 * check happens at all that step.
 *
 * This is a trimmed-down variant of model_pendulum_pivot_adaptive.c: an
 * earlier run instrumenting all three IDA failure counters (Error test
 * fails / NLS step fails / NLS fails) found netf alone accounted for 45 of
 * 46 successful pivots, with NLS step fails never firing and NLS fails
 * firing only 4 times -- so this driver drops those two entirely and
 * tracks only netf. Monitoring only happens once per tstep (0.1), since
 * DDSolve advances directly to the next output time under IDA_NORMAL --
 * trouble that appears and resolves within one 0.1 window is invisible to
 * this heuristic.
 */
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
  PIVMatrix pJ0 = PIVMatWrapDense(J0);
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

  /* Create pivot memory and compute initial spec. */
  PIVMem pm = PIVCreate(sunctx, si, pJ0, PendulumJacf0);
  TEST_ASSERT(pm);
  TEST_ASSERT(PIVSetUserData(pm, data) == SUN_SUCCESS);
  sunbooleantype spec_changed;
  TEST_ASSERT(PIVPivot(pm, ZERO, t0, Y, &spec_changed) == SUN_SUCCESS);

  /* Create solver session. */
  DDMem dd_mem = DDCreate(sunctx);
  TEST_ASSERT(dd_mem);

  TEST_ASSERT(DDInit(dd_mem, si, PendulumRes, pm->spec, t0, Y) == IDA_SUCCESS);

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
  sprintf(filename, "pendulum_%c_adaptive_netf.csv", mat_type);
  FILE* file = fopen(filename, "w");
  TEST_ASSERT(file);

  /* Set up pivot-stats log file. */
  char stats_filename[64];
  sprintf(stats_filename, "pendulum_stats_%c_adaptive_netf.log", mat_type);
  FILE* stats_file = fopen(stats_filename, "w");
  TEST_ASSERT(stats_file);

  /* Solve and output solution. */
  DD_LOG(file, "t,x,y,λ,ΔL,p\n"); /* print header */

  void* ida_mem = DDGetIDAMem(dd_mem);

  long int prev_netf;
  TEST_ASSERT(IDAGetNumErrTestFails(ida_mem, &prev_netf) == IDA_SUCCESS);

  int flag      = IDA_SUCCESS;
  sunrealtype t = t0;
  sunrealtype tret;
  int pivot_num      = 0;
  int check_num      = 0;
  int piv_call_count = 0;

  while (flag != IDA_TSTOP_RETURN)
  {
    spec_changed = SUNFALSE;

    long int netf;
    TEST_ASSERT(IDAGetNumErrTestFails(ida_mem, &netf) == IDA_SUCCESS);
    sunbooleantype trouble = netf > prev_netf;
    prev_netf              = netf;

    if (trouble)
    {
      check_num++;
      piv_call_count++;

      TEST_ASSERT(PIVPivot(pm, ZERO, t, Y, &spec_changed) == SUN_SUCCESS);
      DD_LOG(stats_file,
             "# check %d at t=%.6f (netf=%ld, spec_changed=%d)\n",
             check_num,
             (double)t,
             netf,
             spec_changed ? 1 : 0);

      if (spec_changed)
      {
        pivot_num++;
        DD_LOG(stats_file, "# pivot %d at t=%.6f\n", pivot_num, (double)t);
        TEST_ASSERT(DD_PRINT_ALL_STATS(ida_mem,
                                       stats_file,
                                       SUN_OUTPUTFORMAT_TABLE) == IDA_SUCCESS);
        TEST_ASSERT(DDSetSpec(dd_mem, pm->spec) == SUN_SUCCESS);
      }
    }

    const sunrealtype x = P_Ith(Y, 0, 0), y = P_Ith(Y, 1, 0),
                      lam = P_Ith(Y, 2, 0);

    DD_LOG(file,
           "%.20f,%.20f,%.20f,%.20f,%.20f,%d\n",
           t,
           x,
           y,
           lam,
           x * x + y * y - l * l,
           spec_changed ? 1 : 0);

    t += tstep;

    flag = DDSolve(dd_mem, t, &tret, Y, IDA_NORMAL);
    TEST_ASSERT(flag >= 0);
  }

  DD_LOG(stats_file, "# total PIVPivot calls: %d\n", piv_call_count);
  DD_LOG(stats_file, "# total pivots: %d\n", pivot_num);
  DD_LOG(stats_file,
         "# precision (pivots / checks): %.1f%%\n",
         piv_call_count > 0 ? 100.0 * (double)pivot_num / (double)piv_call_count
                            : 0.0);

  /* Cleanup */
  DDFree(&dd_mem);
  PIVDestroy(&pm);
  PIVMatDestroy(pJ0);
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
