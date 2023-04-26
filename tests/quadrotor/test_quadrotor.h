#include "osqp.h"        // OSQP API
#include "auxil.h"       // Needed for cold_start()
#include "cs.h"          // CSC data structure
#include "util.h"        // Utilities for testing
#include "osqp_tester.h" // Basic testing script header

#include "quadrotor/data.h"
#include <time.h>

void test_quadrotor_solve()
{
  printf("===== TEST DOUBLE INTEGRATOR OSQP ===== \n");

  c_int exitflag, tmp_int;
  c_float tmp_float;
  csc *tmp_mat, *P_tmp;

  // Problem settings
  OSQPSettings *settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings));

  // Structures
  OSQPWorkspace *work; // Workspace
  OSQPData *data;      // Data

  // Populate data
  data = generate_problem_quadrotor();

  // Define Solver settings as default
  osqp_set_default_settings(settings);
  settings->max_iter   = 2000;
  settings->eps_abs    = 1e-4;
  settings->eps_prim_inf = 1e-4;
  settings->eps_rel    = 1e-4;
  settings->alpha      = 1.6;
  settings->polish     = 0;
  settings->scaling    = 0;
  settings->verbose    = 1;
  settings->warm_start = 1;

  // Setup workspace
  exitflag = osqp_setup(&work, data, settings);

  // Setup correct
  mu_assert("Quadrotor test solve: Setup error!", exitflag == 0);

  clock_t start, end;
  double cpu_time_used;
  start = clock();
  // Solve Problem
  osqp_solve(work);
  end = clock();
  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  printf("time: %f\n", cpu_time_used);

  for (int k = 0; k < 20; ++k) {
    for (int j = 0; j < 4; ++j) {
      printf("Uhrz = %f\n", work->solution->x[k*(16)+j]);
    }
    for (int j = 0; j < 12; ++j) {
      printf("Xhrz = %f\n", work->solution->x[k*16+4+j]);
    }   
  }
  // Cleanup data
  clean_problem_quadrotor(data);

  // Cleanup
  c_free(settings);
}