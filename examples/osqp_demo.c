#include "osqp.h"


int main(int argc, char **argv) {
  // Load problem data
  c_float P_x[3] = { 4.0, 1.0, 2.0, };  // non-zero values of P (upper tri)
  c_int   P_nnz  = 3;                   // number on non-zeros
  c_int   P_i[3] = { 0, 0, 1, };        // row Index 
  c_int   P_p[3] = { 0, 1, 3, };        // col Ptr
  c_float q[2]   = { 1.0, 1.0, };       // q
  c_float A_x[4] = { 1.0, 1.0, 1.0, 1.0, };  // non-zero values of A
  c_int   A_nnz  = 4;                   // number on non-zeros
  c_int   A_i[4] = { 0, 1, 0, 2, };     // row Index
  c_int   A_p[3] = { 0, 2, 4, };        // col Ptr
  c_float l[3]   = { 1.0, 0.0, 0.0, };  // lower bound 
  c_float u[3]   = { 1.0, 0.7, 0.7, };  // upper bound
  c_int n = 2;    // number of decision vars
  c_int m = 3;    // number of constraint rows

  // Exitflag
  c_int exitflag = 0;

  // Workspace structures
  OSQPWorkspace *work;
  OSQPSettings  *settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings));
  OSQPData      *data     = (OSQPData *)c_malloc(sizeof(OSQPData));

  // Populate data
  if (data) {
    data->n = n;
    data->m = m;
    data->P = csc_matrix(data->n, data->n, P_nnz, P_x, P_i, P_p);
    data->q = q;
    data->A = csc_matrix(data->m, data->n, A_nnz, A_x, A_i, A_p);
    data->l = l;
    data->u = u;
  }

  // Define solver settings as default
  if (settings) osqp_set_default_settings(settings);

  // Setup workspace
  exitflag = osqp_setup(&work, data, settings);

  // Solve Problem
  osqp_solve(work);

  // Clean workspace
  osqp_cleanup(work);
  if (data) {
    if (data->A) c_free(data->A);
    if (data->P) c_free(data->P);
    c_free(data);
  }
  if (settings)  c_free(settings);

  return exitflag;
}
