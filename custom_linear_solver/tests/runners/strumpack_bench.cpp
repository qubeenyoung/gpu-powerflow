// Fair head-to-head harness: STRUMPACK (GPU multifrontal) on a power-flow Jacobian .mtx.
// Mirrors our solver's reporting: reorder(once) + factor + solve timed separately, GPU on,
// DIRECT (no Krylov/iter-refine) so it's a pure 1-factor+1-solve like our NR step.
// Batched power-flow has no STRUMPACK analog → emulate as reorder-once + B*(refactor+solve),
// which is exactly what a single-system solver must do for B same-pattern systems.
#include <chrono>
#include <vector>
#include <cstdio>
#include <string>
#include <cuda_runtime.h>
#include "StrumpackSparseSolver.hpp"
using namespace strumpack;
using clk = std::chrono::high_resolution_clock;
static double ms(clk::time_point a, clk::time_point b){
  return std::chrono::duration<double,std::milli>(b-a).count();
}
int main(int argc, char* argv[]){
  if (argc < 2){ std::printf("usage: %s J.mtx [B] [repeat]\n", argv[0]); return 1; }
  std::string f = argv[1];
  int B = (argc>2)? std::atoi(argv[2]) : 1;
  int repeat = (argc>3)? std::atoi(argv[3]) : 10;

  CSRMatrix<double,int> A;
  if (A.read_matrix_market(f) != 0){ std::printf("read failed: %s\n", f.c_str()); return 1; }
  int N = A.size();
  std::vector<double> b(N, 1.0), x(N, 0.0);

  StrumpackSparseSolver<double,int> spss;
  spss.options().set_from_command_line(argc, argv);
  if (const char* ndp = getenv("NDP")) spss.options().set_nd_param(atoi(ndp)); // tree-depth knob (fairness)
  spss.options().set_Krylov_solver(KrylovSolver::DIRECT); // pure direct solve
  spss.options().enable_gpu();
  spss.options().set_compression(CompressionType::NONE);  // no low-rank: dense multifrontal
  spss.set_matrix(A);

  // analyze (symbolic) once — shared across all same-pattern systems, like our solver.
  cudaDeviceSynchronize();
  auto t0=clk::now(); spss.reorder(); cudaDeviceSynchronize(); auto t1=clk::now();
  double reorder_ms = ms(t0,t1);

  // single-system: warm one factor+solve, then time.
  spss.factor(); spss.solve(b.data(), x.data());
  cudaDeviceSynchronize();
  auto f0=clk::now(); for(int r=0;r<repeat;++r) spss.factor(); cudaDeviceSynchronize(); auto f1=clk::now();
  auto s0=clk::now(); for(int r=0;r<repeat;++r) spss.solve(b.data(), x.data()); cudaDeviceSynchronize(); auto s1=clk::now();
  double factor_ms = ms(f0,f1)/repeat;
  double solve_ms  = ms(s0,s1)/repeat;
  double resid = A.max_scaled_residual(x.data(), b.data());

  // batched emulation: B same-pattern systems = reorder once + B*(refactor + solve).
  // per-system cost = avg over B (and over `repeat` outer reps for stability).
  double per_sys_fac=0, per_sys_sol=0, per_sys_update=0;
  {
    // (a) update-only cost (H2D value re-upload / re-marshal, no factor)
    cudaDeviceSynchronize();
    auto u0=clk::now();
    for(int r=0;r<repeat;++r)
      for(int k=0;k<B;++k){ spss.update_matrix_values(A); }
    cudaDeviceSynchronize(); auto u1=clk::now();
    per_sys_update = ms(u0,u1)/(repeat*(double)B);

    cudaDeviceSynchronize();
    auto bf0=clk::now();
    for(int r=0;r<repeat;++r)
      for(int k=0;k<B;++k){ spss.update_matrix_values(A); spss.factor(); }
    cudaDeviceSynchronize(); auto bf1=clk::now();
    auto bs0=clk::now();
    for(int r=0;r<repeat;++r)
      for(int k=0;k<B;++k) spss.solve(b.data(), x.data());
    cudaDeviceSynchronize(); auto bs1=clk::now();
    per_sys_fac = ms(bf0,bf1)/(repeat*(double)B);
    per_sys_sol = ms(bs0,bs1)/(repeat*(double)B);
  }

  std::printf("solver=STRUMPACK file=%s\n", f.c_str());
  std::printf("n=%d nnz=%d\n", N, A.nnz());
  std::printf("reorder_ms=%.4f\n", reorder_ms);
  std::printf("factor_ms=%.6f\n", factor_ms);
  std::printf("solve_ms=%.6f\n", solve_ms);
  std::printf("batch=%d\n", B);
  std::printf("batch_factor_per_sys_ms=%.6f\n", per_sys_fac);
  std::printf("batch_update_per_sys_ms=%.6f\n", per_sys_update);
  std::printf("batch_factor_only_per_sys_ms=%.6f\n", per_sys_fac - per_sys_update);
  std::printf("batch_solve_per_sys_ms=%.6f\n", per_sys_sol);
  std::printf("max_scaled_residual=%.3e\n", resid);
  return 0;
}
