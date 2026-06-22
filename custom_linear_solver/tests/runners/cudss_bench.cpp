// Head-to-head harness for NVIDIA cuDSS (GPU sparse direct LU) on a SuiteSparse .mtx.
//
// Pipeline: analyze (once) -> warmup -> timed refactorize + solve, GPU, DIRECT (no
// iterative refinement). Factor is timed as REFACTORIZATION (same sparsity pattern,
// like a Newton step) so it mirrors how the custom runner reuses its symbolic phase.
// Timing uses a warmup burst followed by the per-iteration MEDIAN, which is robust to
// noise and symmetric with the custom runner's methodology.
//
// Supports fp32/fp64 and a micro-batch (UBATCH) of B independent systems sharing one
// sparsity pattern; batch results are reported per-system (divided by B).
//
// The Matrix Market reader handles coordinate real/integer/pattern, general or symmetric
// (symmetric/skew-symmetric/hermitian are expanded to full for a general-LU comparison).
// Built against cuDSS 0.7.
//
// usage: cudss_bench A.mtx [repeat=10] [B=1] [fp32|fp64=fp64] [warmup=5]

#include <chrono>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <cudss.h>

using clk = std::chrono::high_resolution_clock;

static double ms(clk::time_point a, clk::time_point b){
  return std::chrono::duration<double, std::milli>(b - a).count();
}

static double median(std::vector<double>& v){
  if(v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  size_t n = v.size();
  return (n & 1) ? v[n/2] : 0.5 * (v[n/2 - 1] + v[n/2]);
}

#define CU_CHECK(x) do{ cudaError_t e = (x); if(e != cudaSuccess){ \
  std::printf("CUDA error %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); std::exit(1);} }while(0)
#define DSS_CHECK(x) do{ cudssStatus_t s = (x); if(s != CUDSS_STATUS_SUCCESS){ \
  std::printf("cuDSS error %s:%d status=%d\n", __FILE__, __LINE__, (int)s); std::exit(1);} }while(0)

// ---- Matrix Market -> CSR (0-based, double, full/general) ----
struct CSR { int n = 0; long nnz = 0; std::vector<int> rp, ci; std::vector<double> v; };

static bool read_mm_to_csr(const std::string& path, CSR& A){
  std::ifstream f(path);
  if(!f){ std::printf("open failed: %s\n", path.c_str()); return false; }

  std::string line;
  std::getline(f, line);
  bool pattern = line.find("pattern") != std::string::npos;
  bool sym     = line.find("symmetric")      != std::string::npos ||
                 line.find("skew-symmetric") != std::string::npos ||
                 line.find("hermitian")      != std::string::npos;

  // skip comment lines
  while(std::getline(f, line)) if(!line.empty() && line[0] != '%') break;

  int nr, nc; long nz;
  { std::istringstream is(line); is >> nr >> nc >> nz; }

  std::vector<int> I, J; std::vector<double> V;
  I.reserve(sym ? 2*nz : nz); J.reserve(sym ? 2*nz : nz); V.reserve(sym ? 2*nz : nz);
  for(long k = 0; k < nz; ++k){
    if(!std::getline(f, line)){ std::printf("mtx truncated\n"); return false; }
    std::istringstream is(line); int i, j; double val = 1.0;
    is >> i >> j; if(!pattern) is >> val;
    --i; --j;
    I.push_back(i); J.push_back(j); V.push_back(val);
    if(sym && i != j){ I.push_back(j); J.push_back(i); V.push_back(val); }
  }

  A.n = nr; A.nnz = (long)I.size();
  // COO -> CSR
  A.rp.assign(nr + 1, 0);
  for(long k = 0; k < A.nnz; ++k) A.rp[I[k] + 1]++;
  for(int i = 0; i < nr; ++i) A.rp[i + 1] += A.rp[i];
  A.ci.resize(A.nnz); A.v.resize(A.nnz);
  std::vector<int> cur(A.rp.begin(), A.rp.end() - 1);
  for(long k = 0; k < A.nnz; ++k){ int r = I[k]; int d = cur[r]++; A.ci[d] = J[k]; A.v[d] = V[k]; }
  return true;
}

int main(int argc, char* argv[]){
  if(argc < 2){
    std::printf("usage: %s A.mtx [repeat=10] [B=1] [fp32|fp64=fp64] [warmup=5]\n", argv[0]);
    return 1;
  }
  std::string file = argv[1];
  int  repeat = (argc > 2) ? std::atoi(argv[2]) : 10;
  int  B      = (argc > 3) ? std::atoi(argv[3]) : 1;
  bool fp32   = (argc > 4) && std::strcmp(argv[4], "fp32") == 0;
  int  warmup = (argc > 5) ? std::atoi(argv[5]) : 5;

  cudaDataType valtype = fp32 ? CUDA_R_32F : CUDA_R_64F;
  size_t       elem    = fp32 ? sizeof(float) : sizeof(double);

  CSR A;
  if(!read_mm_to_csr(file, A)) return 1;
  int n = A.n; long nnz = A.nnz;

  // ---- device buffers (B independent systems share rp/ci) ----
  int  *d_rp, *d_ci;
  void *d_v, *d_b, *d_x;
  CU_CHECK(cudaMalloc(&d_rp, (n + 1) * sizeof(int)));
  CU_CHECK(cudaMalloc(&d_ci, nnz * sizeof(int)));
  CU_CHECK(cudaMalloc(&d_v, (size_t)B * nnz * elem));
  CU_CHECK(cudaMalloc(&d_b, (size_t)B * n * elem));
  CU_CHECK(cudaMalloc(&d_x, (size_t)B * n * elem));
  CU_CHECK(cudaMemset(d_x, 0, (size_t)B * n * elem));
  CU_CHECK(cudaMemcpy(d_rp, A.rp.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CU_CHECK(cudaMemcpy(d_ci, A.ci.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));

  // upload values (broadcast the single matrix to all B systems) and rhs = 1
  if(fp32){
    std::vector<float> vv((size_t)B * nnz), bb((size_t)B * n, 1.0f);
    for(int k = 0; k < B; ++k) for(long t = 0; t < nnz; ++t) vv[(size_t)k*nnz + t] = (float)A.v[t];
    CU_CHECK(cudaMemcpy(d_v, vv.data(), (size_t)B * nnz * elem, cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemcpy(d_b, bb.data(), (size_t)B * n   * elem, cudaMemcpyHostToDevice));
  } else {
    std::vector<double> vv((size_t)B * nnz), bb((size_t)B * n, 1.0);
    for(int k = 0; k < B; ++k) for(long t = 0; t < nnz; ++t) vv[(size_t)k*nnz + t] = A.v[t];
    CU_CHECK(cudaMemcpy(d_v, vv.data(), (size_t)B * nnz * elem, cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemcpy(d_b, bb.data(), (size_t)B * n   * elem, cudaMemcpyHostToDevice));
  }

  cudssHandle_t handle; DSS_CHECK(cudssCreate(&handle));
  cudssConfig_t config; DSS_CHECK(cudssConfigCreate(&config));
  cudssData_t   data;   DSS_CHECK(cudssDataCreate(handle, &data));
  if(B > 1){ int ub = B; DSS_CHECK(cudssConfigSet(config, CUDSS_CONFIG_UBATCH_SIZE, &ub, sizeof(ub))); }

  cudssMatrix_t Am, Bm, Xm;
  DSS_CHECK(cudssMatrixCreateCsr(&Am, n, n, nnz, d_rp, nullptr, d_ci, d_v,
            CUDA_R_32I, valtype, CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));
  DSS_CHECK(cudssMatrixCreateDn(&Bm, n, 1, n, d_b, valtype, CUDSS_LAYOUT_COL_MAJOR));
  DSS_CHECK(cudssMatrixCreateDn(&Xm, n, 1, n, d_x, valtype, CUDSS_LAYOUT_COL_MAJOR));

  // ---- analyze (symbolic) once ----
  CU_CHECK(cudaDeviceSynchronize());
  auto a0 = clk::now();
  DSS_CHECK(cudssExecute(handle, CUDSS_PHASE_ANALYSIS, config, data, Am, Xm, Bm));
  CU_CHECK(cudaDeviceSynchronize());
  double analysis_ms = ms(a0, clk::now());

  // one full factorization to populate the numeric factors, then warmup refactor+solve
  DSS_CHECK(cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, config, data, Am, Xm, Bm));
  DSS_CHECK(cudssExecute(handle, CUDSS_PHASE_SOLVE,         config, data, Am, Xm, Bm));
  for(int r = 0; r < warmup; ++r){
    DSS_CHECK(cudssExecute(handle, CUDSS_PHASE_REFACTORIZATION, config, data, Am, Xm, Bm));
    DSS_CHECK(cudssExecute(handle, CUDSS_PHASE_SOLVE,           config, data, Am, Xm, Bm));
  }
  CU_CHECK(cudaDeviceSynchronize());

  // ---- timed: per-iteration median of refactorize and of solve ----
  std::vector<double> ft(repeat), st(repeat);
  for(int r = 0; r < repeat; ++r){
    CU_CHECK(cudaDeviceSynchronize());
    auto t = clk::now();
    DSS_CHECK(cudssExecute(handle, CUDSS_PHASE_REFACTORIZATION, config, data, Am, Xm, Bm));
    CU_CHECK(cudaDeviceSynchronize());
    ft[r] = ms(t, clk::now());
  }
  for(int r = 0; r < repeat; ++r){
    CU_CHECK(cudaDeviceSynchronize());
    auto t = clk::now();
    DSS_CHECK(cudssExecute(handle, CUDSS_PHASE_SOLVE, config, data, Am, Xm, Bm));
    CU_CHECK(cudaDeviceSynchronize());
    st[r] = ms(t, clk::now());
  }
  double factor_ms = median(ft) / B;  // per system
  double solve_ms  = median(st) / B;  // per system

  // ---- residual ||b - A x|| / ||b|| for system 0, b = 1 ----
  std::vector<double> x(n);
  if(fp32){
    std::vector<float> xf(n);
    CU_CHECK(cudaMemcpy(xf.data(), d_x, n * elem, cudaMemcpyDeviceToHost));
    for(int i = 0; i < n; ++i) x[i] = xf[i];
  } else {
    CU_CHECK(cudaMemcpy(x.data(), d_x, n * elem, cudaMemcpyDeviceToHost));
  }
  double rn = 0, bn = 0;
  for(int i = 0; i < n; ++i){
    double ax = 0;
    for(int k = A.rp[i]; k < A.rp[i+1]; ++k) ax += A.v[k] * x[A.ci[k]];
    double r = 1.0 - ax;  // b[i] == 1
    rn += r * r; bn += 1.0;
  }
  double relres = std::sqrt(rn) / std::sqrt(bn);

  std::printf("solver=cuDSS file=%s\n", file.c_str());
  std::printf("n=%d nnz=%ld prec=%s B=%d warmup=%d\n", n, nnz, fp32 ? "fp32" : "fp64", B, warmup);
  std::printf("analysis_ms=%.4f\n", analysis_ms);
  std::printf("factor_per_sys_ms=%.6f\n", factor_ms);
  std::printf("solve_per_sys_ms=%.6f\n", solve_ms);
  std::printf("relres=%.3e\n", relres);

  cudssMatrixDestroy(Am); cudssMatrixDestroy(Bm); cudssMatrixDestroy(Xm);
  cudssDataDestroy(handle, data); cudssConfigDestroy(config); cudssDestroy(handle);
  cudaFree(d_rp); cudaFree(d_ci); cudaFree(d_v); cudaFree(d_b); cudaFree(d_x);
  return 0;
}
