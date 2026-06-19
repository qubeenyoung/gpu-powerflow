// Fair head-to-head harness: NVIDIA cuDSS (GPU sparse direct LU) on a SuiteSparse .mtx.
// Mirrors strumpack_bench.cpp: analyze(once) + factor + solve timed separately, GPU, FP64,
// DIRECT (no iterative refinement). Reports analysis/factor/solve ms + scaled residual.
// Matrix Market reader handles coordinate real/integer/pattern, general or symmetric (expanded
// to full for a general-LU comparison). Built against cuDSS 0.7.
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
  return std::chrono::duration<double,std::milli>(b-a).count();
}
#define CU_CHECK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  std::printf("CUDA error %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); std::exit(1);} }while(0)
#define DSS_CHECK(x) do{ cudssStatus_t s=(x); if(s!=CUDSS_STATUS_SUCCESS){ \
  std::printf("cuDSS error %s:%d status=%d\n",__FILE__,__LINE__,(int)s); std::exit(1);} }while(0)

// ---- Matrix Market -> CSR (0-based, double, full/general) ----
struct CSR { int n=0; long nnz=0; std::vector<int> rp, ci; std::vector<double> v; };

static bool read_mm_to_csr(const std::string& path, CSR& A){
  std::ifstream f(path);
  if(!f){ std::printf("open failed: %s\n", path.c_str()); return false; }
  std::string line; std::getline(f, line);
  bool pattern = line.find("pattern")!=std::string::npos;
  bool sym     = line.find("symmetric")!=std::string::npos ||
                 line.find("skew-symmetric")!=std::string::npos ||
                 line.find("hermitian")!=std::string::npos;
  // skip comments
  while(std::getline(f,line)) if(!line.empty() && line[0]!='%') break;
  int nr,nc; long nz;
  { std::istringstream is(line); is>>nr>>nc>>nz; }
  std::vector<int> I; std::vector<int> J; std::vector<double> V;
  I.reserve(sym?2*nz:nz); J.reserve(sym?2*nz:nz); V.reserve(sym?2*nz:nz);
  for(long k=0;k<nz;++k){
    if(!std::getline(f,line)){ std::printf("mtx truncated\n"); return false; }
    std::istringstream is(line); int i,j; double val=1.0;
    is>>i>>j; if(!pattern) is>>val;
    --i; --j;
    I.push_back(i); J.push_back(j); V.push_back(val);
    if(sym && i!=j){ I.push_back(j); J.push_back(i); V.push_back(val); }
  }
  A.n = nr; A.nnz = (long)I.size();
  // COO -> CSR
  A.rp.assign(nr+1,0);
  for(long k=0;k<A.nnz;++k) A.rp[I[k]+1]++;
  for(int i=0;i<nr;++i) A.rp[i+1]+=A.rp[i];
  A.ci.resize(A.nnz); A.v.resize(A.nnz);
  std::vector<int> cur(A.rp.begin(), A.rp.end()-1);
  for(long k=0;k<A.nnz;++k){ int r=I[k]; int d=cur[r]++; A.ci[d]=J[k]; A.v[d]=V[k]; }
  return true;
}

int main(int argc, char* argv[]){
  if(argc<2){ std::printf("usage: %s A.mtx [repeat]\n", argv[0]); return 1; }
  std::string file = argv[1];
  int repeat = (argc>2)? std::atoi(argv[2]) : 10;

  CSR A;
  if(!read_mm_to_csr(file, A)) return 1;
  int n = A.n; long nnz = A.nnz;
  std::vector<double> b(n,1.0), x(n,0.0);

  // device buffers
  int *d_rp,*d_ci; double *d_v,*d_b,*d_x;
  CU_CHECK(cudaMalloc(&d_rp,(n+1)*sizeof(int)));
  CU_CHECK(cudaMalloc(&d_ci, nnz*sizeof(int)));
  CU_CHECK(cudaMalloc(&d_v,  nnz*sizeof(double)));
  CU_CHECK(cudaMalloc(&d_b,  n*sizeof(double)));
  CU_CHECK(cudaMalloc(&d_x,  n*sizeof(double)));
  CU_CHECK(cudaMemcpy(d_rp,A.rp.data(),(n+1)*sizeof(int),cudaMemcpyHostToDevice));
  CU_CHECK(cudaMemcpy(d_ci,A.ci.data(), nnz*sizeof(int),cudaMemcpyHostToDevice));
  CU_CHECK(cudaMemcpy(d_v, A.v.data(),  nnz*sizeof(double),cudaMemcpyHostToDevice));
  CU_CHECK(cudaMemcpy(d_b, b.data(),    n*sizeof(double),cudaMemcpyHostToDevice));

  cudssHandle_t handle; DSS_CHECK(cudssCreate(&handle));
  cudssConfig_t config; DSS_CHECK(cudssConfigCreate(&config));
  cudssData_t data;     DSS_CHECK(cudssDataCreate(handle,&data));

  cudssMatrix_t Am, Xm, Bm;
  DSS_CHECK(cudssMatrixCreateCsr(&Am, n, n, nnz, d_rp, nullptr, d_ci, d_v,
            CUDA_R_32I, CUDA_R_64F, CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));
  DSS_CHECK(cudssMatrixCreateDn(&Bm, n, 1, n, d_b, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR));
  DSS_CHECK(cudssMatrixCreateDn(&Xm, n, 1, n, d_x, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR));

  // analyze (symbolic) once
  CU_CHECK(cudaDeviceSynchronize());
  auto t0=clk::now();
  DSS_CHECK(cudssExecute(handle,CUDSS_PHASE_ANALYSIS,config,data,Am,Xm,Bm));
  CU_CHECK(cudaDeviceSynchronize());
  auto t1=clk::now();
  double analysis_ms = ms(t0,t1);

  // warm one factor+solve
  DSS_CHECK(cudssExecute(handle,CUDSS_PHASE_FACTORIZATION,config,data,Am,Xm,Bm));
  DSS_CHECK(cudssExecute(handle,CUDSS_PHASE_SOLVE,config,data,Am,Xm,Bm));
  CU_CHECK(cudaDeviceSynchronize());

  // time factor (refactorization = same pattern, like a Newton step)
  auto f0=clk::now();
  for(int r=0;r<repeat;++r)
    DSS_CHECK(cudssExecute(handle,CUDSS_PHASE_REFACTORIZATION,config,data,Am,Xm,Bm));
  CU_CHECK(cudaDeviceSynchronize());
  auto f1=clk::now();

  auto s0=clk::now();
  for(int r=0;r<repeat;++r)
    DSS_CHECK(cudssExecute(handle,CUDSS_PHASE_SOLVE,config,data,Am,Xm,Bm));
  CU_CHECK(cudaDeviceSynchronize());
  auto s1=clk::now();

  double factor_ms = ms(f0,f1)/repeat;
  double solve_ms  = ms(s0,s1)/repeat;

  // residual ||b - A x|| / ||b||
  CU_CHECK(cudaMemcpy(x.data(),d_x,n*sizeof(double),cudaMemcpyDeviceToHost));
  double rn=0, bn=0;
  for(int i=0;i<n;++i){
    double ax=0; for(int k=A.rp[i];k<A.rp[i+1];++k) ax += A.v[k]*x[A.ci[k]];
    double r=b[i]-ax; rn+=r*r; bn+=b[i]*b[i];
  }
  double relres = std::sqrt(rn)/std::sqrt(bn);

  std::printf("solver=cuDSS file=%s\n", file.c_str());
  std::printf("n=%d nnz=%ld\n", n, nnz);
  std::printf("analysis_ms=%.4f\n", analysis_ms);
  std::printf("factor_ms=%.6f\n", factor_ms);
  std::printf("solve_ms=%.6f\n", solve_ms);
  std::printf("relres=%.3e\n", relres);

  cudssMatrixDestroy(Am); cudssMatrixDestroy(Bm); cudssMatrixDestroy(Xm);
  cudssDataDestroy(handle,data); cudssConfigDestroy(config); cudssDestroy(handle);
  cudaFree(d_rp);cudaFree(d_ci);cudaFree(d_v);cudaFree(d_b);cudaFree(d_x);
  return 0;
}
