// Head-to-head harness for GLU v3.0 (UC Riverside GPU sparse LU for circuit simulation)
// on a SuiteSparse / power-flow .mtx.
//
// GLU pipeline (mirrors src/lu_cmd.cpp, but fed from in-memory CSC so it accepts our
// row-major .mtx files, which NICSLU's column-major file reader rejects):
//   1. NICSLU preprocess  — AMD reorder + MC64 scaling on CPU  (NicsLU_Analyze)
//   2. Symbolic_Matrix    — fill_in -> csr -> predictLU -> leveling (CPU, level scheduling)
//   3. LUonDevice         — numeric LU factorization on GPU (overwrites val in place)
//   4. solve              — forward/backward substitution on CPU, un-permuted to original order
//
// Timing is symmetric with cudss_bench: analyze (once) + warmup + per-iteration MEDIAN of
// factor and of solve. GLU is SINGLE PRECISION (REAL=float) and B=1 only (no micro-batch),
// so results are reported per system with an fp32 residual (~1e-5 is expected, not a bug).
//
// The factor loop re-runs predictLU (cheap, untimed) before each LUonDevice to restore the
// original numeric values into val, since LUonDevice overwrites val with the LU factors --
// this mirrors a refactorization with a fixed sparsity pattern (a Newton step).
//
// usage: glu_bench A.mtx [repeat=10] [warmup=5] [-p]      (-p enables GESP perturbation)

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

// GLU / NICSLU headers — include order mirrors lu_cmd.cpp (yields C linkage for NICSLU).
#include "symbolic.h"
#include "numeric.h"
#include "preprocess.h"
#include "nicslu.h"

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

// ---- Matrix Market -> CSC (0-based; values double, indices uint32 for NICSLU) ----
// Row indices within each column are sorted (NICSLU/AMD expect a clean CSC).
struct CSC {
  unsigned n = 0, nnz = 0;
  std::vector<unsigned> ap;   // column pointers, size n+1
  std::vector<unsigned> ai;   // row indices,     size nnz
  std::vector<double>   ax;   // values,          size nnz
};

static bool read_mm_to_csc(const std::string& path, CSC& A){
  std::ifstream f(path);
  if(!f){ std::printf("open failed: %s\n", path.c_str()); return false; }

  // First line may be a MatrixMarket banner ("%%..."), a comment, or — for the bare
  // COO files GLU ships (e.g. add32.mtx) — already the dimensions line. Detect each.
  std::string line;
  std::getline(f, line);
  bool pattern = line.find("pattern") != std::string::npos;
  bool sym     = line.find("symmetric")      != std::string::npos ||
                 line.find("skew-symmetric") != std::string::npos ||
                 line.find("hermitian")      != std::string::npos;
  if(!line.empty() && line[0] == '%')          // banner/comment -> skip to dims line
    while(std::getline(f, line)) if(!line.empty() && line[0] != '%') break;

  int nr, nc; long nz;
  { std::istringstream is(line); is >> nr >> nc >> nz; }
  if(nr != nc){ std::printf("matrix is not square (%dx%d)\n", nr, nc); return false; }

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

  A.n = (unsigned)nr; A.nnz = (unsigned)I.size();
  // COO -> CSC (count per column, prefix sum, scatter)
  A.ap.assign(A.n + 1, 0);
  for(unsigned k = 0; k < A.nnz; ++k) A.ap[J[k] + 1]++;
  for(unsigned j = 0; j < A.n; ++j) A.ap[j + 1] += A.ap[j];
  A.ai.resize(A.nnz); A.ax.resize(A.nnz);
  std::vector<unsigned> cur(A.ap.begin(), A.ap.end() - 1);
  for(unsigned k = 0; k < A.nnz; ++k){ unsigned c = J[k]; unsigned d = cur[c]++; A.ai[d] = I[k]; A.ax[d] = V[k]; }

  // sort row indices within each column
  std::vector<std::pair<unsigned,double>> col;
  for(unsigned j = 0; j < A.n; ++j){
    unsigned b = A.ap[j], e = A.ap[j + 1];
    col.clear();
    for(unsigned k = b; k < e; ++k) col.emplace_back(A.ai[k], A.ax[k]);
    std::sort(col.begin(), col.end(),
              [](const std::pair<unsigned,double>& a, const std::pair<unsigned,double>& c){
                return a.first < c.first;
              });
    for(unsigned k = b; k < e; ++k){ A.ai[k] = col[k - b].first; A.ax[k] = col[k - b].second; }
  }
  return true;
}

int main(int argc, char* argv[]){
  if(argc < 2){
    std::printf("usage: %s A.mtx [repeat=10] [warmup=5] [-p]\n", argv[0]);
    return 1;
  }
  std::string file = argv[1];
  int  repeat = 10, warmup = 5;
  bool perturb = false;
  for(int i = 2; i < argc; ++i){
    if(std::strcmp(argv[i], "-p") == 0) perturb = true;
    else if(repeat == 10 && warmup == 5 && i == 2) repeat = std::atoi(argv[i]);
    else warmup = std::atoi(argv[i]);
  }

  CSC A;
  if(!read_mm_to_csc(file, A)) return 1;
  unsigned n = A.n, nnz = A.nnz;

  // ---- NICSLU preprocessing: AMD reorder + scaling, dumped back to CSC ----
  // (replicates preprocess_from_arrays against this NICSLU's 5-arg CreateMatrix)
  SNicsLU* nicslu = (SNicsLU*)std::malloc(sizeof(SNicsLU));
  if(!nicslu){ std::printf("alloc nicslu failed\n"); return 1; }
  NicsLU_Initialize(nicslu);
  if(NicsLU_CreateMatrix(nicslu, n, nnz, A.ax.data(), A.ai.data(), A.ap.data()) != NICS_OK){
    std::printf("NicsLU_CreateMatrix error\n"); return 1;
  }
  nicslu->cfgi[0] = 1;   // enable column ordering
  nicslu->cfgf[1] = 0;
  if(NicsLU_Analyze(nicslu) != NICS_OK){ std::printf("NicsLU_Analyze error\n"); return 1; }

  double*       rax = nullptr;   // reordered CSC handed to the symbolic phase
  unsigned int* rai = nullptr;
  unsigned int* rap = nullptr;
  if(my_DumpA(nicslu, &rax, &rai, &rap) != 0){ std::printf("my_DumpA failed\n"); return 1; }

  // ---- symbolic analysis (CPU, once): fill-in, CSR transpose, level scheduling ----
  std::ostringstream sout, serr;   // swallow GLU's chatty stdout
  auto a0 = clk::now();
  Symbolic_Matrix A_sym(nicslu->n, sout, serr);
  A_sym.fill_in(rai, rap);
  A_sym.csr();
  A_sym.predictLU(rai, rap, rax);
  A_sym.leveling();
  double analysis_ms = ms(a0, clk::now());

  // ---- warmup: factor + solve a few times ----
  // predictLU appends to val (it assumes a fresh, empty val), so clear it before each
  // refactor to restore the original numeric values with zeros at the fill-in positions.
  std::vector<REAL> b(nicslu->n, 1.0f);
  for(int w = 0; w < warmup; ++w){
    A_sym.val.clear();
    A_sym.predictLU(rai, rap, rax);             // restore values before refactor
    LUonDevice(A_sym, sout, serr, perturb);
    (void)A_sym.solve(nicslu, b);
  }

  // ---- timed: per-iteration median of factor (GPU) and solve (CPU) ----
  std::vector<double> ft(repeat), st(repeat);
  for(int r = 0; r < repeat; ++r){
    A_sym.val.clear();
    A_sym.predictLU(rai, rap, rax);             // untimed reset (refactorization pattern)
    auto t = clk::now();
    LUonDevice(A_sym, sout, serr, perturb);     // includes H2D + GPU kernels + D2H
    ft[r] = ms(t, clk::now());
  }
  std::vector<REAL> x;
  for(int r = 0; r < repeat; ++r){
    auto t = clk::now();
    x = A_sym.solve(nicslu, b);
    st[r] = ms(t, clk::now());
  }
  double factor_ms = median(ft);
  double solve_ms  = median(st);

  // ---- residual ||b - A x|| / ||b|| against the ORIGINAL matrix, b = 1 ----
  // x is in original ordering; accumulate A x from the original CSC.
  std::vector<double> Ax(n, 0.0);
  for(unsigned j = 0; j < n; ++j)
    for(unsigned k = A.ap[j]; k < A.ap[j + 1]; ++k)
      Ax[A.ai[k]] += A.ax[k] * (double)x[j];
  double rn = 0, bn = 0;
  for(unsigned i = 0; i < n; ++i){ double r = 1.0 - Ax[i]; rn += r*r; bn += 1.0; }
  double relres = std::sqrt(rn) / std::sqrt(bn);

  std::printf("solver=GLU file=%s\n", file.c_str());
  std::printf("n=%u nnz=%u prec=fp32 B=1 warmup=%d perturb=%d\n", n, nnz, warmup, (int)perturb);
  std::printf("analysis_ms=%.4f\n", analysis_ms);
  std::printf("factor_per_sys_ms=%.6f\n", factor_ms);
  std::printf("solve_per_sys_ms=%.6f\n", solve_ms);
  std::printf("relres=%.3e\n", relres);

  std::free(rax); std::free(rai); std::free(rap);
  NicsLU_Destroy(nicslu);
  std::free(nicslu);
  return 0;
}
