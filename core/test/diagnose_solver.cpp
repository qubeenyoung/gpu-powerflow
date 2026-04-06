/**
 * @file diagnose_solver.cpp
 * @brief cuDSS REFACTORIZATION / SOLVE 병목 정밀 진단
 *
 * 목표: Solve가 Factorize보다 2.7x 느린 이유 특정
 *
 * 실험 1: IR_N_STEPS (default, 0, 1, 2, 3) → Refact / Solve 분리 시간
 *   가설 A: cuDSS 기본값으로 Iterative Refinement가 돌고 있음
 *   → IR=0 (off) 시 Solve ≈ Factorize 수준이면 A 확인
 *
 * 실험 2: batch_size별 REFACTORIZATION / SOLVE 분리 타이밍 (IR=0)
 *   → 각 단계가 batch에 따라 어떻게 스케일되는가?
 *
 * 실험 3: 전체 step breakdown (IR=default, batch=8)
 *   → REFACT vs SOLVE 비율을 직접 측정
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <numeric>
#include <chrono>
#include <cmath>
#include <complex>

#include "nr_data.hpp"
#include "newtonpf.hpp"    // Jacobian struct, mismatch()
#include "cuda_accel.hpp"
#include "spdlog/spdlog.h"

#ifdef USE_CUDSS
#include "cudss_solver.hpp"
#endif

using namespace nr_data;
using clk = std::chrono::high_resolution_clock;

static double wall_ms(clk::time_point t0, clk::time_point t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static void sep(const char* title) {
    std::cout << "\n" << std::string(64, '=') << "\n"
              << "  " << title << "\n"
              << std::string(64, '=') << "\n";
}

#if defined(USE_CUDA) && defined(USE_CUDSS)

// ─── iteration 단계별 타이밍 ─────────────────────────────────────────────────
struct IterTimings {
    float jacobian_ms  = 0;
    float perm_ms      = 0;
    float refact_ms    = 0;
    float solve_ms     = 0;
    float updateV_ms   = 0;
    float mismatch_ms  = 0;
};

// ─── NR 파이프라인 수동 구동 (분리 타이밍) ───────────────────────────────────
static std::vector<IterTimings> run_nr_manual(
    const NRData& d,
    CuDSSSolver& solver,
    NewtonCudaAccel& accel,
    int batch_size,
    double tol = 1e-8,
    int max_iter = 50
) {
    const int npv  = d.pv.size();
    const int npq  = d.pq.size();
    const int dimF = npv + 2 * npq;

    // 초기 V/Sbus GPU 업로드
    {
        std::vector<const void*> V0_ptrs(batch_size);
        std::vector<const void*> sbus_ptrs(batch_size);
        for (int b = 0; b < batch_size; ++b) {
            V0_ptrs[b]   = static_cast<const void*>(d.V0.data());
            sbus_ptrs[b] = static_cast<const void*>(d.Sbus.data());
        }
        accel.upload_batch_initial(
            V0_ptrs.data(),    // V_ptrs
            sbus_ptrs.data(),  // Sbus_ptrs
            batch_size
        );
    }

    std::vector<double> normF_batch(batch_size, 1e10);
    bool converged = false;
    int iter = 0;
    std::vector<IterTimings> timings;

    while (!converged && iter < max_iter) {
        IterTimings t;

        // (1) Mismatch — normF_out은 필수 (nullptr 불가)
        {
            cudaDeviceSynchronize();
            auto t0 = clk::now();
            accel.compute_mismatch_batch(
                d.pv.data(), d.pq.data(), npv, npq,
                nullptr,              // F_batch_out: nullptr=GPU 상주
                normF_batch.data()    // normF_out: 필수 (D→H 소량)
            );
            cudaDeviceSynchronize();
            t.mismatch_ms = (float)wall_ms(t0, clk::now());
        }

        double maxNorm = *std::max_element(normF_batch.begin(), normF_batch.end());
        if (maxNorm < tol) { converged = true; break; }
        ++iter;

        // (2) Jacobian (GPU FP32, V GPU 상주)
        float* d_J = nullptr;
        {
            cudaDeviceSynchronize();
            auto t0 = clk::now();
            d_J = accel.update_jacobian_batch_no_upload();
            cudaDeviceSynchronize();
            t.jacobian_ms = (float)wall_ms(t0, clk::now());
        }

        // (3) Permutation
        {
            cudaDeviceSynchronize();
            auto t0 = clk::now();
            solver.applyPermutationUBatchFP32(d_J);
            cudaDeviceSynchronize();
            t.perm_ms = (float)wall_ms(t0, clk::now());
        }

        // (4) b = -F (GPU FP64→FP32 변환), REFACT + SOLVE (분리 타이밍)
        {
            float* d_b = solver.getBatchBBuffer();
            float* d_x = solver.getBatchXBuffer();
            accel.negate_F_to_fp32(d_b, dimF);
            solver.factorizeAndSolveUBatchFP32_timed(d_b, d_x, t.refact_ms, t.solve_ms);
        }

        // (5) UpdateV (GPU FP32 device 직접)
        {
            cudaDeviceSynchronize();
            auto t0 = clk::now();
            float* d_x = solver.getBatchXBuffer();
            accel.update_voltage_batch_from_fp32_device(
                d_x, d.pv.data(), d.pq.data(), npv, npq);
            cudaDeviceSynchronize();
            t.updateV_ms = (float)wall_ms(t0, clk::now());
        }

        timings.push_back(t);
    }

    return timings;
}

// ─── 공통 초기화 ─────────────────────────────────────────────────────────────
static void init_pipeline(
    const NRData& d, const Jacobian& jac,
    CuDSSSolver& solver, NewtonCudaAccel& accel,
    int batch_size, int ir_steps = -1  // -1 = default
) {
    accel.initialize(
        d.V0.size(), jac.J.nonZeros(), &d.Ybus,
        jac.mapJ11, jac.mapJ21, jac.mapJ12, jac.mapJ22,
        jac.diagMapJ11, jac.diagMapJ21, jac.diagMapJ12, jac.diagMapJ22
    );
    solver.analyzePattern(jac.J);
    if (ir_steps >= 0) solver.setIRSteps(ir_steps);
    solver.analyzePatternUBatch(jac.J, batch_size);
}

// ─────────────────────────────────────────────────────────────────────────────
// 실험 1: IR_N_STEPS별 Refact / Solve 시간 (batch=1)
// ─────────────────────────────────────────────────────────────────────────────
static void exp1_ir_steps(const NRData& d, const Jacobian& jac) {
    sep("Exp 1: IR_N_STEPS vs REFACT / SOLVE Time (batch=1)");

    std::cout << "가설: cuDSS 기본 IR이 Solve를 비대하게 만드는 주범\n\n";
    std::cout << std::setw(12) << "IR_N_STEPS"
              << std::setw(16) << "Refact(ms)"
              << std::setw(16) << "Solve(ms)"
              << std::setw(14) << "S/R ratio"
              << "\n" << std::string(58, '-') << "\n";

    const int BS = 1, WARM = 3, RUNS = 7;

    for (int ir : {-1, 0, 1, 2, 3}) {
        CuDSSSolver solver;
        NewtonCudaAccel accel;
        init_pipeline(d, jac, solver, accel, BS, ir);

        // warm-up
        for (int w = 0; w < WARM; ++w)
            run_nr_manual(d, solver, accel, BS);

        float refact_sum = 0, solve_sum = 0;
        int ni = 0;
        for (int r = 0; r < RUNS; ++r) {
            auto tv = run_nr_manual(d, solver, accel, BS);
            for (auto& t : tv) { refact_sum += t.refact_ms; solve_sum += t.solve_ms; }
            ni += (int)tv.size();
        }
        float ra = refact_sum / ni, sa = solve_sum / ni;

        std::string label = (ir < 0) ? "default" : std::to_string(ir);
        std::cout << std::setw(12) << label
                  << std::setw(16) << std::fixed << std::setprecision(3) << ra
                  << std::setw(16) << std::setprecision(3) << sa
                  << std::setw(14) << std::setprecision(2) << sa / ra
                  << "\n";
    }
    std::cout << "\n해석:\n"
              << "  S/R >> 1  → IR이 Solve 시간 지배 (IR 줄이면 효과적)\n"
              << "  S/R ≈ 1   → triangular solve 자체가 병목 (다른 접근 필요)\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// 실험 2: batch_size별 REFACT / SOLVE 분리 (IR=0)
// ─────────────────────────────────────────────────────────────────────────────
static void exp2_batch_split(const NRData& d, const Jacobian& jac) {
    sep("Exp 2: Batch별 REFACT / SOLVE 분리 타이밍 (IR=0)");

    std::cout << "IR=0 (refinement off) → 순수 triangular 연산 스케일링 확인\n\n";
    std::cout << std::setw(8)  << "Batch"
              << std::setw(16) << "Refact/iter"
              << std::setw(16) << "Solve/iter"
              << std::setw(18) << "Refact/case"
              << std::setw(18) << "Solve/case"
              << std::setw(10) << "S/R"
              << "\n" << std::string(86, '-') << "\n";

    const int WARM = 3, RUNS = 5;

    for (int bs : {1, 2, 4, 8, 16, 32, 64, 128}) {
        CuDSSSolver solver;
        NewtonCudaAccel accel;
        init_pipeline(d, jac, solver, accel, bs, 0);  // IR=0

        for (int w = 0; w < WARM; ++w)
            run_nr_manual(d, solver, accel, bs);

        float rsum = 0, ssum = 0;
        int ni = 0;
        for (int r = 0; r < RUNS; ++r) {
            auto tv = run_nr_manual(d, solver, accel, bs);
            for (auto& t : tv) { rsum += t.refact_ms; ssum += t.solve_ms; }
            ni += (int)tv.size();
        }
        float ri = rsum / ni, si = ssum / ni;

        std::cout << std::setw(8)  << bs
                  << std::setw(16) << std::fixed << std::setprecision(3) << ri
                  << std::setw(16) << std::setprecision(3) << si
                  << std::setw(18) << std::setprecision(3) << ri / bs
                  << std::setw(18) << std::setprecision(3) << si / bs
                  << std::setw(10) << std::setprecision(2) << si / ri
                  << "\n";
    }
    std::cout << "\n해석:\n"
              << "  per-case가 batch 증가에 따라 줄어들면: GPU 병렬화 여지 있음\n"
              << "  per-case가 일찍 포화되면: memory-bandwidth 병목\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// 실험 3: 전체 step breakdown (IR=default, batch=8)
// ─────────────────────────────────────────────────────────────────────────────
static void exp3_breakdown(const NRData& d, const Jacobian& jac) {
    sep("Exp 3: 전체 Step Breakdown (batch=8, IR=default)");

    const int BS = 8, WARM = 3, RUNS = 5;
    CuDSSSolver solver;
    NewtonCudaAccel accel;
    init_pipeline(d, jac, solver, accel, BS);  // default IR

    for (int w = 0; w < WARM; ++w)
        run_nr_manual(d, solver, accel, BS);

    float mis=0, jac_t=0, perm=0, refact=0, solve=0, upv=0;
    int ni = 0;
    for (int r = 0; r < RUNS; ++r) {
        auto tv = run_nr_manual(d, solver, accel, BS);
        for (auto& t : tv) {
            mis   += t.mismatch_ms;
            jac_t += t.jacobian_ms;
            perm  += t.perm_ms;
            refact+= t.refact_ms;
            solve += t.solve_ms;
            upv   += t.updateV_ms;
        }
        ni += (int)tv.size();
    }

    float total = mis + jac_t + perm + refact + solve + upv;
    auto pct = [&](float v) { return total > 0 ? v / total * 100.0f : 0.0f; };
    auto bar = [](float p, int w = 22) {
        int n = std::max(0, std::min((int)(p / 100.0f * w + 0.5f), w));
        std::string s(w, ' ');
        for (int i = 0; i < n; ++i) s[i] = '#';
        return "[" + s + "]";
    };

    std::cout << "iters=" << ni << " (avg=" << std::fixed << std::setprecision(1)
              << (float)ni / RUNS << "/run)\n\n";
    std::cout << std::setw(18) << std::left  << "Step"
              << std::setw(14) << std::right << "Per-iter(ms)"
              << std::setw(8)  << "Pct"
              << "  Bar\n"
              << std::string(66, '-') << "\n";

    auto row = [&](const char* name, float sum) {
        float per = sum / ni;
        float p   = pct(sum);
        std::cout << std::setw(18) << std::left  << name
                  << std::setw(14) << std::right << std::fixed << std::setprecision(3) << per
                  << std::setw(7)  << std::setprecision(1) << p << "%"
                  << "  " << bar(p) << "\n";
    };

    row("Mismatch",    mis);
    row("Jacobian",    jac_t);
    row("Permutation", perm);
    row("REFACT",      refact);
    row("SOLVE",       solve);
    row("UpdateV",     upv);
    std::cout << std::string(66, '-') << "\n";
    std::cout << std::setw(18) << std::left << "Total/iter"
              << std::setw(14) << std::right << std::setprecision(3) << total / ni << "\n";

    std::cout << "\nREFACT/SOLVE ratio: " << std::setprecision(2) << solve / refact << "x\n";
}

#endif // USE_CUDA && USE_CUDSS

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    spdlog::set_level(spdlog::level::warn);
    NewtonCudaAccel::verbose = false;

    std::string case_name = "pglib_opf_case9241_pegase";
    if (argc > 1) case_name = argv[1];

    std::cout << "============================================================\n"
              << "  cuDSS Solver 병목 진단 (REFACT vs SOLVE)\n"
              << "============================================================\n"
              << "Case: " << case_name << "\n";

    NRData d;
    d.load_data(case_name);
    std::cout << "Buses: " << d.V0.size()
              << " (PV=" << d.pv.size() << ", PQ=" << d.pq.size() << ")\n"
              << "Ybus nnz: " << d.Ybus.nonZeros() << "\n";

#if defined(USE_CUDA) && defined(USE_CUDSS)
    // Jacobian 패턴 분석 (모든 실험 공유)
    std::cout << "\nJacobian 패턴 분석...\n";
    Jacobian jac;
    jac.analyze(d.Ybus, d.pv, d.pq);
    std::cout << "J: " << jac.J.rows() << "x" << jac.J.cols()
              << "  nnz=" << jac.J.nonZeros() << "\n";

    // GPU 전체 warm-up (2회)
    std::cout << "GPU warm-up...\n";
    newtonPF(d.Ybus, d.Sbus, d.V0, d.pv, d.pq, 1e-8, 50);
    newtonPF(d.Ybus, d.Sbus, d.V0, d.pv, d.pq, 1e-8, 50);

    exp3_breakdown(d, jac);   // 현재 상태 breakdown
    exp1_ir_steps(d, jac);    // IR 효과 측정
    exp2_batch_split(d, jac); // batch별 분리

#else
    std::cout << "\n[ERROR] USE_CUDA + USE_CUDSS 빌드 필요\n";
#endif

    std::cout << "\n============================================================\n"
              << "  진단 완료\n"
              << "============================================================\n";
    return 0;
}
