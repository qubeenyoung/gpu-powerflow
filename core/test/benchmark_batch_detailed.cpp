/**
 * @file benchmark_batch_detailed.cpp
 * @brief newtonPF_batch() 단계별 세부 타이밍 분석
 */

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <chrono>
#include <cstdio>

#include "nr_data.hpp"
#include "newtonpf.hpp"
#include "timer.hpp"
#include "spdlog/spdlog.h"
#include "spdlog/sinks/basic_file_sink.h"

using namespace nr_data;
using clk = std::chrono::high_resolution_clock;

static const char* TIMER_LOG = "/tmp/batch_timer_detail.log";

// ============================================================================
struct StepTimings {
    double analyze_jacobian_ms = 0, cuda_init_ms = 0;
    double cudss_analyze_ms = 0,    upload_initial_ms = 0;
    double mismatch_ms = 0, jacobian_ms = 0, permutation_ms = 0;
    double factorize_solve_ms = 0,  update_v_ms = 0;
    int num_iters = 0;

    double init_ms() const  { return analyze_jacobian_ms+cuda_init_ms+cudss_analyze_ms+upload_initial_ms; }
    double iter_ms() const  { return mismatch_ms+jacobian_ms+permutation_ms+factorize_solve_ms+update_v_ms; }
    double total_ms() const { return init_ms()+iter_ms(); }
};

static StepTimings parse_log() {
    StepTimings t;
    std::ifstream f(TIMER_LOG);
    if (!f.is_open()) { std::cerr << "[WARN] log open fail\n"; return t; }
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream ss(line); std::string name; double val;
        if (!(ss >> name >> val)) continue;
        double ms = val * 1000.0;
        if      (name == "Batch_AnalyzeJacobian")               t.analyze_jacobian_ms  += ms;
        else if (name == "Batch_CUDA_Initialize")               t.cuda_init_ms         += ms;
        else if (name == "Batch_cuDSS_AnalyzePattern")          t.cudss_analyze_ms     += ms;
        else if (name == "Batch_UploadInitial")                  t.upload_initial_ms    += ms;
        else if (name.rfind("Batch_Mismatch_",      0)==0) { t.mismatch_ms            += ms; t.num_iters++; }
        else if (name.rfind("Batch_Jacobian_",      0)==0)   t.jacobian_ms            += ms;
        else if (name.rfind("Batch_Permutation_",   0)==0)   t.permutation_ms         += ms;
        else if (name.rfind("Batch_FactorizeSolve_",0)==0)   t.factorize_solve_ms     += ms;
        else if (name.rfind("Batch_UpdateV_",       0)==0)   t.update_v_ms            += ms;
    }
    return t;
}

static void bar(double frac, int w=26) {
    int n=std::max(0,std::min((int)(frac*w+.5),w));
    std::cout<<"["; for(int i=0;i<w;++i) std::cout<<(i<n?'#':' '); std::cout<<"]";
}

// ============================================================================
int main(int argc, char* argv[]) {
    std::string case_name = "pglib_opf_case9241_pegase";
    int batch_size = 8;
    if (argc > 1) case_name  = argv[1];
    if (argc > 2) batch_size = std::stoi(argv[2]);

    spdlog::set_level(spdlog::level::warn);

    std::cout << "========================================================\n"
              << "  newtonPF_batch() 단계별 세부 타이밍\n"
              << "========================================================\n\n"
              << "Case  : " << case_name << "\n"
              << "Batch : " << batch_size << "\n\n";

    NRData d; d.load_data(case_name);
    std::cout << "Buses : " << d.V0.size()
              << " (PV=" << d.pv.size() << ", PQ=" << d.pq.size() << ")\n"
              << "Y nnz : " << d.Ybus.nonZeros() << "\n\n";

    std::vector<VectorXcd> sv(batch_size, d.Sbus), vv(batch_size, d.V0);

    // warm-up (타이머 없이)
    std::cout << "Warm-up...\n";
    newtonPF_batch(d.Ybus, sv, vv, d.pv, d.pq, 1e-8, 50);

    // 타이머 파일 로거 설정 (1회만 생성, 이후 계속 append)
    std::remove(TIMER_LOG);
    {
        auto flog = spdlog::basic_logger_mt("bbd_t", TIMER_LOG, /*truncate=*/true);
        flog->set_pattern("%v");
        flog->flush_on(spdlog::level::info);
        init_timer_logger(flog);
    }

    // 타이밍 수집: 1회 실행
    std::cout << "단계별 타이밍 수집 중 (1회)...\n";
    auto tw0 = clk::now();
    newtonPF_batch(d.Ybus, sv, vv, d.pv, d.pq, 1e-8, 50);
    double wall_ms = std::chrono::duration<double,std::milli>(clk::now()-tw0).count();

    // 로거 flush (spdlog 전역 로거는 유지 — nullptr 설정 안 함)
    if (auto l = spdlog::get("bbd_t")) l->flush();

    // 파싱
    StepTimings t = parse_log();
    int ni  = std::max(t.num_iters, 1);
    double im  = t.iter_ms();
    double tot = t.total_ms();

    std::cout << "\nWall-clock: " << std::fixed << std::setprecision(2)
              << wall_ms << " ms  (per-case: " << wall_ms/batch_size << " ms)\n";

    // ─── 초기화 ───
    std::cout << "\n  ┌─ 초기화 (1회)\n";
    auto pi=[&](const char* name, double ms){
        double pct = tot>0 ? ms/tot*100.0 : 0;
        std::cout << "  │  " << std::setw(26)<<std::left<<name
                  << std::setw(8)<<std::right<<std::setprecision(2)<<ms<<" ms"
                  << "  "<<std::setw(5)<<std::setprecision(1)<<pct<<"% ";
        bar(pct/100.0); std::cout<<"\n";
    };
    pi("AnalyzeJacobian",      t.analyze_jacobian_ms);
    pi("CUDA_Initialize",      t.cuda_init_ms);
    pi("cuDSS_AnalyzePattern", t.cudss_analyze_ms);
    pi("UploadInitial",        t.upload_initial_ms);
    std::cout << "  │  합계: " << std::setprecision(2) << t.init_ms() << " ms\n";

    // ─── 반복 ───
    std::cout << "  └─ 반복 (" << ni << " iter 합계)\n";
    auto pr=[&](const char* name, double ms_tot){
        double pct = im>0 ? ms_tot/im*100.0 : 0;
        std::cout << "     " << std::setw(26)<<std::left<<name
                  << std::setw(8)<<std::right<<std::setprecision(2)<<ms_tot<<" ms"
                  << "  "<<std::setw(5)<<std::setprecision(1)<<pct<<"% ";
        bar(pct/100.0);
        std::cout << "  ["<<std::setprecision(2)<<ms_tot/ni<<" ms/iter]\n";
    };
    pr("Mismatch",       t.mismatch_ms);
    pr("Jacobian(FP32)", t.jacobian_ms);
    pr("Permutation",    t.permutation_ms);
    pr("FactorizeSolve", t.factorize_solve_ms);
    pr("UpdateV",        t.update_v_ms);
    std::cout << "     합계: " << std::setprecision(2) << im << " ms\n"
              << "\n  Grand total (timer): " << std::setprecision(2) << tot
              << " ms  [batch=" << batch_size << ", " << ni << " iters]\n";

    // ─── 타이머 로거 drop (이후엔 stdout fallback) ───
    spdlog::drop("bbd_t");

    // ─── Batch sweep ───
    std::cout << "\n========================================================\n"
              << "  Batch Size Sweep (wall-clock, 5회 평균)\n"
              << "========================================================\n";

    double single_ms = 0;
    {
        std::vector<VectorXcd> sv1{d.Sbus}, vv1{d.V0};
        newtonPF_batch(d.Ybus, sv1, vv1, d.pv, d.pq, 1e-8, 50);
        for (int r=0;r<5;++r) {
            auto t0=clk::now();
            newtonPF(d.Ybus, d.Sbus, d.V0, d.pv, d.pq, 1e-8, 50);
            single_ms += std::chrono::duration<double,std::milli>(clk::now()-t0).count();
        }
        single_ms /= 5;
    }
    std::cout << "  Single-case: " << std::fixed << std::setprecision(2) << single_ms << " ms\n\n";
    std::cout << std::setw(7)<<"Batch"<<std::setw(13)<<"Total(ms)"
              <<std::setw(15)<<"Per-case(ms)"<<std::setw(13)<<"Thruput(/s)"<<std::setw(10)<<"Speedup"
              <<"\n"<<std::string(58,'-')<<"\n";

    for (int bs : {1,2,4,8,16,32}) {
        std::vector<VectorXcd> sv2(bs,d.Sbus), vv2(bs,d.V0);
        newtonPF_batch(d.Ybus,sv2,vv2,d.pv,d.pq,1e-8,50); // warm-up
        double tot2=0;
        for (int r=0;r<5;++r) {
            auto t0=clk::now();
            newtonPF_batch(d.Ybus,sv2,vv2,d.pv,d.pq,1e-8,50);
            tot2 += std::chrono::duration<double,std::milli>(clk::now()-t0).count();
        }
        tot2/=5;
        double per=tot2/bs;
        std::cout<<std::setw(7)<<bs
                 <<std::setw(13)<<std::fixed<<std::setprecision(2)<<tot2
                 <<std::setw(15)<<std::setprecision(2)<<per
                 <<std::setw(12)<<std::setprecision(1)<<1000.0/per<<"/s"
                 <<std::setw(9)<<std::setprecision(2)<<single_ms/per<<"x\n";
    }
    return 0;
}
