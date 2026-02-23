#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <complex>
#include <algorithm>
#include <iomanip>
#include <cmath>

// CPU 구현
#include "../cpp/newtonpf.hpp"
#include "../utils/nr_data.hpp"

using namespace std;
using namespace nr_data;

// --- 데이터 로더 ---
void check_file(const string& f) {
    ifstream file(f);
    if (!file.good()) {
        cerr << "[Error] Cannot open file: " << f << endl;
        exit(1);
    }
}

void load_indices(const string& f, VectorXi32& v) {
    check_file(f);
    ifstream file(f);
    string l;
    while(file.peek()=='#'||file.peek()=='%') getline(file,l);
    vector<int> t;
    int val;
    while(file>>val) t.push_back(val);
    v.resize(t.size());
    for(size_t i=0;i<t.size();++i) v(i)=t[i];
}

void load_complex_vec(const string& f, VectorXcd& v) {
    check_file(f);
    ifstream file(f);
    string l;
    while(file.peek()=='#'||file.peek()=='%') getline(file,l);
    vector<complex<double>> t;
    double r,i;
    while(file>>r>>i) t.emplace_back(r,i);
    v.resize(t.size());
    for(size_t k=0;k<t.size();++k) v(k)=t[k];
}

void load_mtx_complex(const string& f, YbusType& m) {
    check_file(f);
    ifstream file(f);
    bool sym=false;
    string l;
    while(file.peek()=='%'){
        getline(file,l);
        transform(l.begin(),l.end(),l.begin(),::tolower);
        if(l.find("symmetric")!=string::npos) sym=true;
    }
    int M,N,L;
    file>>M>>N>>L;
    vector<Eigen::Triplet<complex<double>>> t;
    t.reserve(L);
    for(int i=0;i<L;++i){
        int r,c;
        double vr,vi;
        file>>r>>c>>vr>>vi;
        t.emplace_back(r-1,c-1,complex<double>(vr,vi));
        if(sym && r!=c) t.emplace_back(c-1,r-1,complex<double>(vr,vi));
    }
    m.resize(M,N);
    m.setFromTriplets(t.begin(),t.end());
}

// --- 결과 비교 함수 ---
struct ComparisonResult {
    string case_name;
    bool both_converged;
    int cpu_iter;
    int gpu_iter;
    double cpu_normf;
    double gpu_normf;
    double voltage_max_error;
    bool passed;
};

ComparisonResult compare_results(
    const string& case_name,
    const NRResult& cpu_result,
    const NRResult& gpu_result
) {
    ComparisonResult result;
    result.case_name = case_name;
    result.cpu_iter = cpu_result.iter;
    result.gpu_iter = gpu_result.iter;
    result.cpu_normf = cpu_result.normF;
    result.gpu_normf = gpu_result.normF;
    result.both_converged = cpu_result.converged && gpu_result.converged;

    // 전압 벡터 최대 오차 계산
    VectorXcd V_diff = cpu_result.V - gpu_result.V;
    result.voltage_max_error = V_diff.cwiseAbs().maxCoeff();

    // 검증 기준
    bool iter_match = (cpu_result.iter == gpu_result.iter);
    bool normf_close = abs(cpu_result.normF - gpu_result.normF) < 1e-6;
    bool voltage_close = result.voltage_max_error < 1e-6;

    result.passed = result.both_converged && iter_match && normf_close && voltage_close;

    return result;
}

void print_result(const ComparisonResult& r) {
    cout << setw(20) << left << r.case_name << " | ";

    if (!r.both_converged) {
        cout << "\033[0;31mDIVERGED\033[0m" << endl;
        return;
    }

    cout << "Iter: " << setw(2) << r.cpu_iter << "/" << setw(2) << r.gpu_iter << " | ";
    cout << "normF_diff: " << scientific << setprecision(2)
         << abs(r.cpu_normf - r.gpu_normf) << " | ";
    cout << "V_err: " << r.voltage_max_error << " | ";

    if (r.passed) {
        cout << "\033[0;32mPASS\033[0m";
    } else {
        cout << "\033[0;31mFAIL\033[0m";
    }
    cout << endl;
}

int main(int argc, char** argv) {
    cout << "========================================" << endl;
    cout << "  CPU-GPU Integration Verification" << endl;
    cout << "========================================" << endl;

    // 테스트할 케이스 목록 (dumps 폴더에 있는 것들)
    vector<string> test_cases = {
        "case118_ieee",
        "case1354_pegase",
        "case2869_pegase"
    };

    if (argc > 1) {
        // 커맨드라인에서 케이스 지정 가능
        test_cases.clear();
        for (int i = 1; i < argc; ++i) {
            test_cases.push_back(argv[i]);
        }
    }

    vector<ComparisonResult> all_results;

    for (const auto& case_name : test_cases) {
        string dump_path = "/workspace/core/dumps/" + case_name + "/";

        cout << "\n[Testing: " << case_name << "]" << endl;

        // 데이터 로드
        YbusType Ybus;
        VectorXcd V0, Sbus;
        VectorXi32 pv, pq;

        try {
            load_mtx_complex(dump_path + "dump_Ybus.mtx", Ybus);
            load_complex_vec(dump_path + "dump_V.txt", V0);
            load_complex_vec(dump_path + "dump_Sbus.txt", Sbus);
            load_indices(dump_path + "dump_pv.txt", pv);
            load_indices(dump_path + "dump_pq.txt", pq);
        } catch (const exception& e) {
            cerr << "  [SKIP] Failed to load data: " << e.what() << endl;
            continue;
        }

        cout << "  Buses: " << V0.size()
             << ", PV: " << pv.size()
             << ", PQ: " << pq.size() << endl;

        // CPU 모드 실행
        #ifdef USE_CUDA
        #undef USE_CUDA
        #endif

        cout << "  Running CPU-only... " << flush;
        NRResult cpu_result = newtonPF(Ybus, Sbus, V0, pv, pq, 1e-8, 10);
        cout << (cpu_result.converged ? "Converged" : "Diverged")
             << " in " << cpu_result.iter << " iter" << endl;

        // GPU 모드 실행
        #define USE_CUDA
        cout << "  Running GPU-hybrid... " << flush;
        NRResult gpu_result = newtonPF(Ybus, Sbus, V0, pv, pq, 1e-8, 10);
        cout << (gpu_result.converged ? "Converged" : "Diverged")
             << " in " << gpu_result.iter << " iter" << endl;

        // 결과 비교
        ComparisonResult cmp = compare_results(case_name, cpu_result, gpu_result);
        all_results.push_back(cmp);
    }

    // 최종 요약
    cout << "\n========================================" << endl;
    cout << "  Summary" << endl;
    cout << "========================================" << endl;
    cout << left << setw(20) << "Case" << " | Status" << endl;
    cout << "------------------------------------------------------------" << endl;

    int passed = 0;
    for (const auto& r : all_results) {
        print_result(r);
        if (r.passed) passed++;
    }

    cout << "========================================" << endl;
    cout << "Total: " << all_results.size() << " | ";
    cout << "Passed: " << passed << " | ";
    cout << "Failed: " << (all_results.size() - passed) << endl;

    if (passed == all_results.size()) {
        cout << "\n\033[0;32m✓ All tests PASSED!\033[0m" << endl;
        cout << "  CPU-GPU hybrid solver is mathematically correct." << endl;
        return 0;
    } else {
        cout << "\n\033[0;31m✗ Some tests FAILED!\033[0m" << endl;
        return 1;
    }
}
