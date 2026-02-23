#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <complex>
#include <algorithm>
#include <iomanip>

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

int main(int argc, char** argv) {
    cout << "============================================" << endl;
    cout << "   CPU-GPU Hybrid Solver Integration Test" << endl;
    cout << "============================================" << endl;

#ifdef USE_CUDA
    cout << "  Mode: GPU-HYBRID" << endl;
#else
    cout << "  Mode: CPU-ONLY" << endl;
#endif
    cout << "============================================" << endl;

    // 테스트할 케이스 목록
    vector<string> test_cases = {
        "case118_ieee",
        "case1354_pegase",
        "case2869_pegase"
    };

    if (argc > 1) {
        test_cases.clear();
        for (int i = 1; i < argc; ++i) {
            test_cases.push_back(argv[i]);
        }
    }

    int total = 0;
    int passed = 0;
    int failed = 0;

    for (const auto& case_name : test_cases) {
        string dump_path = "/workspace/core/dumps/" + case_name + "/";

        cout << "\n[" << (++total) << "] Testing: " << case_name << endl;

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
            cout << "  [SKIP] Failed to load data: " << e.what() << endl;
            continue;
        }

        cout << "  System: " << V0.size() << " buses, "
             << pv.size() << " PV, " << pq.size() << " PQ" << endl;

        // Newton-Raphson 실행
        cout << "  Running Newton-Raphson solver... " << flush;
        NRResult result = newtonPF(Ybus, Sbus, V0, pv, pq, 1e-8, 10);

        if (result.converged) {
            cout << "\033[0;32mCONVERGED\033[0m" << endl;
            cout << "  Iterations: " << result.iter << endl;
            cout << "  Final normF: " << scientific << setprecision(6) << result.normF << endl;
            passed++;
        } else {
            cout << "\033[0;31mDIVERGED\033[0m" << endl;
            cout << "  Iterations: " << result.iter << endl;
            cout << "  Final normF: " << scientific << setprecision(6) << result.normF << endl;
            failed++;
        }
    }

    // 최종 요약
    cout << "\n============================================" << endl;
    cout << "  Summary" << endl;
    cout << "============================================" << endl;
    cout << "  Total Tests: " << total << endl;
    cout << "  Passed:      " << passed << endl;
    cout << "  Failed:      " << failed << endl;
    cout << "============================================" << endl;

    if (failed == 0 && passed > 0) {
        cout << "\n\033[0;32m✓ All tests PASSED!\033[0m" << endl;
#ifdef USE_CUDA
        cout << "  CPU-GPU hybrid solver is working correctly." << endl;
#else
        cout << "  CPU-only solver is working correctly." << endl;
#endif
        return 0;
    } else {
        cout << "\n\033[0;31m✗ Some tests FAILED!\033[0m" << endl;
        return 1;
    }
}
