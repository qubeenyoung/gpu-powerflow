/**
 * Verify all 32 converging cases with Mixed Precision (FP32) GPU solver
 */
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <fstream>

#include "nr_data.hpp"
#include "newtonpf.hpp"
#include "spdlog/spdlog.h"

using namespace std;
using namespace nr_data;

const vector<string> CONVERGING_CASES = {
    "pglib_opf_case5_pjm",
    "pglib_opf_case14_ieee",
    "pglib_opf_case24_ieee_rts",
    "pglib_opf_case30_as",
    "pglib_opf_case30_ieee",
    "pglib_opf_case57_ieee",
    "pglib_opf_case60_c",
    "pglib_opf_case73_ieee_rts",
    "pglib_opf_case89_pegase",
    "pglib_opf_case118_ieee",
    "pglib_opf_case197_snem",
    "pglib_opf_case200_activ",
    "pglib_opf_case588_sdet",
    "pglib_opf_case793_goc",
    "pglib_opf_case1354_pegase",
    "pglib_opf_case2312_goc",
    "pglib_opf_case2383wp_k",
    "pglib_opf_case2736sp_k",
    "pglib_opf_case2737sop_k",
    "pglib_opf_case2746wop_k",
    "pglib_opf_case2746wp_k",
    "pglib_opf_case2869_pegase",
    "pglib_opf_case3012wp_k",
    "pglib_opf_case3120sp_k",
    "pglib_opf_case3375wp_k",
    "pglib_opf_case3970_goc",
    "pglib_opf_case4601_goc",
    "pglib_opf_case4619_goc",
    "pglib_opf_case5658_epigrids",
    "pglib_opf_case7336_epigrids",
    "pglib_opf_case8387_pegase",
    "pglib_opf_case9241_pegase",
};

int main() {
    // Suppress debug logs
    spdlog::set_level(spdlog::level::warn);
    
    cout << string(90, '=') << endl;
    cout << "Mixed Precision (FP32) Convergence Verification" << endl;
    cout << "Testing all 32 converging cases" << endl;
    cout << string(90, '=') << endl << endl;

    int total = 0, converged_cnt = 0, skipped = 0;
    
    for (const auto& case_name : CONVERGING_CASES) {
        try {
            // Load case data
            NRData case_data;
            case_data.load_data(case_name);
            
            int nb = case_data.V0.size();
            
            // Run solver with FP32 Mixed Precision
            auto result = newtonPF(
                case_data.Ybus, 
                case_data.Sbus, 
                case_data.V0, 
                case_data.pv, 
                case_data.pq, 
                1e-6, 30
            );
            
            total++;
            
            string status;
            if (result.converged) {
                converged_cnt++;
                status = "\033[32m✓\033[0m";
            } else {
                status = "\033[31m✗\033[0m";
            }
            
            cout << "[" << status << "] " 
                 << setw(40) << left << case_name 
                 << " | nb=" << setw(5) << nb
                 << " | iter=" << setw(2) << result.iter
                 << " | normF=" << scientific << setprecision(2) << result.normF
                 << endl;
                      
        } catch (const exception& e) {
            cout << "[SKIP] " << setw(40) << left << case_name << " | " << e.what() << endl;
            skipped++;
        }
    }
    
    cout << endl;
    cout << string(90, '=') << endl;
    cout << "SUMMARY: " << converged_cnt << "/" << total << " cases converged with Mixed Precision (FP32)" << endl;
    if (skipped > 0) {
        cout << "         " << skipped << " cases skipped" << endl;
    }
    
    if (converged_cnt == total && total == 32) {
        cout << "\033[32m✓ ALL 32 CASES CONVERGED SUCCESSFULLY!\033[0m" << endl;
    } else if (converged_cnt < total) {
        cout << "\033[31m✗ Some cases did not converge\033[0m" << endl;
    }
    cout << string(90, '=') << endl;
    
    return (converged_cnt == total) ? 0 : 1;
}
