/**
 * Verify all 32 converging cases with Mixed Precision GPU solver
 */
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <fstream>

#include "nr_data.hpp"
#include "newtonpf.hpp"
#include "io.hpp"

const std::vector<std::string> CONVERGING_CASES = {
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
    std::cout << std::string(90, '=') << std::endl;
    std::cout << "Mixed Precision (FP32) Convergence Verification" << std::endl;
    std::cout << "Testing all 32 converging cases" << std::endl;
    std::cout << std::string(90, '=') << std::endl << std::endl;

    int total = 0, converged = 0;
    
    for (const auto& case_name : CONVERGING_CASES) {
        std::string mat_path = "/workspace/data/nr/" + case_name + ".mat";
        
        // Check if file exists
        std::ifstream f(mat_path);
        if (!f.good()) {
            std::cout << "[SKIP] " << case_name << ": file not found" << std::endl;
            continue;
        }
        f.close();
        
        try {
            // Load case
            auto [Ybus, Sbus, V0, pv, pq] = load_mat(mat_path);
            int nb = V0.size();
            
            // Run solver
            auto result = newtonPF(Ybus, Sbus, V0, pv, pq, 1e-6, 30);
            
            total++;
            
            std::string status;
            if (result.converged) {
                converged++;
                status = "\033[32m✓\033[0m";
            } else {
                status = "\033[31m✗\033[0m";
            }
            
            std::cout << "[" << status << "] " 
                      << std::setw(40) << std::left << case_name 
                      << " | nb=" << std::setw(5) << nb
                      << " | iter=" << std::setw(2) << result.iter
                      << " | normF=" << std::scientific << std::setprecision(2) << result.normF
                      << std::endl;
                      
        } catch (const std::exception& e) {
            std::cout << "[ERROR] " << case_name << ": " << e.what() << std::endl;
        }
    }
    
    std::cout << std::endl;
    std::cout << std::string(90, '=') << std::endl;
    std::cout << "SUMMARY: " << converged << "/" << total << " cases converged" << std::endl;
    std::cout << std::string(90, '=') << std::endl;
    
    return (converged == total) ? 0 : 1;
}
