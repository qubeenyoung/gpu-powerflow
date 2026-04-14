#pragma once

#include "cupf_dataset_loader.hpp"

#include "newton_solver/core/jacobian_builder.hpp"

#include <cstdint>
#include <vector>

namespace exp_20260409 {

struct PowerFlowLinearSystem {
    JacobianBuilderType jacobian_type = JacobianBuilderType::EdgeBased;
    JacobianStructure structure;
    std::vector<double> values;
    std::vector<double> rhs;
    double mismatch_inf = 0.0;
};

PowerFlowLinearSystem build_linear_system(const CupfDatasetCase& case_data,
                                          JacobianBuilderType jacobian_type);

double rhs_inf_norm(const std::vector<double>& rhs);

double residual_inf_norm(const JacobianStructure& structure,
                         const std::vector<double>& values,
                         const std::vector<double>& rhs,
                         const std::vector<float>& x);

double solution_inf_norm(const std::vector<float>& x);

}  // namespace exp_20260409
