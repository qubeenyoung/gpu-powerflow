#pragma once

#include <memory>
#include <string>
#include <vector>

#include "third_party_solvers/linear_solver.hpp"

namespace sparse_direct::solver {

std::vector<std::unique_ptr<LinearSolver>> make_suitesparse_solvers(
    const std::vector<std::string>& requested_names);

}  // namespace sparse_direct::solver
