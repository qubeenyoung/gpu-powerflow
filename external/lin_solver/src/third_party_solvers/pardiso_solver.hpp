#pragma once

#include <memory>

#include "third_party_solvers/linear_solver.hpp"

namespace sparse_direct::solver {

std::unique_ptr<LinearSolver> make_pardiso_solver();

}  // namespace sparse_direct::solver
