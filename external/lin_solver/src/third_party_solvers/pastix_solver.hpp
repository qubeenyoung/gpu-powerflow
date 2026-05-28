#pragma once

#include <memory>

#include "third_party_solvers/linear_solver.hpp"

namespace sparse_direct::solver {

std::unique_ptr<LinearSolver> make_pastix_cpu_solver();
std::unique_ptr<LinearSolver> make_pastix_gpu_solver();

}  // namespace sparse_direct::solver
