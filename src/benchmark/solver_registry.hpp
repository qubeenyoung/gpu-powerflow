#pragma once

#include <memory>

#include "third_party_solvers/linear_solver.hpp"

namespace sparse_direct::solver {

// The project's own solver, registered under --solver mysolver.
std::unique_ptr<LinearSolver> make_mysolver_solver();

// GPU multifrontal factor+solve variant (--solver mysolver-gpu), gate-safe with
// CPU fallback. Defined in mysolver_gpu_solver.cu (CUDA).
std::unique_ptr<LinearSolver> make_mysolver_gpu_solver();

}  // namespace sparse_direct::solver
