# GOAL (remove-eigen-cpu-solver)

Remove the Eigen dependency from the cuPF CPU (no-CUDA) backend so it configures
and builds with SuiteSparse KLU only. The CPU linear solver ALREADY uses KLU
(klu_analyze/factor/solve/tsolve). The remaining Eigen usage is:

  * `Eigen::SparseMatrix` for the Jacobian J and Ybus (CSC, ColMajor, int32),
    built via triplets + setFromTriplets/makeCompressed and read via
    outerIndexPtr/innerIndexPtr/valuePtr/nonZeros — replace with a plain CSC
    struct holding std::vector<int32_t> indptr/indices + std::vector<double or
    complex> values, plus a small setFromTriplets-equivalent helper.
  * `Eigen::Map<...>` dense complex/real vector arithmetic in compute_ibus,
    cpu_mismatch, fill_jacobian — replace with plain loops over std::vector /
    std::complex<double>.
  * `Eigen::KLUSupport` + Eigen residual norm in newton_solver_adjoint.cpp —
    use the existing CpuLinearSolveKLU (klu_tsolve) and a manual residual.

Finally remove `find_package(Eigen3 REQUIRED)` and `Eigen3::Eigen` from
CMakeLists.txt.

# SUCCESS METRIC (numeric)

cpu_eigen_ref_files == 0 AND cmake_eigen_refs == 0 AND build_ok AND test_ok.
Eigen is NOT installed in the image, so configure fails until Eigen is fully
gone — the build only goes green once Eigen is gone. Partial, correct progress
that reduces the Eigen count and keeps the design sound is valuable and will be
committed even if the build is not yet green.

# SCOPE

Public headers in cpp/inc are already Eigen-free; do not add Eigen there.
DO NOT modify anything under tests/ — the test `cupf_minimal_tests` is the
honest evaluation and must not be weakened. If you believe the test itself must
change, STOP and request approval instead (the orchestrator gates that).

Keep the numerics identical to the Eigen version. When you can no longer
improve the metric this iteration, STOP and end your final message with 2-4
sentences: WHAT you changed and WHY (used verbatim as the ledger explanation).

# CURRENT STATUS NOTE

This goal may already be substantially or fully complete on the experiment
branch. FIRST inspect the current Eigen usage and run `./tryexp` to see the
live metric. If the metric is already at the target (eigen 0, build+test ok),
do NOT invent changes — say so plainly in your final summary and stop.
