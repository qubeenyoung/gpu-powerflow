// ---------------------------------------------------------------------------
// pybind_cupf.cpp
//
// Python bindings for the cuPF solver (module name: _cupf). Exposes the option
// enums/structs, the NR result structs, and the NewtonSolver class.
//
// Memory / ownership contract:
//   - Array inputs are declared py::array_t<..., c_style | forcecast>, so
//     pybind accepts any numpy array and, if the dtype/layout differs, makes a
//     contiguous converted copy before the call (the .data() pointers handed to
//     the solver are valid only for the call's duration).
//   - The CSR Ybus is wrapped as a *borrowing* YbusView (make_ybus_view) — no
//     copy; the view is only valid while the caller's arrays are alive, which
//     they are for the synchronous solve call (the solver uploads/copies the
//     pattern internally during initialize()/upload()).
//   - Results are returned as owning numpy arrays (the *_to_numpy helpers
//     memcpy the solver's std::vectors out), so they safely outlive the result
//     object. The plain def_readonly members expose the same data as Python
//     lists/copies; the *_numpy properties give zero-extra-copy-after-this numpy
//     views (still owning their own buffer).
//
// The optional torch zero-copy entry points are only declared/bound when
// CUPF_WITH_TORCH is set (their bodies live in torch_cupf_extension.cpp); those
// keep tensors on-device with no host copy.
// ---------------------------------------------------------------------------

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>

#include "newton_solver/core/newton_solver.hpp"
#include "newton_solver/core/newton_solver_types.hpp"

#include <cstring>
#include <cstdint>
#include <stdexcept>

namespace py = pybind11;

#ifdef CUPF_WITH_TORCH
#include <torch/extension.h>
AdjointResult solve_with_adjoint_cache_torch_binding(NewtonSolver& self,
                                                     at::Tensor sbus_base_re,
                                                     at::Tensor sbus_base_im,
                                                     at::Tensor load_p,
                                                     at::Tensor load_q,
                                                     at::Tensor v0_va,
                                                     at::Tensor v0_vm,
                                                     at::Tensor va_out,
                                                     at::Tensor vm_out,
                                                     const NRConfig& config,
                                                     const SolveOptions& solve_options);
AdjointResult solve_adjoint_torch_binding(NewtonSolver& self,
                                          at::Tensor grad_va,
                                          at::Tensor grad_vm,
                                          at::Tensor grad_load_p_out,
                                          at::Tensor grad_load_q_out,
                                          const AdjointOptions& options);
#endif


// ---------------------------------------------------------------------------
// Wrap a numpy CSR matrix as a borrowing YbusView (zero-copy). Validates rank /
// lengths so a malformed CSR is rejected here rather than read out of bounds
// downstream. The returned view borrows the caller's numpy buffers, so it must
// only be used within the same binding call (the solver consumes it before
// returning). nnz is taken from the data length.
// ---------------------------------------------------------------------------
static YbusView make_ybus_view(
    py::array_t<int32_t>             indptr,
    py::array_t<int32_t>             indices,
    py::array_t<std::complex<double>> data,
    int32_t rows, int32_t cols)
{
    if (rows <= 0 || cols <= 0) {
        throw std::invalid_argument("Ybus rows/cols must be positive");
    }
    if (indptr.ndim() != 1 || indices.ndim() != 1 || data.ndim() != 1) {
        throw std::invalid_argument("Ybus indptr, indices, and data must be 1D arrays");
    }
    if (indptr.size() != static_cast<py::ssize_t>(rows + 1)) {  // CSR row pointers: rows+1 entries
        throw std::invalid_argument("Ybus indptr length must be rows + 1");
    }
    if (indices.size() != data.size()) {                        // one column index per value
        throw std::invalid_argument("Ybus indices and data must have the same length");
    }
    return YbusView{
        indptr.data(),
        indices.data(),
        data.data(),
        rows, cols,
        static_cast<int32_t>(data.size()),
    };
}

// --- Result converters: copy solver std::vectors into owning numpy arrays ---
//
// Each returns a freshly allocated numpy array and memcpy's the solver vector
// into it, so the returned array owns its buffer and is safe to keep after the
// result object is gone. 1D variants return [N]; the matrix variants reshape a
// batch-major flat vector into [batch_size, dim] (the layout the solver writes:
// case b occupies the contiguous slice [b*dim, (b+1)*dim)).

static py::array_t<std::complex<double>> complex_vector_to_numpy(
    const std::vector<std::complex<double>>& values)
{
    py::array_t<std::complex<double>> out(values.size());
    std::memcpy(out.mutable_data(),
                values.data(),
                values.size() * sizeof(std::complex<double>));
    return out;
}

// Reshape the batch-major voltage vector [batch_size * n_bus] to [batch, n_bus].
static py::array_t<std::complex<double>> batch_voltage_to_numpy(
    const NRBatchResult& result)
{
    py::array_t<std::complex<double>> out({result.batch_size, result.n_bus});
    if (!result.V.empty()) {
        std::memcpy(out.mutable_data(),
                    result.V.data(),
                    result.V.size() * sizeof(std::complex<double>));
    }
    return out;
}

static py::array_t<int32_t> int_vector_to_numpy(const std::vector<int32_t>& values)
{
    py::array_t<int32_t> out(values.size());
    std::memcpy(out.mutable_data(), values.data(), values.size() * sizeof(int32_t));
    return out;
}

static py::array_t<double> double_vector_to_numpy(const std::vector<double>& values)
{
    py::array_t<double> out(values.size());
    std::memcpy(out.mutable_data(), values.data(), values.size() * sizeof(double));
    return out;
}

static py::array_t<double> double_matrix_to_numpy(
    const std::vector<double>& values,
    int32_t rows,
    int32_t cols)
{
    py::array_t<double> out({rows, cols});
    if (!values.empty()) {
        std::memcpy(out.mutable_data(), values.data(), values.size() * sizeof(double));
    }
    return out;
}

static py::array_t<uint8_t> byte_vector_to_numpy(const std::vector<uint8_t>& values)
{
    py::array_t<uint8_t> out(values.size());
    std::memcpy(out.mutable_data(), values.data(), values.size() * sizeof(uint8_t));
    return out;
}


PYBIND11_MODULE(_cupf, m)
{
    m.doc() = "cuPF: GPU 가속 Newton-Raphson 전력조류 solver";

    // -----------------------------------------------------------------------
    // Enum 바인딩
    // -----------------------------------------------------------------------

    // --- Configuration enums ---
    py::enum_<BackendKind>(m, "BackendKind",
        "연산 백엔드 선택 (CPU 또는 CUDA)")
        .value("CPU",  BackendKind::CPU)
        .value("CUDA", BackendKind::CUDA)
        .export_values();

    py::enum_<ComputePolicy>(m, "ComputePolicy",
        "내부 계산 정밀도 정책.\n"
        "FP64: 전 단계 FP64.\n"
        "FP32: CUDA 내부 numeric buffers/operators are FP32.\n"
        "Mixed: public I/O는 FP64, 내부 Jacobian/solve는 FP32 (CUDA 전용, 고정 프로파일).")
        .value("FP64",  ComputePolicy::FP64)
        .value("FP32",  ComputePolicy::FP32)
        .value("Mixed", ComputePolicy::Mixed)
        .export_values();

    py::enum_<CudaLinearSolverKind>(m, "CudaLinearSolverKind",
        "CUDA FP64 linear solver backend.")
        .value("CuDSS",  CudaLinearSolverKind::CuDSS)
        .value("Custom", CudaLinearSolverKind::Custom)
        .export_values();

    py::enum_<CpuJacobianKind>(m, "CpuJacobianKind",
        "CPU Jacobian fill algorithm.")
        .value("Native", CpuJacobianKind::Native)
        .value("Pandapower", CpuJacobianKind::Pandapower)
        .export_values();

    py::enum_<CpuLinearSolverKind>(m, "CpuLinearSolverKind",
        "CPU sparse linear solver backend.")
        .value("KLU", CpuLinearSolverKind::KLU)
        .value("UMFPACK", CpuLinearSolverKind::UMFPACK)
        .export_values();

    py::enum_<CudaJacobianKind>(m, "CudaJacobianKind",
        "CUDA Jacobian fill algorithm.")
        .value("Edge", CudaJacobianKind::Edge)
        .value("EdgeAtomic", CudaJacobianKind::EdgeAtomic)
        .value("VertexWarp", CudaJacobianKind::VertexWarp)
        .export_values();

    py::enum_<AdjointCacheMode>(m, "AdjointCacheMode",
        "forward 종료 시 adjoint cache 준비 방식.")
        .value("None", AdjointCacheMode::None)
        .value("FinalStateFactorization", AdjointCacheMode::FinalStateFactorization)
        .value("ReuseLastNewtonFactorizationIfExact", AdjointCacheMode::ReuseLastNewtonFactorizationIfExact)
        .export_values();

    py::enum_<CuDSSAlgorithm>(m, "CuDSSAlgorithm",
        "cuDSS 알고리즘 선택.")
        .value("DEFAULT", CuDSSAlgorithm::Default)
        .value("ALG_1",   CuDSSAlgorithm::Alg1)
        .value("ALG_2",   CuDSSAlgorithm::Alg2)
        .value("ALG_3",   CuDSSAlgorithm::Alg3)
        .value("ALG_4",   CuDSSAlgorithm::Alg4)
        .value("ALG_5",   CuDSSAlgorithm::Alg5)
        .export_values();

    // -----------------------------------------------------------------------
    // 설정 구조체 바인딩
    // -----------------------------------------------------------------------

    // --- Option / config structs ---
    py::class_<NRConfig>(m, "NRConfig",
        "Newton-Raphson 수렴 조건.")
        .def(py::init<>())
        .def_readwrite("tolerance", &NRConfig::tolerance,
            "수렴 판정 기준 (최대 mismatch 노름, 기본값 1e-8)")
        .def_readwrite("max_iter",  &NRConfig::max_iter,
            "최대 반복 횟수 (기본값 50)");

    py::class_<CuDSSOptions>(m, "CuDSSOptions",
        "CUDA cuDSS direct solver 런타임 설정.")
        .def(py::init<>())
        .def_readwrite("use_matching", &CuDSSOptions::use_matching,
            "cuDSS matching 전처리 활성화")
        .def_readwrite("matching_alg", &CuDSSOptions::matching_alg,
            "matching 알고리즘")
        .def_readwrite("auto_pivot_epsilon", &CuDSSOptions::auto_pivot_epsilon,
            "True이면 cuDSS 기본 pivot epsilon 사용")
        .def_readwrite("pivot_epsilon", &CuDSSOptions::pivot_epsilon,
            "auto_pivot_epsilon=False일 때 사용할 pivot epsilon");

    py::class_<NewtonOptions>(m, "NewtonOptions",
        "solver 생성자에 전달하는 설정.\n"
        "backend, compute policy, cuDSS 옵션을 선택한다.")
        .def(py::init<>())
        .def_readwrite("backend", &NewtonOptions::backend,
            "연산 백엔드 (BackendKind.CPU 또는 BackendKind.CUDA)")
        .def_readwrite("compute", &NewtonOptions::compute,
            "내부 계산 정밀도 정책 (ComputePolicy.FP64, ComputePolicy.FP32 또는 ComputePolicy.Mixed)")
        .def_readwrite("cpu_jacobian", &NewtonOptions::cpu_jacobian,
            "CPU Jacobian fill algorithm (CpuJacobianKind.Native 또는 Pandapower)")
        .def_readwrite("cpu_linear_solver", &NewtonOptions::cpu_linear_solver,
            "CPU sparse linear solver backend (CpuLinearSolverKind.KLU 또는 UMFPACK)")
        .def_readwrite("cuda_jacobian", &NewtonOptions::cuda_jacobian,
            "CUDA Jacobian fill algorithm (CudaJacobianKind.Edge, EdgeAtomic, VertexWarp)")
        .def_readwrite("cuda_linear_solver", &NewtonOptions::cuda_linear_solver,
            "CUDA FP64 linear solver backend (CudaLinearSolverKind.CuDSS 또는 Custom)")
        .def_readwrite("cudss", &NewtonOptions::cudss,
            "CUDA direct solver 런타임 설정");

    py::class_<SolveOptions>(m, "SolveOptions",
        "forward solve 옵션.")
        .def(py::init<>())
        .def_readwrite("prepare_adjoint_cache", &SolveOptions::prepare_adjoint_cache,
            "True이면 forward 종료 시 final-state adjoint cache를 준비")
        .def_readwrite("adjoint_cache_mode", &SolveOptions::adjoint_cache_mode,
            "adjoint cache 준비 방식")
        .def_readwrite("allow_explicit_transpose_fallback", &SolveOptions::allow_explicit_transpose_fallback,
            "cuDSS transpose solve 미지원 시 explicit J^T cache fallback 허용");

    py::class_<AdjointOptions>(m, "AdjointOptions",
        "Implicit power-flow backward 옵션.")
        .def(py::init<>())
        .def_readwrite("reuse_forward_factorization", &AdjointOptions::reuse_forward_factorization,
            "forward factorization 재사용 요청. 현재 final-state backward는 refactorization을 사용한다.")
        .def_readwrite("allow_refactorize", &AdjointOptions::allow_refactorize,
            "final state에서 Jacobian을 재조립하고 transpose solve를 위한 재인수분해를 허용")
        .def_readwrite("require_cached_factorization", &AdjointOptions::require_cached_factorization,
            "True이면 exact final-state adjoint cache가 없을 때 에러")
        .def_readwrite("allow_refactorize_for_backward", &AdjointOptions::allow_refactorize_for_backward,
            "cache miss 때 solve_adjoint 내부 refactorization 허용")
        .def_readwrite("allow_inexact_last_newton_factorization", &AdjointOptions::allow_inexact_last_newton_factorization,
            "마지막 Newton factorization이 final state와 다를 수 있는 경우의 부정확 재사용 허용")
        .def_readwrite("use_transpose_solve", &AdjointOptions::use_transpose_solve,
            "가능하면 같은 factorization의 transpose solve 사용")
        .def_readwrite("allow_explicit_transpose_fallback", &AdjointOptions::allow_explicit_transpose_fallback,
            "native transpose solve가 없을 때 explicit J^T cached factorization fallback 허용")
        .def_readwrite("compute_load_gradients", &AdjointOptions::compute_load_gradients,
            "load_p/load_q gradient projection 계산 여부")
        .def_readwrite("check_residual", &AdjointOptions::check_residual,
            "J^T lambda = grad_state 상대 residual 계산 여부");

    // -----------------------------------------------------------------------
    // 결과 구조체 바인딩
    // -----------------------------------------------------------------------

    // --- Result structs (returned to Python) ---
    py::class_<NRResult>(m, "NRResult",
        "solve() 결과. public I/O는 항상 FP64.")
        .def_readonly("V",              &NRResult::V,
            "최종 복소 전압 벡터 [n_bus]")
        .def_property_readonly("V_numpy",
            [](const NRResult& result) { return complex_vector_to_numpy(result.V); },
            "최종 복소 전압 벡터 [n_bus]를 numpy complex128 배열로 반환")
        .def_readonly("iterations",     &NRResult::iterations,
            "실제 수행한 반복 횟수")
        .def_readonly("final_mismatch", &NRResult::final_mismatch,
            "최종 mismatch 노름 (수렴 시 tolerance 미만)")
        .def_readonly("converged",      &NRResult::converged,
            "수렴 여부");

    py::class_<NRBatchResult>(m, "NRBatchResult",
        "solve_batch() 결과. V는 batch-major [batch_size, n_bus] layout이다.")
        .def_readonly("V",              &NRBatchResult::V,
            "최종 복소 전압 벡터 [batch_size * n_bus]")
        .def_property_readonly("V_numpy", &batch_voltage_to_numpy,
            "최종 복소 전압을 numpy complex128 배열 [batch_size, n_bus]로 반환")
        .def_readonly("n_bus",          &NRBatchResult::n_bus,
            "버스 수")
        .def_readonly("batch_size",     &NRBatchResult::batch_size,
            "배치 크기")
        .def_readonly("iterations",     &NRBatchResult::iterations,
            "배치별 실제 수행 반복 횟수")
        .def_property_readonly("iterations_numpy",
            [](const NRBatchResult& result) { return int_vector_to_numpy(result.iterations); },
            "배치별 반복 횟수를 numpy int32 배열로 반환")
        .def_readonly("final_mismatch", &NRBatchResult::final_mismatch,
            "배치별 최종 mismatch 노름")
        .def_property_readonly("final_mismatch_numpy",
            [](const NRBatchResult& result) { return double_vector_to_numpy(result.final_mismatch); },
            "배치별 최종 mismatch를 numpy float64 배열로 반환")
        .def_readonly("converged",      &NRBatchResult::converged,
            "배치별 수렴 여부")
        .def_property_readonly("converged_numpy",
            [](const NRBatchResult& result) { return byte_vector_to_numpy(result.converged); },
            "배치별 수렴 여부를 numpy uint8 배열로 반환");

    py::class_<AdjointResult>(m, "AdjointResult",
        "solve_adjoint() 결과. lambda와 load gradients는 batch-major layout이다.")
        .def_readonly("lambda", &AdjointResult::lambda,
            "adjoint 변수 lambda [batch_size * dimF]. 주의: 'lambda'는 파이썬 예약어라 "
            "result.lambda 로 직접 접근할 수 없다 — lambda_ / lambda_numpy 또는 "
            "getattr(result, 'lambda') 를 사용한다.")
        .def_readonly("lambda_", &AdjointResult::lambda,
            "lambda의 키워드-안전 별칭 ('lambda'는 파이썬 예약어). 동일한 [batch_size * dimF].")
        .def_property_readonly("lambda_numpy",
            [](const AdjointResult& result) {
                return double_matrix_to_numpy(result.lambda, result.batch_size, result.dimF);
            },
            "lambda를 numpy float64 배열 [batch_size, dimF]로 반환")
        .def_readonly("grad_load_p", &AdjointResult::grad_load_p,
            "load P demand gradient [batch_size * n_bus]")
        .def_property_readonly("grad_load_p_numpy",
            [](const AdjointResult& result) {
                return double_matrix_to_numpy(result.grad_load_p, result.batch_size, result.n_bus);
            },
            "load P demand gradient를 numpy float64 배열 [batch_size, n_bus]로 반환")
        .def_readonly("grad_load_q", &AdjointResult::grad_load_q,
            "load Q demand gradient [batch_size * n_bus]")
        .def_property_readonly("grad_load_q_numpy",
            [](const AdjointResult& result) {
                return double_matrix_to_numpy(result.grad_load_q, result.batch_size, result.n_bus);
            },
            "load Q demand gradient를 numpy float64 배열 [batch_size, n_bus]로 반환")
        .def_readonly("n_bus", &AdjointResult::n_bus)
        .def_readonly("batch_size", &AdjointResult::batch_size)
        .def_readonly("dimF", &AdjointResult::dimF)
        .def_readonly("success", &AdjointResult::success)
        .def_readonly("used_adjoint_cache", &AdjointResult::used_adjoint_cache)
        .def_readonly("adjoint_cache_matches_final_state", &AdjointResult::adjoint_cache_matches_final_state)
        .def_readonly("reused_forward_factorization", &AdjointResult::reused_forward_factorization)
        .def_readonly("reused_final_state_factorization", &AdjointResult::reused_final_state_factorization)
        .def_readonly("refactorized_for_backward", &AdjointResult::refactorized_for_backward)
        .def_readonly("used_explicit_transpose", &AdjointResult::used_explicit_transpose)
        .def_readonly("used_python_scipy", &AdjointResult::used_python_scipy)
        .def_readonly("includes_host_device_transfer", &AdjointResult::includes_host_device_transfer)
        .def_readonly("zero_copy", &AdjointResult::zero_copy)
        .def_readonly("torch_extension_zero_copy", &AdjointResult::torch_extension_zero_copy)
        .def_readonly("raw_pointer_api_used", &AdjointResult::raw_pointer_api_used)
        .def_readonly("current_stream_integrated", &AdjointResult::current_stream_integrated)
        .def_readonly("jt_symbolic_analyzed_at_initialize", &AdjointResult::jt_symbolic_analyzed_at_initialize)
        .def_readonly("jt_values_transposed_on_device", &AdjointResult::jt_values_transposed_on_device)
        .def_readonly("jt_factorized_during_forward_cache", &AdjointResult::jt_factorized_during_forward_cache)
        .def_readonly("jt_refactorized_during_backward", &AdjointResult::jt_refactorized_during_backward)
        .def_readonly("host_roundtrip_for_jt_transpose", &AdjointResult::host_roundtrip_for_jt_transpose)
        .def_readonly("jt_residual_norm", &AdjointResult::jt_residual_norm)
        .def_readonly("solve_time_ms", &AdjointResult::solve_time_ms)
        .def_readonly("transpose_solve_time_ms", &AdjointResult::transpose_solve_time_ms)
        .def_readonly("factorization_time_ms", &AdjointResult::factorization_time_ms)
        .def_readonly("total_time_ms", &AdjointResult::total_time_ms)
        .def_readonly("backend", &AdjointResult::backend)
        .def_readonly("transpose_solve_backend", &AdjointResult::transpose_solve_backend)
        .def_readonly("sign_convention", &AdjointResult::sign_convention);

    // -----------------------------------------------------------------------
    // NewtonSolver 바인딩
    // -----------------------------------------------------------------------

    // --- Solver class (initialize -> solve/solve_batch -> solve_adjoint) ---
    py::class_<NewtonSolver>(m, "NewtonSolver",
        "Newton-Raphson 전력조류 solver.\n\n"
        "사용 예::\n\n"
        "    solver = NewtonSolver()\n"
        "    solver.initialize(indptr, indices, data, rows, cols, pv, pq)\n"
        "    result = solver.solve(indptr, indices, data, rows, cols, sbus, V0, pv, pq)\n")
        .def(py::init<const NewtonOptions&>(),
             py::arg("options") = NewtonOptions{},
             "solver를 생성한다. options 생략 시 CPU FP64가 기본값이다.")
        .def("initialize",
             [](NewtonSolver& self,
                py::array_t<int32_t, py::array::c_style | py::array::forcecast> indptr,
                py::array_t<int32_t, py::array::c_style | py::array::forcecast> indices,
                py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> data,
                int32_t rows,
                int32_t cols,
                py::array_t<int32_t, py::array::c_style | py::array::forcecast> pv,
                py::array_t<int32_t, py::array::c_style | py::array::forcecast> pq) {
                 const YbusView ybus = make_ybus_view(indptr, indices, data, rows, cols);
                 self.initialize(ybus,
                                 pv.data(), static_cast<int32_t>(pv.size()),
                                 pq.data(), static_cast<int32_t>(pq.size()));
             },
             py::arg("indptr"),
             py::arg("indices"),
             py::arg("data"),
             py::arg("rows"),
             py::arg("cols"),
             py::arg("pv"),
             py::arg("pq"),
             "CSR Ybus 구조와 PV/PQ index로 solver symbolic state를 초기화한다.")
        .def("solve",
             [](NewtonSolver& self,
                py::array_t<int32_t, py::array::c_style | py::array::forcecast> indptr,
                py::array_t<int32_t, py::array::c_style | py::array::forcecast> indices,
                py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> data,
                int32_t rows,
                int32_t cols,
                py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> sbus,
                py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> v0,
                py::array_t<int32_t, py::array::c_style | py::array::forcecast> pv,
                py::array_t<int32_t, py::array::c_style | py::array::forcecast> pq,
                const NRConfig& config,
                const SolveOptions& solve_options) {
                 if (sbus.ndim() != 1 || v0.ndim() != 1) {
                     throw std::invalid_argument("sbus and v0 must be 1D arrays for solve()");
                 }
                 if (sbus.size() != rows || v0.size() != rows) {
                     throw std::invalid_argument("sbus/v0 length must match Ybus rows");
                 }
                 const YbusView ybus = make_ybus_view(indptr, indices, data, rows, cols);
                 NRResult result;
                 self.solve(ybus,
                            sbus.data(),
                            v0.data(),
                            pv.data(), static_cast<int32_t>(pv.size()),
                            pq.data(), static_cast<int32_t>(pq.size()),
                            config,
                            solve_options,
                            result);
                 return result;
             },
             py::arg("indptr"),
             py::arg("indices"),
             py::arg("data"),
             py::arg("rows"),
             py::arg("cols"),
             py::arg("sbus"),
             py::arg("v0"),
             py::arg("pv"),
             py::arg("pq"),
             py::arg("config") = NRConfig{},
             py::arg("solve_options") = SolveOptions{},
             "단일 scenario Newton-Raphson solve를 실행한다. "
             "pv/pq는 initialize()에 전달한 것과 동일해야 한다(분석된 symbolic state와 일치).")
        .def("solve_batch",
             [](NewtonSolver& self,
                py::array_t<int32_t, py::array::c_style | py::array::forcecast> indptr,
                py::array_t<int32_t, py::array::c_style | py::array::forcecast> indices,
                py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> data,
                int32_t rows,
                int32_t cols,
                py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> sbus,
                py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> v0,
                py::array_t<int32_t, py::array::c_style | py::array::forcecast> pv,
                py::array_t<int32_t, py::array::c_style | py::array::forcecast> pq,
                const NRConfig& config,
                const SolveOptions& solve_options) {
                 if (sbus.ndim() != 2 || v0.ndim() != 2) {
                     throw std::invalid_argument("sbus and v0 must be 2D arrays [batch, n_bus] for solve_batch()");
                 }
                 if (sbus.shape(0) != v0.shape(0) || sbus.shape(1) != rows || v0.shape(1) != rows) {
                     throw std::invalid_argument("sbus/v0 shape must be [batch, rows]");
                 }
                 const YbusView ybus = make_ybus_view(indptr, indices, data, rows, cols);
                 NRBatchResult result;
                 self.solve_batch(ybus,
                                  sbus.data(),
                                  rows,
                                  v0.data(),
                                  rows,
                                  static_cast<int32_t>(sbus.shape(0)),
                                  pv.data(), static_cast<int32_t>(pv.size()),
                                  pq.data(), static_cast<int32_t>(pq.size()),
                                  config,
                                  solve_options,
                                  result);
                 return result;
             },
             py::arg("indptr"),
             py::arg("indices"),
             py::arg("data"),
             py::arg("rows"),
             py::arg("cols"),
             py::arg("sbus"),
             py::arg("v0"),
             py::arg("pv"),
             py::arg("pq"),
             py::arg("config") = NRConfig{},
             py::arg("solve_options") = SolveOptions{},
             "batch-major scenario 배열 [batch, n_bus]에 대해 solve_batch를 실행한다. "
             "batch_size>1은 CUDA FP32/Mixed/FP64에서 지원된다(CPU는 단일 케이스). "
             "pv/pq는 initialize()에 전달한 것과 동일해야 한다.")
        .def("solve_adjoint",
             [](NewtonSolver& self,
                py::array_t<double, py::array::c_style | py::array::forcecast> grad_va,
                py::array_t<double, py::array::c_style | py::array::forcecast> grad_vm,
                py::array_t<int32_t, py::array::c_style | py::array::forcecast> pv,
                py::array_t<int32_t, py::array::c_style | py::array::forcecast> pq,
                const AdjointOptions& options) {
                 if (grad_va.ndim() != grad_vm.ndim()) {
                     throw std::invalid_argument("grad_va and grad_vm must have the same rank");
                 }
                 int32_t batch_size = 1;
                 int32_t n_bus = 0;
                 if (grad_va.ndim() == 1) {
                     if (grad_vm.ndim() != 1 || grad_va.shape(0) != grad_vm.shape(0)) {
                         throw std::invalid_argument("1D grad_va/grad_vm shapes must match");
                     }
                     n_bus = static_cast<int32_t>(grad_va.shape(0));
                 } else if (grad_va.ndim() == 2) {
                     if (grad_va.shape(0) != grad_vm.shape(0) ||
                         grad_va.shape(1) != grad_vm.shape(1)) {
                         throw std::invalid_argument("2D grad_va/grad_vm shapes must match");
                     }
                     batch_size = static_cast<int32_t>(grad_va.shape(0));
                     n_bus = static_cast<int32_t>(grad_va.shape(1));
                 } else {
                     throw std::invalid_argument("grad_va and grad_vm must be 1D [n_bus] or 2D [batch, n_bus]");
                 }
                 AdjointResult result;
                 self.solve_adjoint(grad_va.data(),
                                    n_bus,
                                    grad_vm.data(),
                                    n_bus,
                                    batch_size,
                                    pv.data(), static_cast<int32_t>(pv.size()),
                                    pq.data(), static_cast<int32_t>(pq.size()),
                                    options,
                                    result);
                 return result;
             },
             py::arg("grad_va"),
             py::arg("grad_vm"),
             py::arg("pv"),
             py::arg("pq"),
             py::arg("options") = AdjointOptions{},
             "마지막 forward state에서 native adjoint solve J^T lambda = dL/dx를 실행한다. "
             "grad_va/grad_vm은 full-bus layout이며 1D 또는 batch-major 2D 배열이다. "
             "pv/pq는 initialize()/직전 forward와 동일해야 하고, batch_size도 직전 forward와 일치해야 한다.")
#ifdef CUPF_WITH_TORCH
        .def("solve_with_adjoint_cache_torch",
             &solve_with_adjoint_cache_torch_binding,
             py::arg("sbus_base_re"),
             py::arg("sbus_base_im"),
             py::arg("load_p"),
             py::arg("load_q"),
             py::arg("v0_va"),
             py::arg("v0_vm"),
             py::arg("va_out"),
             py::arg("vm_out"),
             py::arg("config") = NRConfig{},
             py::arg("solve_options") = SolveOptions{},
             "torch::Tensor CUDA zero-copy forward path. Dynamic load tensors and output tensors stay on device.\n"
             "모든 텐서 dtype은 compute policy와 일치해야 한다: FP64 -> torch.float64, "
             "FP32/Mixed -> torch.float32 (Mixed는 비-torch 경로의 FP64 입력과 달리 float32 텐서를 받는다). "
             "load_p/load_q/va_out/vm_out은 [batch, n_bus], sbus_base_*/v0_*는 [n_bus] (배치 공통).")
        .def("solve_torch",
             &solve_with_adjoint_cache_torch_binding,
             py::arg("sbus_base_re"),
             py::arg("sbus_base_im"),
             py::arg("load_p"),
             py::arg("load_q"),
             py::arg("v0_va"),
             py::arg("v0_vm"),
             py::arg("va_out"),
             py::arg("vm_out"),
             py::arg("config") = NRConfig{},
             py::arg("solve_options") = SolveOptions{},
             "Alias for solve_with_adjoint_cache_torch; set solve_options.prepare_adjoint_cache as needed.\n"
             "텐서 dtype 규칙은 solve_with_adjoint_cache_torch와 동일: FP64 -> float64, FP32/Mixed -> float32.")
        .def("solve_adjoint_torch",
             &solve_adjoint_torch_binding,
             py::arg("grad_va"),
             py::arg("grad_vm"),
             py::arg("grad_load_p_out"),
             py::arg("grad_load_q_out"),
             py::arg("options") = AdjointOptions{},
             "torch::Tensor CUDA zero-copy cached adjoint path using PyTorch current CUDA stream.\n"
             "grad_va/grad_vm/grad_load_*_out은 모두 [batch, n_bus]이고 dtype은 forward와 동일해야 한다 "
             "(FP64 -> float64, FP32/Mixed -> float32). batch는 직전 forward solve와 일치해야 한다.")
#endif
        ;
}
