#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>

#include "newton_solver/core/newton_solver.hpp"
#include "newton_solver/core/newton_solver_types.hpp"

namespace py = pybind11;


// ---------------------------------------------------------------------------
// Helper: numpy CSR 행렬을 YbusViewF64로 감싼다 (zero-copy).
// ---------------------------------------------------------------------------
static YbusViewF64 make_ybus_view(
    py::array_t<int32_t>             indptr,
    py::array_t<int32_t>             indices,
    py::array_t<std::complex<double>> data,
    int32_t rows, int32_t cols)
{
    return YbusViewF64{
        indptr.data(),
        indices.data(),
        data.data(),
        rows, cols,
        static_cast<int32_t>(data.size()),
    };
}


PYBIND11_MODULE(_cupf, m)
{
    m.doc() = "cuPF: GPU 가속 Newton-Raphson 전력조류 solver";

    // -----------------------------------------------------------------------
    // Enum 바인딩
    // -----------------------------------------------------------------------

    py::enum_<BackendKind>(m, "BackendKind",
        "연산 백엔드 선택 (CPU 또는 CUDA)")
        .value("CPU",  BackendKind::CPU)
        .value("CUDA", BackendKind::CUDA)
        .export_values();

    py::enum_<ComputePolicy>(m, "ComputePolicy",
        "내부 계산 정밀도 정책.\n"
        "FP64: 전 단계 FP64.\n"
        "Mixed: public I/O는 FP64, 내부 Jacobian/solve는 FP32 (CUDA 전용, 고정 프로파일).")
        .value("FP64",  ComputePolicy::FP64)
        .value("Mixed", ComputePolicy::Mixed)
        .export_values();

    // -----------------------------------------------------------------------
    // 설정 구조체 바인딩
    // -----------------------------------------------------------------------

    py::class_<NRConfig>(m, "NRConfig",
        "Newton-Raphson 수렴 조건.")
        .def(py::init<>())
        .def_readwrite("tolerance", &NRConfig::tolerance,
            "수렴 판정 기준 (최대 mismatch 노름, 기본값 1e-8)")
        .def_readwrite("max_iter",  &NRConfig::max_iter,
            "최대 반복 횟수 (기본값 50)");

    py::class_<NewtonOptions>(m, "NewtonOptions",
        "solver 생성자에 전달하는 설정.\n"
        "backend와 compute policy만 선택한다.")
        .def(py::init<>())
        .def_readwrite("backend", &NewtonOptions::backend,
            "연산 백엔드 (BackendKind.CPU 또는 BackendKind.CUDA)")
        .def_readwrite("compute", &NewtonOptions::compute,
            "내부 계산 정밀도 정책 (ComputePolicy.FP64 또는 ComputePolicy.Mixed)");

    // -----------------------------------------------------------------------
    // 결과 구조체 바인딩
    // -----------------------------------------------------------------------

    py::class_<NRResultF64>(m, "NRResult",
        "solve() 결과. public I/O는 항상 FP64.")
        .def_readonly("V",              &NRResultF64::V,
            "최종 복소 전압 벡터 [n_bus]")
        .def_readonly("iterations",     &NRResultF64::iterations,
            "실제 수행한 반복 횟수")
        .def_readonly("final_mismatch", &NRResultF64::final_mismatch,
            "최종 mismatch 노름 (수렴 시 tolerance 미만)")
        .def_readonly("converged",      &NRResultF64::converged,
            "수렴 여부");

    // -----------------------------------------------------------------------
    // NewtonSolver 바인딩
    // -----------------------------------------------------------------------

    py::class_<NewtonSolver>(m, "NewtonSolver",
        "Newton-Raphson 전력조류 solver.\n\n"
        "사용 예::\n\n"
        "    solver = NewtonSolver()\n"
        "    solver.analyze(indptr, indices, data, rows, cols, pv, pq)\n"
        "    result = solver.solve(indptr, indices, data, rows, cols, sbus, V0, pv, pq)\n")
        .def(py::init<const NewtonOptions&>(),
             py::arg("options") = NewtonOptions{},
             "solver를 생성한다. options 생략 시 CPU FP64가 기본값이다.")
        // TODO: analyze / solve를 numpy 래퍼로 노출
        ;
}
