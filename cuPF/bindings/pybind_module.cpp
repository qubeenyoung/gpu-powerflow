#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>

#include "newton_solver/core/newton_solver.hpp"

namespace py = pybind11;


// ---------------------------------------------------------------------------
// make_ybus_view: wrap numpy arrays as a YbusView (zero-copy).
//
// The caller is responsible for keeping the numpy arrays alive while the
// YbusView is in use. pybind11 holds a reference via the py::array_t args.
//
// Expected layout:
//   indptr:  int32, shape (rows+1,)
//   indices: int32, shape (nnz,)
//   data:    complex128, shape (nnz,)
// ---------------------------------------------------------------------------
static YbusView make_ybus_view(
    const py::array_t<int32_t>&              indptr,
    const py::array_t<int32_t>&              indices,
    const py::array_t<std::complex<double>>& data,
    int32_t rows, int32_t cols)
{
    YbusView view;
    view.indptr  = indptr.data();
    view.indices = indices.data();
    view.data    = data.data();
    view.rows    = rows;
    view.cols    = cols;
    view.nnz     = static_cast<int32_t>(data.size());
    return view;
}


// ---------------------------------------------------------------------------
// PyNewtonSolver: thin pybind11 wrapper around NewtonSolver.
//
// Exposes analyze(), solve(), and solve_batch() using numpy arrays.
// solve() returns a Python dict with V, iterations, final_mismatch, converged.
// ---------------------------------------------------------------------------
struct PyNewtonSolver {
    NewtonSolver solver;

    explicit PyNewtonSolver(const std::string& backend_str,
                            const std::string& jacobian_str,
                            int32_t            n_batch = 1)
    {
        NewtonOptions opts;

        if      (backend_str == "cpu")  opts.backend = BackendKind::CPU;
        else if (backend_str == "cuda") opts.backend = BackendKind::CUDA;
        else throw std::invalid_argument("backend must be 'cpu' or 'cuda'");

        if      (jacobian_str == "edge_based")   opts.jacobian = JacobianBuilderType::EdgeBased;
        else if (jacobian_str == "vertex_based") opts.jacobian = JacobianBuilderType::VertexBased;
        else throw std::invalid_argument("jacobian must be 'edge_based' or 'vertex_based'");

        opts.n_batch = n_batch;
        solver = NewtonSolver(opts);
    }

    void analyze(
        const py::array_t<int32_t>&              indptr,
        const py::array_t<int32_t>&              indices,
        const py::array_t<std::complex<double>>& data,
        int32_t rows, int32_t cols,
        const py::array_t<int32_t>&              pv,
        const py::array_t<int32_t>&              pq)
    {
        YbusView ybus = make_ybus_view(indptr, indices, data, rows, cols);
        solver.analyze(ybus,
                       pv.data(), static_cast<int32_t>(pv.size()),
                       pq.data(), static_cast<int32_t>(pq.size()));
    }

    py::dict solve(
        const py::array_t<int32_t>&              indptr,
        const py::array_t<int32_t>&              indices,
        const py::array_t<std::complex<double>>& data,
        int32_t rows, int32_t cols,
        const py::array_t<std::complex<double>>& sbus,
        const py::array_t<std::complex<double>>& V0,
        const py::array_t<int32_t>&              pv,
        const py::array_t<int32_t>&              pq,
        double  tolerance = 1e-8,
        int32_t max_iter  = 50)
    {
        YbusView ybus = make_ybus_view(indptr, indices, data, rows, cols);
        NRConfig config{tolerance, max_iter};
        NRResult result;

        solver.solve(ybus,
                     sbus.data(), V0.data(),
                     pv.data(), static_cast<int32_t>(pv.size()),
                     pq.data(), static_cast<int32_t>(pq.size()),
                     config, result);

        py::array_t<std::complex<double>> V_arr(
            {static_cast<py::ssize_t>(result.V.size())});
        std::copy(result.V.begin(), result.V.end(), V_arr.mutable_data());

        py::dict out;
        out["V"]              = V_arr;
        out["iterations"]     = result.iterations;
        out["final_mismatch"] = result.final_mismatch;
        out["converged"]      = result.converged;
        return out;
    }

    py::list solve_batch(
        const py::array_t<int32_t>&              indptr,
        const py::array_t<int32_t>&              indices,
        const py::array_t<std::complex<double>>& data,
        int32_t rows, int32_t cols,
        const py::array_t<std::complex<double>>& sbus_batch,
        const py::array_t<std::complex<double>>& V0_batch,
        const py::array_t<int32_t>&              pv,
        const py::array_t<int32_t>&              pq,
        int32_t n_batch,
        double  tolerance = 1e-8,
        int32_t max_iter  = 50)
    {
        YbusView ybus = make_ybus_view(indptr, indices, data, rows, cols);
        NRConfig config{tolerance, max_iter};

        std::vector<NRResult> results(n_batch);
        solver.solve_batch(ybus,
                           sbus_batch.data(), V0_batch.data(),
                           pv.data(), static_cast<int32_t>(pv.size()),
                           pq.data(), static_cast<int32_t>(pq.size()),
                           n_batch, config, results.data());

        py::list out;
        for (auto& r : results) {
            py::array_t<std::complex<double>> V_arr(
                {static_cast<py::ssize_t>(r.V.size())});
            std::copy(r.V.begin(), r.V.end(), V_arr.mutable_data());

            py::dict d;
            d["V"]              = V_arr;
            d["iterations"]     = r.iterations;
            d["final_mismatch"] = r.final_mismatch;
            d["converged"]      = r.converged;
            out.append(d);
        }
        return out;
    }
};


// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------
PYBIND11_MODULE(cupf, m) {
    m.doc() = "cuPF: GPU-accelerated Newton-Raphson power flow solver";

    py::class_<PyNewtonSolver>(m, "NewtonSolver")
        .def(py::init<const std::string&, const std::string&, int32_t>(),
             py::arg("backend")  = "cpu",
             py::arg("jacobian") = "edge_based",
             py::arg("n_batch")  = 1)
        .def("analyze", &PyNewtonSolver::analyze,
             py::arg("indptr"), py::arg("indices"), py::arg("data"),
             py::arg("rows"),   py::arg("cols"),
             py::arg("pv"),     py::arg("pq"))
        .def("solve", &PyNewtonSolver::solve,
             py::arg("indptr"), py::arg("indices"), py::arg("data"),
             py::arg("rows"),   py::arg("cols"),
             py::arg("sbus"),   py::arg("V0"),
             py::arg("pv"),     py::arg("pq"),
             py::arg("tolerance") = 1e-8,
             py::arg("max_iter")  = 50)
        .def("solve_batch", &PyNewtonSolver::solve_batch,
             py::arg("indptr"),     py::arg("indices"), py::arg("data"),
             py::arg("rows"),       py::arg("cols"),
             py::arg("sbus_batch"), py::arg("V0_batch"),
             py::arg("pv"),         py::arg("pq"),
             py::arg("n_batch"),
             py::arg("tolerance") = 1e-8,
             py::arg("max_iter")  = 50);
}
