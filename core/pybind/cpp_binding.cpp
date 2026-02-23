// cpp_newtonpf.cpp
// @brief Pybind11 bindings for Newton-Raphson Power Flow (Eigen/KLU)

#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>  // dense helper (safety; sparse is handled by our helper)

#include <complex>
#include <cstdint>

#include "newtonpf.hpp"      // newtonpf(nr_data::NRData) -> nr_data::NRResult
#include "nr_data.hpp"       // NRData / NRResult definitions
#include "binding_helper.hpp"// to_eigen_csc / to_eigen_vec / to_eigen_vec_i32

namespace py = pybind11;
using namespace pf_binding;

PYBIND11_MODULE(cpp_newtonpf, m) {
    m.doc() = "C++ Newton-Raphson Power Flow (Eigen/KLU)";

    m.def("newtonpf",
        [](py::handle ybus,
        py::handle sbus,
        py::handle V0,
        py::handle ref,
        py::handle pv,
        py::handle pq,
        py::dict ppopt)
    {
        nr_data::YbusType c_ybus = to_eigen_csc(ybus);
        nr_data::VectorXcd c_sbus = to_eigen_vec<std::complex<double>>(sbus, "sbus");
        nr_data::VectorXcd c_V0 = to_eigen_vec<std::complex<double>>(V0, "V0");
        nr_data::VectorXi32 c_pv = to_eigen_vec_i32(pv, "pv");
        nr_data::VectorXi32 c_pq = to_eigen_vec_i32(pq, "pq");

        double tolerance = ppopt.contains("tolerance") ? ppopt["tolerance"].cast<double>() : 1e-8;
        int max_iter  = ppopt.contains("max_iter")  ? ppopt["max_iter"].cast<int>() : 10;

        nr_data::NRResult res;
        {
            py::gil_scoped_release release;
            res = newtonPF(c_ybus, c_sbus, c_V0, c_pv, c_pq, tolerance, max_iter);
        }

        py::array_t<std::complex<double>> V_np(res.V.size(), res.V.data());

        return py::make_tuple(V_np, py::bool_(res.converged), res.iter);
        },
    py::arg("ybus"), py::arg("sbus"), py::arg("V0"),
    py::arg("ref"), py::arg("pv"), py::arg("pq"),
    py::arg("ppopt"));
}