#pragma once

#include "assembly/bus_block_jacobian_assembler.hpp"
#include "assembly/bus_local_jacobian_assembler.hpp"
#include "assembly/bus_local_residual_assembler.hpp"
#include "assembly/bus_local_voltage_update.hpp"
#include "linear/csr_spmv.hpp"
#include "linear/block_amgx_preconditioner.hpp"
#include "linear/bus_block_jacobi_preconditioner.hpp"
#include "linear/device_vector_ops.hpp"
#include "linear/fgmres_solver.hpp"
#include "model/bus_local_index.hpp"
#include "model/bus_local_jacobian_pattern.hpp"

#include <cstdint>
#include <vector>

namespace exp_20260414::amgx_v2 {

struct HostPowerFlowStructure {
    int32_t n_bus = 0;
    std::vector<int32_t> ybus_row_ptr;
    std::vector<int32_t> ybus_col_idx;
    std::vector<int32_t> pv;
    std::vector<int32_t> pq;
    GraphOrderingMethod ordering = GraphOrderingMethod::Natural;
};

struct DevicePowerFlowState {
    const int32_t* ybus_row_ptr = nullptr;
    const int32_t* ybus_col_idx = nullptr;
    const double* ybus_re = nullptr;
    const double* ybus_im = nullptr;
    const double* sbus_re = nullptr;
    const double* sbus_im = nullptr;
    double* voltage_re = nullptr;
    double* voltage_im = nullptr;
};

class JfnkAmgxSolver {
public:
    explicit JfnkAmgxSolver(SolverOptions options);

    void analyze(const HostPowerFlowStructure& structure);
    SolveStats solve(const DevicePowerFlowState& state);

private:
    SolverOptions options_;
    BusOrdering ordering_;
    BusLocalIndex index_;
    BusLocalJacobianPattern pattern_;
    BusLocalJacobianAssembler jacobian_assembler_;
    BusBlockJacobianAssembler direct_block_jacobian_;
    BusLocalResidualAssembler residual_assembler_;
    BusLocalVoltageUpdate voltage_update_;
    CsrSpmv jacobian_spmv_;
    AmgxPreconditioner amgx_;
    BlockAmgxPreconditioner block_amgx_;
    BusBlockJacobiPreconditioner bus_block_jacobi_;
    FgmresSolver fgmres_;
    DeviceVectorOps vector_ops_;
    DeviceBuffer<double> d_rhs_;
    DeviceBuffer<double> d_dx_;
    bool analyzed_ = false;
};

}  // namespace exp_20260414::amgx_v2
