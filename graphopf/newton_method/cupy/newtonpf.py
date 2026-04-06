from __future__ import annotations
import ctypes

import numpy as np
from scipy.sparse import csr_matrix

import cupy as cp
from nvmath.bindings import cudss, cusparse
import nvmath

from newton_method.cupy.jacobian import Jacobian


def mismatch(
    v: cp.ndarray,            # (B, nbus) complex64/complex128
    ibus: cp.ndarray,         # (B, nbus) complex64/complex128
    sbus: cp.ndarray,         # (B, nbus) complex64/complex128
    pv: cp.ndarray,           # (npv,) int32
    pq: cp.ndarray,           # (npq,) int32
) -> tuple[cp.ndarray, cp.ndarray]:
    # mis = V * conj(Ibus) - Sbus
    mis = v * cp.conj(ibus) - sbus

    # F = [Re(mis[pv]), Re(mis[pq]), Im(mis[pq])] per batch
    f1 = mis[:, pv].real
    f2 = mis[:, pq].real
    f3 = mis[:, pq].imag
    f = cp.concatenate([f1, f2, f3], axis=1).astype(cp.float64, copy=False)

    norm_f = cp.max(cp.abs(f), axis=1).astype(cp.float64, copy=False)
    return f, norm_f


class NewtonSolver:
    def __init__(
        self,
        ybus: csr_matrix,
        pv: np.ndarray,
        pq: np.ndarray,
        batch_size: int,
    ) -> None:
        ybus_c64 = ybus.astype(np.complex64)
        self.jacobian = Jacobian.analyze(ybus_c64, pv, pq, batch_size=batch_size)
        self.batch_size = int(batch_size)

        self.pv = cp.asarray(np.asarray(pv, dtype=np.int32).ravel(), dtype=cp.int32)
        self.pq = cp.asarray(np.asarray(pq, dtype=np.int32).ravel(), dtype=cp.int32)

        nbus = int(self.jacobian.nbus)
        self.npv = int(pv.size)
        self.npq = int(pq.size)

        # ---------------- cuSPARSE (Ybus * V = I) ----------------
        self.v_t = cp.empty((nbus, self.batch_size), dtype=cp.complex128, order="F")
        self.i_t = cp.empty((nbus, self.batch_size), dtype=cp.complex128, order="F")

        self.y_indptr = cp.asarray(ybus.indptr, dtype=cp.int32)
        self.y_indices = cp.asarray(ybus.indices, dtype=cp.int32)
        self.y_data = cp.asarray(ybus.data, dtype=cp.complex128)

        self.sp_handle = cusparse.create()
        self.sp_A = cusparse.create_csr(
            nbus,
            nbus,
            ybus.nnz,
            self.y_indptr.data.ptr,
            self.y_indices.data.ptr,
            self.y_data.data.ptr,
            cusparse.IndexType.INDEX_32I,
            cusparse.IndexType.INDEX_32I,
            cusparse.IndexBase.ZERO,
            nvmath.CudaDataType.CUDA_C_64F,
        )

        self.sp_B = cusparse.create_dn_mat(
            nbus,
            self.batch_size,
            nbus,
            self.v_t.data.ptr,
            nvmath.CudaDataType.CUDA_C_64F,
            cusparse.Order.COL,
        )

        self.sp_C = cusparse.create_dn_mat(
            nbus,
            self.batch_size,
            nbus,
            self.i_t.data.ptr,
            nvmath.CudaDataType.CUDA_C_64F,
            cusparse.Order.COL,
        )

        # Device scalars
        self.sp_alpha = cp.asarray(1.0 + 0.0j, dtype=cp.complex128)
        self.sp_beta = cp.asarray(0.0 + 0.0j, dtype=cp.complex128)

        cusparse.set_pointer_mode(self.sp_handle, cusparse.PointerMode.DEVICE)

        # Compute type must match the compute you pass to sp_mm (complex128 here)
        buf_sz = cusparse.sp_mm_buffer_size(
            self.sp_handle,
            cusparse.Operation.NON_TRANSPOSE,
            cusparse.Operation.NON_TRANSPOSE,
            self.sp_alpha.data.ptr,
            self.sp_A,
            self.sp_B,
            self.sp_beta.data.ptr,
            self.sp_C,
            nvmath.CudaDataType.CUDA_C_64F,
            cusparse.SpMMAlg.DEFAULT,
        )
        self.sp_workspace = cp.empty((int(buf_sz),), dtype=cp.uint8)

        # ---------------- cuDSS (uniform-batch) ----------------
        self.dss_handle = cudss.create()
        self.dss_config = cudss.config_create()
        self.dss_data = cudss.data_create(self.dss_handle)

        ubatch = ctypes.c_int(int(self.batch_size))
        cudss.config_set(
            self.dss_config,
            cudss.ConfigParam.UBATCH_SIZE,
            ctypes.addressof(ubatch),
            ctypes.sizeof(ubatch),
        )

        nJ = self.jacobian.R
        nnzJ = self.jacobian.nnzJ

        # CSR: uniform batch values buffer is expected to be contiguous (B * nnz) if ubatch is enabled
        self.dss_A = cudss.matrix_create_csr(
            nJ,
            nJ,
            nnzJ,
            self.jacobian.J_indptr.data.ptr,
            0,
            self.jacobian.J_indices.data.ptr,
            self.jacobian.J_data.data.ptr,
            nvmath.CudaDataType.CUDA_R_32I,
            nvmath.CudaDataType.CUDA_R_32F,
            cudss.MatrixType.GENERAL,
            cudss.MatrixViewType.FULL,
            cudss.IndexBase.ZERO,
        )

        # RHS / solution buffers (ubatch vectors flattened)
        self.dx = cp.zeros(self.batch_size * nJ, dtype=cp.float32)
        self.F = cp.zeros(self.batch_size * nJ, dtype=cp.float32)

        # Create dense matrices that wrap the flattened ubatch buffers
        self.dss_B = cudss.matrix_create_dn(
            nJ,
            1,
            nJ,
            self.F.data.ptr,
            nvmath.CudaDataType.CUDA_R_32F,
            cudss.Layout.COL_MAJOR,
        )
        self.dss_X = cudss.matrix_create_dn(
            nJ,
            1,
            nJ,
            self.dx.data.ptr,
            nvmath.CudaDataType.CUDA_R_32F,
            cudss.Layout.COL_MAJOR,
        )

        cudss.execute(
            self.dss_handle,
            cudss.Phase.ANALYSIS,
            self.dss_config,
            self.dss_data,
            self.dss_A,
            self.dss_X,
            self.dss_B,
        )

        cudss.execute(
            self.dss_handle,
            cudss.Phase.FACTORIZATION,
            self.dss_config,
            self.dss_data,
            self.dss_A,
            self.dss_X,
            self.dss_B,
        )

        # for backward operation in physics layer
        # J^T CSR descriptor (structure/value buffer must be provided by Jacobian: JT_indptr/JT_indices/JT_data)
        self.dss_A_T = cudss.matrix_create_csr(
            nJ,
            nJ,
            nnzJ,
            self.jacobian.JT_indptr.data.ptr,
            0,
            self.jacobian.JT_indices.data.ptr,
            self.jacobian.JT_data.data.ptr,
            nvmath.CudaDataType.CUDA_R_32I,
            nvmath.CudaDataType.CUDA_R_32F,
            cudss.MatrixType.GENERAL,
            cudss.MatrixViewType.FULL,
            cudss.IndexBase.ZERO,
        )

        cudss.execute(
            self.dss_handle,
            cudss.Phase.ANALYSIS,
            self.dss_config,
            self.dss_data,
            self.dss_A_T,
            self.dss_X,
            self.dss_B,
        )

        cudss.execute(
            self.dss_handle,
            cudss.Phase.FACTORIZATION,
            self.dss_config,
            self.dss_data,
            self.dss_A_T,
            self.dss_X,
            self.dss_B,
        )

    def close(self) -> None:
        # cuDSS
        cudss.matrix_destroy(self.dss_B)
        cudss.matrix_destroy(self.dss_X)
        cudss.matrix_destroy(self.dss_A)
        cudss.matrix_destroy(self.dss_A_T)
        cudss.data_destroy(self.dss_handle, self.dss_data)
        cudss.config_destroy(self.dss_config)
        cudss.destroy(self.dss_handle)

        # cuSPARSE
        cusparse.destroy_dn_mat(self.sp_B)
        cusparse.destroy_dn_mat(self.sp_C)
        cusparse.destroy_sp_mat(self.sp_A)
        cusparse.destroy(self.sp_handle)

    def solve(
        self,
        sbus: cp.ndarray,       # (B, nbus) complex64/complex128
        v0: cp.ndarray,         # (B, nbus) complex64/complex128
        tolerance: float,
        max_iter: int,
    ) -> cp.ndarray:
        if self.batch_size != int(v0.shape[0]):
            raise ValueError("Batch size mismatch between solver and input v0.")
        
        v = v0.copy()
        va = cp.angle(v).astype(cp.float64, copy=False)
        vm = cp.abs(v).astype(cp.float64, copy=False)

        converged = cp.zeros((self.batch_size,), dtype=cp.bool_)
        norm_f = cp.full((self.batch_size,), cp.inf, dtype=cp.float64)

        v64 = cp.empty_like(v, dtype=cp.complex64)

        for _ in range(int(max_iter)):
            cp.copyto(self.v_t, v.T)
            
            cusparse.sp_mm(
                self.sp_handle,
                cusparse.Operation.NON_TRANSPOSE,
                cusparse.Operation.NON_TRANSPOSE,
                self.sp_alpha.data.ptr,
                self.sp_A,
                self.sp_B,
                self.sp_beta.data.ptr,
                self.sp_C,
                nvmath.CudaDataType.CUDA_C_64F,
                cusparse.SpMMAlg.DEFAULT,
                self.sp_workspace.data.ptr,
            )

            ibus = self.i_t.T.copy(order="C")  # (B, nbus) complex128
            f, norm_f = mismatch(v, ibus, sbus, self.pv, self.pq)
            
            newly = norm_f < float(tolerance)
            converged = converged | newly
            if bool(cp.all(converged)):
                break

            # Update Jacobian values in-place (must keep same underlying buffer for cuDSS descriptor)
            v64.real = v.real.astype(cp.float32, copy=False)
            v64.imag = v.imag.astype(cp.float32, copy=False)
            self.jacobian.update(V=v64)

            # RHS = -f (in-place into the ubatch RHS buffer that dss_B wraps)
            self.F[:] = (-f).astype(cp.float32, copy=False).reshape(-1)
            self.dx.fill(0.0)

            cudss.execute(
                self.dss_handle,
                cudss.Phase.REFACTORIZATION,
                self.dss_config,
                self.dss_data,
                self.dss_A,
                self.dss_X,
                self.dss_B,
            )
            cudss.execute(
                self.dss_handle,
                cudss.Phase.SOLVE,
                self.dss_config,
                self.dss_data,
                self.dss_A,
                self.dss_X,
                self.dss_B,
            )

            npv = self.npv
            npq = self.npq

            k0 = 0
            k1 = k0 + npv
            k2 = k1 + npq
            k3 = k2 + npq

            dx2 = self.dx.reshape(self.batch_size, -1)

            va[:, self.pv] += dx2[:, k0:k1]
            va[:, self.pq] += dx2[:, k1:k2]
            vm[:, self.pq] += dx2[:, k2:k3]

            # Rebuild V from (Vm, Va) without extra allocations
            cp.multiply(vm, cp.cos(va), out=v.real)
            cp.multiply(vm, cp.sin(va), out=v.imag)

        v64.real = v.real.astype(cp.float32, copy=False)
        v64.imag = v.imag.astype(cp.float32, copy=False)
        self.jacobian.update_JT(V=v64)

        return v
