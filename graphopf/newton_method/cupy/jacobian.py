from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp
import cupy as cp


@dataclass
class Jacobian:
    nbus: int
    nnzJ: int
    nnzY: int

    batch_size: int
    R: int
    C: int

    # Ybus on GPU (CSR, complex64 / int32)
    Y_indptr: cp.ndarray          # int32 [nbus + 1]
    Y_indices: cp.ndarray         # int32 [nnzY]
    Y_data: cp.ndarray            # complex64 [nnzY]
    Y_row_indices: cp.ndarray     # int32 [nnzY]

    # J on GPU (CSR, float32 / int32) - structure only; values are per-batch output
    J_indptr: cp.ndarray          # int32 [R + 1]
    J_indices: cp.ndarray         # int32 [nnzJ]
    J_data: cp.ndarray            # float32 [batch_size, nnzJ]

    # J^T on GPU (CSR, float32 / int32)
    JT_indptr: cp.ndarray         # int32 [C + 1] (same as R+1)
    JT_indices: cp.ndarray        # int32 [nnzJ]
    JT_data: cp.ndarray           # float32 [batch_size * nnzJ]
    j_to_jt_map: cp.ndarray       # int32 [nnzJ]  (maps J nnz position -> JT nnz position)

    # Maps on GPU (int32)
    mapJ11: cp.ndarray            # int32 [nnzY]
    mapJ21: cp.ndarray            # int32 [nnzY]
    mapJ12: cp.ndarray            # int32 [nnzY]
    mapJ22: cp.ndarray            # int32 [nnzY]
    diagMapJ11: cp.ndarray        # int32 [nbus]
    diagMapJ21: cp.ndarray        # int32 [nbus]
    diagMapJ12: cp.ndarray        # int32 [nbus]
    diagMapJ22: cp.ndarray        # int32 [nbus]

    # Kernels
    update_kernel: cp.RawKernel

    @staticmethod
    def analyze(
        ybus: sp.csr_matrix,  # complex64 recommended
        pv: np.ndarray,
        pq: np.ndarray,
        batch_size: int,
    ) -> "Jacobian":
        if not sp.isspmatrix_csr(ybus):
            raise TypeError("ybus must be scipy.sparse.csr_matrix (CSR)")
        if ybus.shape[0] != ybus.shape[1]:
            raise ValueError("ybus must be square")

        pv = np.asarray(pv, dtype=np.int32).ravel()
        pq = np.asarray(pq, dtype=np.int32).ravel()

        nbus = int(ybus.shape[0])
        npv = int(pv.size)
        npq = int(pq.size)
        npvpq = npv + npq
        pvpq = np.concatenate([pv, pq]).astype(np.int32, copy=False)

        R = npvpq + npq
        C = R

        rmap_pvpq = np.full(nbus, -1, dtype=np.int32)
        rmap_pq = np.full(nbus, -1, dtype=np.int32)
        cmap_pvpq = np.full(nbus, -1, dtype=np.int32)
        cmap_pq = np.full(nbus, -1, dtype=np.int32)

        rmap_pvpq[pvpq] = np.arange(npvpq, dtype=np.int32)
        rmap_pq[pq] = np.arange(npq, dtype=np.int32) + npvpq
        cmap_pvpq[pvpq] = np.arange(npvpq, dtype=np.int32)
        cmap_pq[pq] = np.arange(npq, dtype=np.int32) + npvpq

        y_indptr = np.asarray(ybus.indptr, dtype=np.int32)
        y_indices = np.asarray(ybus.indices, dtype=np.int32)
        y_data = np.asarray(ybus.data, dtype=np.complex64)

        rows: list[int] = []
        cols: list[int] = []
        vals: list[np.float32] = []

        for i in range(nbus):
            start = int(y_indptr[i])
            end = int(y_indptr[i + 1])

            i_pvpq = int(rmap_pvpq[i])
            i_pq = int(rmap_pq[i])

            for t in range(start, end):
                j = int(y_indices[t])
                if i == j:
                    continue

                j_pvpq = int(cmap_pvpq[j])
                j_pq = int(cmap_pq[j])

                if i_pvpq >= 0 and j_pvpq >= 0:
                    rows.append(i_pvpq)
                    cols.append(j_pvpq)
                    vals.append(np.float32(1.0))
                if i_pq >= 0 and j_pvpq >= 0:
                    rows.append(i_pq)
                    cols.append(j_pvpq)
                    vals.append(np.float32(1.0))
                if i_pvpq >= 0 and j_pq >= 0:
                    rows.append(i_pvpq)
                    cols.append(j_pq)
                    vals.append(np.float32(1.0))
                if i_pq >= 0 and j_pq >= 0:
                    rows.append(i_pq)
                    cols.append(j_pq)
                    vals.append(np.float32(1.0))

        for bus in range(nbus):
            i_pvpq = int(rmap_pvpq[bus])
            i_pq = int(rmap_pq[bus])
            j_pvpq = int(cmap_pvpq[bus])
            j_pq = int(cmap_pq[bus])

            if i_pvpq >= 0 and j_pvpq >= 0:
                rows.append(i_pvpq)
                cols.append(j_pvpq)
                vals.append(np.float32(1.0))
            if i_pq >= 0 and j_pvpq >= 0:
                rows.append(i_pq)
                cols.append(j_pvpq)
                vals.append(np.float32(1.0))
            if i_pvpq >= 0 and j_pq >= 0:
                rows.append(i_pvpq)
                cols.append(j_pq)
                vals.append(np.float32(1.0))
            if i_pq >= 0 and j_pq >= 0:
                rows.append(i_pq)
                cols.append(j_pq)
                vals.append(np.float32(1.0))

        J_coo = sp.coo_matrix(
            (
                np.asarray(vals, dtype=np.float32),
                (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32)),
            ),
            shape=(R, C),
        )
        J_csr = J_coo.tocsr()
        J_csr.sort_indices()

        J_indptr = np.asarray(J_csr.indptr, dtype=np.int32)
        J_indices = np.asarray(J_csr.indices, dtype=np.int32)

        def find_coeff_index_csr(row: int, col: int) -> int:
            # Find the (row, col) position in CSR (sorted indices), return linear nnz index.
            start = int(J_indptr[row])
            end = int(J_indptr[row + 1])
            seg = J_indices[start:end]
            pos = int(np.searchsorted(seg, col))
            if pos < (end - start) and int(seg[pos]) == col:
                return start + pos
            return -1

        nnzY = int(y_indices.size)
        mapJ11 = np.full(nnzY, -1, dtype=np.int32)
        mapJ21 = np.full(nnzY, -1, dtype=np.int32)
        mapJ12 = np.full(nnzY, -1, dtype=np.int32)
        mapJ22 = np.full(nnzY, -1, dtype=np.int32)

        t = 0
        for i in range(nbus):
            start = int(y_indptr[i])
            end = int(y_indptr[i + 1])

            i_pvpq = int(rmap_pvpq[i])
            i_pq = int(rmap_pq[i])

            for _ in range(start, end):
                j = int(y_indices[t])

                j_pvpq = int(cmap_pvpq[j])
                j_pq = int(cmap_pq[j])

                if i_pvpq >= 0 and j_pvpq >= 0:
                    mapJ11[t] = find_coeff_index_csr(i_pvpq, j_pvpq)
                if i_pq >= 0 and j_pvpq >= 0:
                    mapJ21[t] = find_coeff_index_csr(i_pq, j_pvpq)
                if i_pvpq >= 0 and j_pq >= 0:
                    mapJ12[t] = find_coeff_index_csr(i_pvpq, j_pq)
                if i_pq >= 0 and j_pq >= 0:
                    mapJ22[t] = find_coeff_index_csr(i_pq, j_pq)

                t += 1

        diagMapJ11 = np.full(nbus, -1, dtype=np.int32)
        diagMapJ21 = np.full(nbus, -1, dtype=np.int32)
        diagMapJ12 = np.full(nbus, -1, dtype=np.int32)
        diagMapJ22 = np.full(nbus, -1, dtype=np.int32)

        for bus in range(nbus):
            i_pvpq = int(rmap_pvpq[bus])
            i_pq = int(rmap_pq[bus])
            j_pvpq = int(cmap_pvpq[bus])
            j_pq = int(cmap_pq[bus])

            if i_pvpq >= 0 and j_pvpq >= 0:
                diagMapJ11[bus] = find_coeff_index_csr(i_pvpq, j_pvpq)
            if i_pq >= 0 and j_pvpq >= 0:
                diagMapJ21[bus] = find_coeff_index_csr(i_pq, j_pvpq)
            if i_pvpq >= 0 and j_pq >= 0:
                diagMapJ12[bus] = find_coeff_index_csr(i_pvpq, j_pq)
            if i_pq >= 0 and j_pq >= 0:
                diagMapJ22[bus] = find_coeff_index_csr(i_pq, j_pq)

        nnzJ = int(J_indices.size)

        counts = (y_indptr[1:] - y_indptr[:-1]).astype(np.int32, copy=False)
        y_row_indices = np.repeat(np.arange(nbus, dtype=np.int32), counts)

        update_kernel = Jacobian._compile_kernels()

        # ---- Build transpose CSR structure for J^T once (CPU) ----
        jt_indptr = np.zeros(R + 1, dtype=np.int32)
        for k in range(nnzJ):
            col = int(J_indices[k])
            jt_indptr[col + 1] += 1
        np.cumsum(jt_indptr, out=jt_indptr)

        jt_indices = np.empty(nnzJ, dtype=np.int32)
        j_to_jt_map = np.empty(nnzJ, dtype=np.int32)

        write_ptr = jt_indptr[:-1].copy()
        for r in range(R):
            start = int(J_indptr[r])
            end = int(J_indptr[r + 1])
            for k in range(start, end):
                c = int(J_indices[k])
                pos = int(write_ptr[c])
                jt_indices[pos] = r
                j_to_jt_map[k] = pos
                write_ptr[c] += 1

        return Jacobian(
            nbus=nbus,
            nnzY=nnzY,
            nnzJ=nnzJ,
            batch_size=batch_size,
            R=R,
            C=C,
            Y_indptr=cp.asarray(y_indptr, dtype=cp.int32),
            Y_indices=cp.asarray(y_indices, dtype=cp.int32),
            Y_data=cp.asarray(y_data, dtype=cp.complex64),
            Y_row_indices=cp.asarray(y_row_indices, dtype=cp.int32),
            J_indptr=cp.asarray(J_indptr, dtype=cp.int32),
            J_indices=cp.asarray(J_indices, dtype=cp.int32),
            J_data=cp.zeros(batch_size * nnzJ, dtype=cp.float32),
            JT_indptr=cp.asarray(jt_indptr, dtype=cp.int32),
            JT_indices=cp.asarray(jt_indices, dtype=cp.int32),
            JT_data=cp.zeros(batch_size * nnzJ, dtype=cp.float32),
            j_to_jt_map=cp.asarray(j_to_jt_map, dtype=cp.int32),
            mapJ11=cp.asarray(mapJ11, dtype=cp.int32),
            mapJ21=cp.asarray(mapJ21, dtype=cp.int32),
            mapJ12=cp.asarray(mapJ12, dtype=cp.int32),
            mapJ22=cp.asarray(mapJ22, dtype=cp.int32),
            diagMapJ11=cp.asarray(diagMapJ11, dtype=cp.int32),
            diagMapJ21=cp.asarray(diagMapJ21, dtype=cp.int32),
            diagMapJ12=cp.asarray(diagMapJ12, dtype=cp.int32),
            diagMapJ22=cp.asarray(diagMapJ22, dtype=cp.int32),
            update_kernel=update_kernel,
        )

    def update(
        self,
        V: cp.ndarray,                         # complex64 [B, nbus] (row-major)
    ) -> None:
        if V.dtype != cp.complex64:
            raise TypeError("V must be cupy.complex64")
        if V.ndim != 2 or V.shape[1] != self.nbus:
            raise ValueError("V must have shape (B, nbus)")

        B = int(V.shape[0])
        nbus = self.nbus
        nnzY = self.nnzY
        nnzJ = self.nnzJ

        self.J_data.fill(0)

        threads = 256
        blocks = (nnzY + threads - 1) // threads

        self.update_kernel(
            (blocks,),
            (threads,),
            (
                nnzY,
                nbus,
                B,
                nnzJ,
                self.Y_row_indices,      # row[k]
                self.Y_indices,          # col[k]
                self.Y_data,             # y[k] complex64
                V.reshape(-1),           # [B*nbus] complex64
                self.mapJ11,
                self.mapJ21,
                self.mapJ12,
                self.mapJ22,
                self.diagMapJ11,
                self.diagMapJ21,
                self.diagMapJ12,
                self.diagMapJ22,
                self.J_data.reshape(-1),  # [B*nnzJ] float32
            ),
        )
    
    def update_JT(
        self,
        V: cp.ndarray,  # complex64 [B, nbus] (row-major)
    ) -> None:
        self.update(V)

        B = int(V.shape[0])
        nnzJ = int(self.nnzJ)

        j_vals_2d = self.J_data.reshape(B, nnzJ)
        self.JT_data[:] = cp.take(j_vals_2d, self.j_to_jt_map, axis=1).reshape(-1)

    @staticmethod
    def _compile_kernels() -> cp.RawKernel:
        cuda_src = r'''
            #include <cuComplex.h>

            extern "C" __global__
            void update_jacobian_kernel(
                const int n_elements,
                const int nb,
                const int B,
                const int nnzJ,

                const int* __restrict__ row,                 // [n_elements]
                const int* __restrict__ col,                 // [n_elements]
                const cuFloatComplex* __restrict__ Y,         // [n_elements]

                const cuFloatComplex* __restrict__ V,         // [B*nb] (row-major)
                const int* __restrict__ map11,                // [n_elements]
                const int* __restrict__ map21,
                const int* __restrict__ map12,
                const int* __restrict__ map22,
                const int* __restrict__ diag11,               // [nb]
                const int* __restrict__ diag21,
                const int* __restrict__ diag12,
                const int* __restrict__ diag22,

                float* __restrict__ J_values                  // [B*nnzJ]
            ) {
                const int k = (int)(blockIdx.x * blockDim.x + threadIdx.x);
                if (k >= n_elements) return;

                const int i = row[k];
                const int j = col[k];

                const cuFloatComplex y = Y[k];

                const int p11 = map11[k];
                const int p21 = map21[k];
                const int p12 = map12[k];
                const int p22 = map22[k];

                const int d11 = diag11[i];
                const int d21 = diag21[i];
                const int d12 = diag12[i];
                const int d22 = diag22[i];

                for (int b = 0; b < B; ++b) {
                    const cuFloatComplex Vi = V[b * nb + i];
                    const cuFloatComplex Vj = V[b * nb + j];

                    // curr = Y_ij * V_j
                    const cuFloatComplex curr = cuCmulf(y, Vj);

                    // term_va = (-j * Vi) * conj(curr)
                    const cuFloatComplex minus_j_Vi = make_cuFloatComplex(cuCimagf(Vi), -cuCrealf(Vi));
                    const cuFloatComplex term_va = cuCmulf(minus_j_Vi, cuConjf(curr));

                    // term_vm = Vi * conj(curr / |Vj|)
                    const float vj_norm = cuCabsf(Vj);
                    cuFloatComplex term_vm;
                    if (vj_norm > 1e-6f) {
                        const cuFloatComplex curr_over = make_cuFloatComplex(
                            cuCrealf(curr) / vj_norm,
                            cuCimagf(curr) / vj_norm
                        );
                        term_vm = cuCmulf(Vi, cuConjf(curr_over));
                    } else {
                        term_vm = make_cuFloatComplex(0.0f, 0.0f);
                    }

                    float* Jb = J_values + b * nnzJ;

                    // Same scatter/correction logic as your single-batch kernel (use atomicAdd everywhere)
                    if (p11 >= 0) atomicAdd(&Jb[p11], cuCrealf(term_va));
                    if (p21 >= 0) atomicAdd(&Jb[p21], cuCimagf(term_va));
                    if (p12 >= 0) atomicAdd(&Jb[p12], cuCrealf(term_vm));
                    if (p22 >= 0) atomicAdd(&Jb[p22], cuCimagf(term_vm));

                    if (d11 >= 0) atomicAdd(&Jb[d11], -cuCrealf(term_va));
                    if (d21 >= 0) atomicAdd(&Jb[d21], -cuCimagf(term_va));

                    const float vi_norm = cuCabsf(Vi);
                    if (vi_norm > 1e-6f) {
                        const cuFloatComplex Vi_unit = make_cuFloatComplex(
                            cuCrealf(Vi) / vi_norm,
                            cuCimagf(Vi) / vi_norm
                        );
                        const cuFloatComplex term_vm2 = cuCmulf(Vi_unit, cuConjf(curr));
                        if (d12 >= 0) atomicAdd(&Jb[d12], cuCrealf(term_vm2));
                        if (d22 >= 0) atomicAdd(&Jb[d22], cuCimagf(term_vm2));
                    }
                }
            }
        '''

        fused = cp.RawKernel(cuda_src, "update_jacobian_kernel")
        return fused
