# CUDA 백엔드

**헤더**: `inc/newton_solver/backend/cuda_backend.hpp`  
**내부 구현 헤더**: `src/newton_solver/backend/cuda/cuda_backend_impl.hpp`  
**구현 파일들**:

| 파일 | 내용 |
|------|------|
| `cuda/initialize.cpp` | 생성자, `analyze()`, `initialize()`, `downloadV()` |
| `cuda/compute_mismatch.cu` | `computeMismatch()` + CUDA 커널 |
| `cuda/update_jacobian.cu` | `updateJacobian()`, `updateVoltage()` + CUDA 커널 |
| `cuda/cudss_solver.cpp` | `solveLinearSystem()` |

---

## 정밀도 (혼합)

| 데이터 | 정밀도 | 이유 |
|--------|--------|------|
| Jacobian | FP32 | cuDSS FP32 인수분해 (~2× 속도 향상) |
| 전압 Va, Vm, V | FP64 | 수렴 정확도 유지 |
| Ybus SpMV | FP64 | mismatch 계산 정확도 |

---

## Pimpl 구조

공개 헤더(`cuda_backend.hpp`)에 CUDA 헤더가 노출되지 않는다.  
CPU-only 빌드(`-DWITH_CUDA` 없음)에서도 include 가능.

```
cuda_backend.hpp        ← CUDA 헤더 없음 (공개)
    └── Impl (cuda_backend_impl.hpp)
            ├── d_G_f, d_B_f, d_Y_row, d_Y_col   (Ybus FP32 + 인덱스)
            ├── d_Ybus_val/rp/ci                  (Ybus FP64 CSR, SpMV용)
            ├── d_mapJ**, d_diagJ**               (JacobianMaps on GPU)
            ├── d_pv, d_pq                        (버스 인덱스, analyze에서 업로드)
            ├── d_V_cd, d_Va, d_Vm, d_V_f        (전압 상태)
            ├── d_Sbus, d_Ibus, d_F, d_dx        (계산 버퍼, pre-allocated)
            ├── d_J_csr_f, d_J_csr_rp, d_J_csr_ci (Jacobian CSR FP32)
            ├── cusparseHandle/descriptors         (SpMV)
            └── cudssHandle/config/data/matrices   (직접 분해)
```

---

## analyze()

```
Ybus FP32 (G, B) + 행/열 인덱스 → GPU 업로드
Ybus FP64 복소 CSR → GPU + cuSPARSE 핸들 생성
JacobianMaps (mapJ**, diagJ**) → GPU 업로드
pv/pq 버스 인덱스 (maps.pvpq에서 추출) → GPU 업로드 (1회)
전압/버퍼 GPU 메모리 할당 (d_dx 포함, pre-allocated)
JacobianStructure CSR (row_ptr, col_idx) → GPU 업로드
cuDSS 핸들/config 초기화 + ANALYSIS 단계 실행 (심볼릭 분해)
```

**이전 대비 변경**: `JacobianStructure`가 CSR이므로 CSC→CSR 변환(`build_csc_to_csr_perm`) 불필요. pv/pq도 `analyze()`에서 한 번 업로드해 `computeMismatch`의 lazy alloc이 제거됨.

---

## computeMismatch()

```
[GPU] cuSPARSE SpMV: Ibus = Ybus * V_cd  (FP64 복소)
[GPU] mismatch_pack_kernel:
        mis(bus) = V[bus] * conj(Ibus[bus]) - Sbus[bus]
        F 패킹: [Re(pv), Re(pq), Im(pq)]
[GPU] max_abs_kernel: 블록 단위 shared-memory reduction
[Host] 블록 최댓값 다운로드 → host에서 max → normF
[Host] F 다운로드 (dimF 원소, 소량)
```

`d_pv`, `d_pq`는 이미 GPU에 있어 별도 업로드 없음.

---

## updateJacobian()

```
[GPU] cudaMemset(d_J_csr_f, 0)
[GPU] convert_V_cd_to_f_kernel:
        FP64 복소 V_cd → FP32 인터리브 V_f[i*2], V_f[i*2+1]
[GPU] EdgeBased:
        update_jacobian_kernel_fp32
          스레드 1개 per Ybus 비제로 k
          atomicAdd 기반 누적

[GPU] VertexBased:
        update_jacobian_vertex_kernel_fp32
          warp 1개 per active bus i (i in pvpq)
          Ybus CSR row i를 직접 순회
          off-diagonal은 direct write
          diagonal은 warp-local sum 후 1회 write
```

**이전 대비 변경**: JacobianMaps 인덱스가 CSR 위치이므로 별도 permutation 커널 없이 `d_J_csr_f`에 직접 기록.

---

## solveLinearSystem()

```
[Host] F(FP64) → b(FP32) 변환, d_b_f 업로드
[GPU]  cuDSS FACTORIZATION: FP32 LU (심볼릭은 analyze에서 완료)
[GPU]  cuDSS SOLVE: J·x = b
[Host] d_x_f 다운로드, FP32 → FP64 변환 → dx
```

---

## updateVoltage()

```
[Host] dx → d_dx (pre-allocated, cudaMemcpy only)
[GPU]  update_voltage_kernel_fp64:
         Va[pv[tid]] += dx[tid]          (FP64)
         Va[pq[tid]] += dx[n_pv + tid]
         Vm[pq[tid]] += dx[n_pv+n_pq+tid]
[GPU]  reconstruct_V_kernel:
         V_cd[i] = Vm[i] * (cos(Va[i]) + j*sin(Va[i]))
```

`d_dx`는 `analyze()`에서 pre-allocated — 루프 내 GPU alloc 없음.
