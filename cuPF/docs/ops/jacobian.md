# ops/jacobian — Jacobian 분석과 조립

NR 선형계 `J·dx = F`의 Jacobian을 만든다. 두 부분: (1) 1회성 **심볼릭 분석**(구조),
(2) 매 반복 **수치 조립**(값).

**소스**: `cpp/src/newton_solver/ops/jacobian/` — `jacobian_analysis.{cpp,hpp}`(분석, 호스트),
`fill_jacobian.cpp`(CPU 조립), `fill_jacobian_gpu.cu`(GPU 디스패치) +
`jacobian_gpu_common.hpp`, `fill_jacobian_edge_kernel.hpp`,
`fill_jacobian_edge_atomic_kernel.hpp`, `fill_jacobian_vertex_warp_kernel.hpp`.

---

## 1. 블록 구조

극형식 NR에서 미지수는 전압 각도 Va·크기 Vm, 잔차는 P·Q. 2×2 블록:

```
[ J11 = dP/dVa   J12 = dP/dVm ]
[ J21 = dQ/dVa   J22 = dQ/dVm ]
```

PV 버스는 각도(pvpq) 행/열만, PQ 버스는 크기(pq) 행/열도 가진다. Ybus nonzero (i,j)는
off-diagonal(i≠j) 4개 블록에 기여하고, 대각(i==i)은 버스 전류
`Ibus_i = Σ_j Y_ij V_j`로 만든 self 항을 더한다.

## 2. 심볼릭 분석 (`jacobian_analysis.cpp`, 호스트·1회)

`initialize`에서만 실행되는 인덱스 분석(부동소수 없음, NR 핫패스 아님):

- `make_jacobian_indexing`: 버스 ↔ Jacobian 행/열 인덱스 테이블(PV/PQ 블록).
- `JacobianPatternGenerator::generate`: J의 **CSR sparsity 패턴**. Ybus 구조 대칭성을
  이용(상삼각 순회 후 대칭 추가)하고 per-row sort/unique로 CSR 압축.
- `JacobianMapBuilder::build`: **scatter map** — 각 Ybus nonzero가 J의 어느 value
  슬롯에 들어가는지(`mapJ11..22`) + 대각 슬롯(`diagJ11..22`). 매 반복 조립은 이 맵을
  통한 순수 scatter가 된다.

출력은 `InitializeContext`로 storage에 전달(CUDA는 CSR 그대로, CPU는 CSC로 리맵 —
[storage.md](../storage.md)).

## 3. GPU 조립 (variant 3종)

off-diagonal 민감도(엣지당 curr·각도항·크기항)는 모든 variant가 공유하는 device 헬퍼
`compute_edge_sensitivity`(`jacobian_gpu_common.hpp`, 연산 순서 보존)로 계산하고,
대각 처리만 다르다. `CudaJacobianKind`로 선택(`NewtonOptions.cuda_jacobian`):

- **Edge** (기본, `fill_jacobian_edge_kernel.hpp`): 엣지당 한 스레드, off-diagonal은
  직접 store(슬롯당 writer 1개), 대각은 **캐시된 Ibus**를 재사용해 self 항을 더함
  (Ibus 단계가 직전에 돌았으므로). 프로덕션 경로.
- **EdgeAtomic** (`fill_jacobian_edge_atomic_kernel.hpp`): 엣지당 한 스레드, 모든 기여를
  `atomicAdd`로 흩뿌림(대각은 −각도항 + 엣지별 크기항 누적). 대안/실험 레이아웃.
- **VertexWarp** (`fill_jacobian_vertex_warp_kernel.hpp`): 행(=pvpq 버스)당 한 warp,
  레인이 행의 nonzero를 stride 처리하고 대각 partial을 워프 리덕션. 대안/실험.

공용 device 헬퍼: `cupf_atomic_add`(f32/f64, 구형 arch는 CAS), `warp_reduce_sum`,
`dump_cuda_jacobian_if_enabled`. 디스패치(`fill_jacobian_gpu.cu`)는 `CudaJacobianOp<T>::run`
3종이 프로파일별 스칼라(FP64=double/double/double, Mixed=float/double/double,
FP32=float/float/float)로 variant 런처를 호출한다.

## 4. CPU 조립 (`fill_jacobian.cpp`)

호스트에서 scatter map으로 J 값을 채운다(`CpuJacobianOpF64`). `CpuJacobianKind`(Native/
Pandapower)로 세부 선택.
