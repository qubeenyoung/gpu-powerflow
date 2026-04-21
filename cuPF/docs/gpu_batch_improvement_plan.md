# GPU Batch and Kernel Improvement Plan

이 문서는 CUDA NR 루프를 배치 실행에 맞게 정리하기 위한 개선 항목을 모은다.
목표는 같은 배치 안에서 Ybus 희소 패턴이 고정된다는 가정을 활용해, Newton hot path를 가능한 한 device-resident 형태로 유지하는 것이다.

기본 실행 모델은 항상 batch다. 단일 케이스 solve는 별도 구현이 아니라 `batch_size = 1`인 특수한 batch 실행으로 처리한다. 따라서 `cuda_batch` 폴더에 별도 operator/storage를 만들지 않고, 기존 CUDA storage와 op를 batch-aware 형태로 확장한다.

---

## 배치 가정

기본 구현 범위는 같은 sparse pattern을 공유하는 배치다.

```
batch b = 0..B-1
Ybus.indptr, Ybus.indices, Jacobian pattern, mapJ**, diagJ**: batch 공통
Ybus values: 공통 또는 batch별 값을 모두 허용할 수 있지만, 구조는 고정
Sbus, V0, V, F, dx, J_values: batch별 상태
pv, pq, pvpq: batch 공통
```

이 가정에서는 `analyze()` 결과를 배치 전체가 공유할 수 있다. 즉 Jacobian structure, scatter maps, cuDSS symbolic analysis 입력 구조는 한 번 만든 뒤 재사용한다.

Jacobian과 linear solve의 배치 표현은 cuDSS uniform batch를 따른다. 배치 안의 모든 Jacobian은 같은 `dimF`, `nnz_J`, `row_ptr`, `col_idx`를 공유하고, 값과 RHS/solution만 batch별로 다르다.

---

## 전압 상태

전압 상태는 `Va/Vm`을 FP64 authoritative state로 둔다. 현재 ablation은
`Ybus32 * V64 -> Ibus64` 경로를 보기 위해 `V_re/V_im` derived cache도 FP64로 둔다.
초기에는 `V_re/V_im` FP32 cache가 Texas Univ `case_ACTIVSg200` B=1 수렴성을 악화시켰고,
이후 FP64 cache로 개선됐었다. 이번 단계에서는 `Ibus` 저장을 FP64로 올린 상태에서
`V_re/V_im`도 FP64로 유지해 영향을 재측정한다.

```
authoritative state: Va, Vm       // double
derived cache:       V_re, V_im   // double
```

Voltage update는 `Va/Vm`을 갱신하고, 갱신된 active bus의 `V_re/V_im`만 재구성한다. 이후 mismatch와 Jacobian은 `V_re/V_im`을 읽는다.

장점:
- update 단계에서 매번 `V_re/V_im -> Va/Vm` 분해 커널을 실행할 필요가 없다.
- mismatch/Jacobian은 복소 전압을 바로 읽을 수 있어 `sincos(Va)` 반복 계산을 피한다.
- slack bus는 갱신되지 않으므로 active bus만 재구성할 수 있다.
- 기존 CUDA 커널의 입력 형태와 잘 맞아 이행 비용이 낮다.

단점:
- 전압 상태가 4개 FP64 배열이므로 메모리를 더 쓴다.
- `Va/Vm` 갱신 뒤 `V_re/V_im` cache가 반드시 같은 stream에서 갱신되어야 한다.

불변식:

```
V_re[i] = Vm[i] * cos(Va[i])
V_im[i] = Vm[i] * sin(Va[i])
```

위 불변식은 update 커널 이후, mismatch/Jacobian 실행 전에 성립해야 한다.
최종 `NRResultF64.V`를 만들 때는 FP64 `Va/Vm`에서 재구성해 public output 정밀도를 유지한다.

---

## 권장 전압 업데이트

현재 CUDA voltage update는 세 단계다.

```
decompose V_re/V_im -> Va/Vm
apply dx to Va/Vm
reconstruct Va/Vm -> V_re/V_im
```

권장 구조는 `Va/Vm`을 authoritative state로 유지하고 단일 커널에서 active bus만 처리하는 것이다.

```
for k in pvpq:
    bus = pvpq[k]
    Va[bus] -= dx[k]              // RHS convention에 따라 부호 선택
    if bus is PQ:
        Vm[bus] -= dx[n_pvpq + pq_slot]
    V_re[bus], V_im[bus] = Vm[bus] * sincos(Va[bus])
```

부호 convention을 `J * dx = F`, `state -= dx`로 통일하면 cuDSS 전에 `-F`를 만들 필요가 없다.

---

## 부호 convention과 cuDSS RHS

현재 mismatch는 실질적으로 다음을 계산한다.

```
F = S_calc - S_spec
```

따라서 Newton 보정은 다음 두 방식이 동치다.

```
현재 방식: J * dx = -F, state += dx
권장 방식: J * dx =  F, state -= dx
```

권장 방식의 장점:
- FP64 cuDSS에서는 RHS matrix가 `d_F`를 직접 가리킬 수 있다.
- `d_F -> host -> -F -> device rhs` 왕복을 제거할 수 있다.
- Mixed FP32 cuDSS에서도 host 왕복 대신 device cast kernel로 `F64 -> rhs32`만 수행하면 된다.

문서와 코드 주석은 모두 `F = S_calc - S_spec` 기준으로 맞춘다.

---

## FP32 Ybus + FP64 V/Ibus 프로파일

초기 실험상 `Ybus`, `Sbus`, `Ibus`를 FP32로 유지해도 수렴한다는 판단이 있었지만,
Texas Univ B=1 sweep에서 `1e-8` 수렴 바닥이 남았다. 현재 CUDA Mixed profile은
`Sbus`, `Ibus`, `V_re/im`을 FP64로 되돌리고 `Ybus/J/dx`를 FP32로 유지한다.
별도 `cuda_batch` profile을 만들지 않고 기존 CUDA Mixed storage/op를 batch-aware로 확장한다.
전압 상태 `Va/Vm`과 rectangular cache `V_re/V_im`은 FP64로 유지한다.

권장 정밀도:

```
Ybus_re/im: float
Sbus_re/im: double
Ibus_re/im: double
Va/Vm:      double
V_re/im:    double
F:          double
J_values:   float
dx:         float
```

Mismatch 흐름:

```
V64    = Va/Vm에서 재구성된 V cache
Ibus64 = custom_batch_csr_spmv(Ybus32, V64)
F64    = V64 * conj(Ibus64) - Sbus64
normF  = max(abs(F64))
```

이 profile의 핵심 이득은 Ybus/J/dx bandwidth 감소와 Ibus 재사용이다. 지정 전력, Ibus 저장,
최종 수렴 판정과 전압 상태는 FP64로 남긴다.
엄격한 tolerance나 수치적으로 민감한 계통에서는 선택적으로 FP64 residual check path를 둔다.

검증 항목:
- 전체 케이스 수렴률
- 반복 횟수 증가량
- final mismatch
- 최종 전압 오차
- ill-conditioned 계통 fallback 필요 여부

---

## Ibus custom kernel

CUDA mismatch는 현재 F entry thread가 필요한 bus row를 직접 순회한다. 배치와 FP32 profile에서는 `Ibus`를 명시 버퍼로 두고, cuSPARSE SpMM이 아니라 custom batch CSR kernel을 기본 경로로 사용한다.

권장 stage:

```
kernel A: Ibus64 = custom_batch_csr_spmv(Ybus32, V64)
kernel B: F64 = V64 * conj(Ibus64) - Sbus64, norm reduction input 작성
kernel C: Jacobian off-diagonal fill
kernel D: Jacobian diagonal fill from V64, Ibus64, Ydiag32
```

기본 custom kernel은 batch와 bus row를 직접 매핑한다.

```
for each (batch b, bus i):
    acc = 0 + j0
    for k in row_ptr[i] .. row_ptr[i + 1]:
        j = col_idx[k]
        y = Ybus[k] or Ybus[b * nnz_Y + k]
        v = V[b * n_bus + j]
        acc += y * v
    Ibus[b * n_bus + i] = acc
```

초기 구현은 warp-per-(batch, bus)를 기본으로 한다. row length histogram을 보고 short row는 thread-per-bus, long row는 block-per-bus variant를 추가한다.

cuSPARSE SpMM은 비교 실험용 후보로 남긴다. Ybus 값이 batch 공통이고, `V`를 cuSPARSE-friendly complex dense matrix layout으로 둘 수 있으며, batch size가 충분히 큰 경우에만 우선 검토한다.

장점:
- mismatch와 Jacobian diagonal 계산이 Ibus를 공유한다.
- edge Jacobian의 diagonal atomic을 제거할 수 있다.
- diagonal fill을 direct write로 만들면 `d_J_values.memsetZero()` 제거가 가능해진다.

---

## Jacobian fill 개선

### Edge kernel

Off-diagonal write는 atomic이 필요 없다. `mapJ**[k]`가 Ybus entry별 고유 위치를 가리킨다면 direct write가 맞다.

```
if (map11[k] >= 0) J[map11[k]] = term_va_re
if (map21[k] >= 0) J[map21[k]] = term_va_im
if (map12[k] >= 0) J[map12[k]] = term_vm_re
if (map22[k] >= 0) J[map22[k]] = term_vm_im
```

대각은 여러 edge가 같은 위치에 기여하므로 atomic 또는 별도 diagonal kernel이 필요하다.

`d_J_values.memsetZero()`를 제거하려면 모든 J entry가 매 iteration 정확히 한 번 write되어야 한다. 따라서 edge kernel에서 memset을 완전히 없애려면 diagonal을 별도 direct-write kernel로 분리한다.

### Vertex kernel

현재 warp 하나가 active bus 하나를 맡는 구조는 vertex 구현 취지와 맞다. 다만 row length 분포에 따라 다음 adaptive variant를 검토할 수 있다.

```
short row:  thread-per-bus 또는 half-warp
normal row: warp-per-bus
long row:   block-per-bus
```

### Vm 재사용

Jacobian에서 `hypot(V_re, V_im)`를 반복하지 않는다. `Vm[j]` 또는 `invVm[j]`를 storage에 유지하고 kernel input으로 넘긴다.

---

## Stream 지원

stream을 제대로 쓰려면 storage와 모든 op가 같은 stream-aware 실행 모델을 따라야 한다.

필요 항목:

```
DeviceBuffer::assignAsync(..., stream)
DeviceBuffer::copyToAsync(..., stream)
DeviceBuffer::memsetZeroAsync(stream)
kernel <<<grid, block, 0, stream>>>
cuDSS handle/config/data가 같은 stream에서 실행되도록 설정
timing은 cudaDeviceSynchronize 대신 cudaEvent 기반으로 측정
```

`CUPF_ENABLE_TIMING`에서 stage마다 `cudaDeviceSynchronize()`를 호출하면 stream overlap이 깨진다. stream 지원과 함께 timer 정책을 바꾼다.

---

## Multi-batch 실행 모델

CUDA backend의 기본 메모리 레이아웃은 batch-major다. 기존 single-case API도 내부적으로는 `B=1`인 같은 layout을 사용한다.

```
V_re[b * n_bus + i]        // float
V_im[b * n_bus + i]        // float
Va[b * n_bus + i]          // double
Vm[b * n_bus + i]          // double
Ibus_re[b * n_bus + i]     // double
Ibus_im[b * n_bus + i]     // double
F[b * dimF + k]            // double
dx[b * dimF + k]           // float
J_values[b * nnz_J + p]    // float
```

Ybus values가 batch 공통이면 `Ybus_re/im[nnz_Y]` 하나만 둔다. batch별 값이 필요하면 `Ybus_re/im[B * nnz_Y]`를 둔다.

Jacobian과 linear solve는 cuDSS uniform batch에 맞춘다.

```
J_row_ptr:  [dimF + 1]       // batch 공통
J_col_idx:  [nnz_J]          // batch 공통
J_values:   [B * nnz_J]      // batch별 값
F/rhs:      [B * dimF]       // batch별 RHS
dx:         [B * dimF]       // batch별 solution
```

cuDSS descriptor는 uniform batch matrix를 표현하도록 만든다. 즉 batch별로 별도 sparse pattern descriptor를 만들지 않고, 같은 CSR structure 위에 batch-strided values/RHS/solution을 얹는다.

기본 구현:
- pattern과 maps는 batch 공통
- storage는 batch dimension을 가진 device buffers를 소유
- Jacobian fill kernel은 `J_values[b * nnz_J + p]`에 write
- linear solve는 cuDSS uniform batch factorize/solve 경로 사용
- active mask는 mismatch/update에는 바로 적용하고, cuDSS는 full uniform batch 또는 compacted active uniform batch 중 하나를 선택
- 별도 `CudaBatch*` class, `BatchExecutionPlan`, `BatchPlanBuilder`는 만들지 않는다. 기존 CUDA plan이 batch dimension을 갖고 `B=1`도 같은 경로로 실행한다.

수렴 처리:

```
active[b] = normF[b] > tolerance
inactive batch는 mismatch/update에서 skip
cuDSS uniform batch는 full batch를 유지하거나 active batch를 compact해서 호출
```

처음에는 full uniform batch를 유지하는 쪽이 단순하다. 배치가 크고 수렴 iteration 편차가 크면 active batch compaction을 추가한다.

---

## Host-device 왕복 제거

우선 제거할 왕복:

1. mismatch norm 계산을 위한 `d_F` 전체 다운로드
2. cuDSS RHS 부호 반전을 위한 `d_F` 다운로드와 `rhs` 업로드
3. solve마다 반복되는 임시 host vector allocation

권장 구조:

```
d_F 생성
device reduction으로 normF 또는 converged flag 계산
cuDSS RHS는 d_F 또는 device-cast rhs buffer 사용
host는 batch별 norm scalar만 필요할 때 복사
```

---

## 구현 우선순위와 현재 상태

1. 완료: `F = S_calc - S_spec`, `J * dx = F`, `state -= dx` CUDA convention 정리
2. 완료: FP64 cuDSS RHS host roundtrip 제거
3. 완료: Voltage update를 Va/Vm authoritative state 기반으로 변경
4. 완료: Mixed mismatch norm device reduction
5. 완료: Mixed edge off-diagonal direct write
6. 변경: Mixed `Ybus` FP32 layout, `Sbus`, `Ibus`, voltage cache는 FP64 유지
7. 완료: Ibus custom batch CSR kernel과 diagonal fill 분리
8. 진행 전: edge/vertex `d_J_values.memsetZero()` 제거
9. 진행 전: stream-aware DeviceBuffer와 op launch
10. 부분 완료: CUDA Mixed storage/op와 cuDSS FP32 uniform batch 경로 확장

현재 `solve_batch(batch_size > 1)`은 CUDA Mixed path에서 활성화되어 있다.
CPU FP64와 CUDA FP64 storage는 아직 `B=1` compatibility path다.

---

## 검증 계획

각 개선은 다음 기준으로 검증한다.

```
CPU FP64 reference 대비 final voltage 오차
CUDA 기존 profile 대비 iteration count
final mismatch와 convergence flag
J poison fill 검사: 모든 J entry가 write되는지 확인
edge/vertex Jacobian numerical diff
FP32 Ybus + FP64 V/Ibus profile의 실패 케이스 목록
batch size별 throughput, per-case latency
stream on/off timing 비교
```

특히 memset 제거 전에는 `J_values`를 NaN 또는 poison 값으로 채워 모든 entry가 매 iteration write되는지 확인한다.
