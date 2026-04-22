# cuPF — Newton-Raphson 전력조류 수학 배경

## 1. 기본 방정식

전력 계통의 모선 어드미턴스 행렬 **Y**bus가 주어질 때, 각 버스 i의 복소 전력 주입은 다음과 같다.

```
S_i = V_i · conj(I_i)
    = V_i · conj( Σ_j Y_ij · V_j )
```

여기서 V_i = |V_i| · e^{jθ_i} 는 복소 전압, Y_ij 는 Ybus의 (i,j) 원소다.

전력조류 문제는 지정된 전력 S_spec을 만족하는 전압 벡터 V를 구하는 것이다.

---

## 2. Newton-Raphson 알고리즘

### 미스매치 벡터 F

지정값(spec)과 계산값(calc)의 차이를 F 벡터로 정의한다.

```
mis_i = S_calc_i - S_spec_i = V_i · conj(I_i) - Sbus_i

F = [ ΔP_pv ]     ΔP_i = Re(mis_i)
    [ ΔP_pq ]     ΔQ_i = Im(mis_i)
    [ ΔQ_pq ]
```

버스 종류에 따라 포함 여부가 다르다.

| 버스 유형 | θ 제어 | \|V\| 제어 | ΔP 포함 | ΔQ 포함 |
|----------|-------|-----------|---------|---------|
| Slack    | ✓     | ✓         | ✗       | ✗       |
| PV       | ✗     | ✓         | ✓       | ✗       |
| PQ       | ✗     | ✗         | ✓       | ✓       |

dimF = n_pvpq + n_pq = n_pv + 2·n_pq

### NR 선형화 및 갱신

각 반복에서 다음 선형 시스템을 푼다.

```
J · dx = F

J = [ ∂P/∂θ     ∂P/∂|V| ] = [ J11  J12 ]
    [ ∂Q/∂θ     ∂Q/∂|V| ]   [ J21  J22 ]
```

dx를 적용해 전압을 갱신한다.

```
θ_pv  ← θ_pv  - Δθ_pv      (dx[0, n_pv))
θ_pq  ← θ_pq  - Δθ_pq      (dx[n_pv, n_pvpq))
|V|_pq ← |V|_pq - Δ|V|_pq  (dx[n_pvpq, dimF))
```

수렴 조건: `max|F_i| ≤ tolerance`

---

## 3. Jacobian 편미분 공식 (극좌표계)

S_i = V_i · conj(I_i) 이고 I_i = Σ_j Y_ij · V_j 이므로,

### 오프 대각 원소 (i ≠ j)

```
∂S_i/∂θ_j  = -j · V_i · conj(Y_ij · V_j)
∂S_i/∂|V_j| =      V_i · conj(Y_ij · V̂_j)    (V̂ = V/|V|)
```

Jacobian에 기록되는 값:
```
J11[i,j] = Re(∂S_i/∂θ_j)   = ∂P_i/∂θ_j
J21[i,j] = Im(∂S_i/∂θ_j)   = ∂Q_i/∂θ_j
J12[i,j] = Re(∂S_i/∂|V_j|) = ∂P_i/∂|V_j|
J22[i,j] = Im(∂S_i/∂|V_j|) = ∂Q_i/∂|V_j|
```

### 대각 원소 (i = i)

```
∂S_i/∂θ_i  = +j · V_i · conj(I_i)
∂S_i/∂|V_i| =      conj(I_i) · V̂_i
```

대각 항에는 자기 어드미턴스(Y_ii · V_i)와 모든 이웃(Y_ij · V_j, j≠i)의 기여가 합산된다.

---

## 4. Jacobian 희소 구조

Ybus와 Jacobian의 희소 패턴은 망 위상(network topology)에서 직접 결정된다.
버스 i와 j 사이에 Y_ij ≠ 0이면 J의 해당 블록 위치가 채워진다.

```
(Y_i, Y_j) ≠ 0
 → Ji_pvpq ∈ [0, n_pvpq)  : J11, J21에 행 Ji_pvpq 존재
 → Ji_pq   ∈ [n_pvpq, dimF) : J21, J22에 행 Ji_pq 존재
 → Jj_pvpq ∈ [0, n_pvpq)  : J11, J12에 열 Jj_pvpq 존재
 → Jj_pq   ∈ [n_pvpq, dimF) : J12, J22에 열 Jj_pq 존재
```

`JacobianPatternGenerator`와 `JacobianMapBuilder`는 이 패턴을 한 번 분석하고, Ybus 원소 k번째가 J의 어느 CSR 위치에 기여하는지를 `mapJ11/12/21/22`에 기록한다. 이후 NR 반복마다 이 맵으로 직접 scatter한다.

---

## 5. 복소수 연산의 실수 분리 (CUDA 커널)

CUDA 커널은 복소수를 실수·허수로 분리해 FP64(또는 FP32)로 연산한다.

`term_va = -j · V_i · conj(Y_ij · V_j)` 를 실수로 전개하면:

```
curr_re = yr·vj_re - yi·vj_im     # Re(Y_ij · V_j)
curr_im = yr·vj_im + yi·vj_re     # Im(Y_ij · V_j)

# -j · V_i = V_im - j·V_re
neg_j_vi_re =  vi_im
neg_j_vi_im = -vi_re

# (a+jb)·conj(c+jd) = (ac+bd) + j(bc-ad)
term_va_re = neg_j_vi_re·curr_re + neg_j_vi_im·curr_im   # J11
term_va_im = neg_j_vi_im·curr_re - neg_j_vi_re·curr_im   # J21
```

`term_vm = V_i · conj(Y_ij · V̂_j)`:

```
scaled_re = curr_re / |V_j|
scaled_im = curr_im / |V_j|

term_vm_re = vi_re·scaled_re + vi_im·scaled_im   # J12
term_vm_im = vi_im·scaled_re - vi_re·scaled_im   # J22
```

---

## 6. Mixed 정밀도 전략

Mixed 모드는 수렴 신뢰성을 유지하면서 연산 속도를 높이기 위해 stage별로 정밀도를 선택한다.

| Stage          | 정밀도 | 이유 |
|---------------|--------|------|
| Mismatch      | FP64 Ybus/V/Ibus/Sbus/F | 전압, Ybus, Ibus 저장, 지정 전력, normF는 FP64 |
| Jacobian fill | FP64 Ybus/voltage/Ibus + FP32 J | 입력은 FP64 buffer에서 읽고 FP32 산술로 `J_values`를 기록 |
| Linear solve  | FP32   | cuDSS FP32로 충분히 수렴 가능 |
| Voltage update| FP64 Va/Vm / FP64 V cache / FP32 dx | 상태 벡터와 mismatch/Jacobian voltage cache는 FP64, correction만 solve 타입 |

공통 convention은 `J * dx = F`를 풀고 dx를 FP64 Va/Vm에서 빼는 것이다.
전압 cache `V_re/V_im`은 FP64로 재구성하고, Jacobian은 FP64 Ybus/전압/Ibus 입력을 커널 안에서 FP32로 변환한다.
최종 public output은 FP64 Va/Vm에서 만든다.
