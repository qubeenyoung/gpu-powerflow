# 09 — 정밀도 & 텐서코어

> **층위**: 상세. FP64/FP32/TF32 세 정밀도가 무엇을 다르게 하는지, TF32가 텐서코어를 쓰면서 FP32 정확도를 어떻게
> 회복하는지(Ozaki), 그리고 비-TF32 커널에 TC 코드가 **아예 들어가지 않게** 하는 컴파일 메커니즘.

---

## 1. 정밀도 매트릭스

| 모드 | front 저장 | Phase-3 trailing GEMM | accumulate | power-grid Jacobian 정확도 |
|---|---|---|---|---|
| `FP64` | FP64 | scalar FP64 | FP64 | ~1e-13 |
| `FP32` | FP32 | staged-scalar FP32 | FP32 | ~1e-4 |
| `TF32` | FP32 | TF32 PTX `mma.m16n8k8` (Ozaki 보정) | FP32 | ~1e-4 (TF32 rounding) |

- 정밀도는 `SolverConfig.precision`(또는 CLI `--precision`)으로만 고른다. tier·라우팅과 **직교**다.
- **front 저장**: FP64는 `d_front_batch`(double), FP32/TF32는 `d_front_batch_f`(float). 즉 TF32도 데이터는 FP32다 —
  바뀌는 건 trailing GEMM을 텐서코어로 돌리느냐다.
- TF32가 권장인 이유와 정직한 천장(best-vs-best ~1.1×, low-fill에선 net≈0)은 [`../README.md` §8](../README.md).

## 2. TF32 trailing — PTX mma + Ozaki 보정

trailing(Schur GEMM `CB = A22 − L21·U12`)은 factor FLOP의 대부분이라, 이걸 텐서코어로 돌리는 게 TF32 모드의 핵심이다.

문제: TF32는 가수 10비트라 그냥 쓰면 relres가 ~1e-2로 나빠진다. 해결: **Ozaki 2성분(head/tail) 곱**.

```
각 FP32 값 x 를 TF32 두 조각으로:  x ≈ head(x) + tail(x)     (Tf32OzakiPair)
곱 a·b 를 4-pass 로 누적(CLS_MMA_TF32_OZAKI2):
    head(a)·head(b) + tail(a)·head(b) + head(a)·tail(b) + tail(a)·tail(b)
→ TF32 텐서코어로 ~FP32 정확도 회복
```

- mma 호출은 `front_ops.cuh`의 인라인 PTX 매크로 `CLS_MMA_TF32_M16N8K8`(한 mma) / `CLS_MMA_TF32_OZAKI2`(위 4-pass).
  이건 내부 헬퍼지 사용자 노브가 아니다.
- Ozaki 4-pass는 TF32 경로에 **상시 컴파일**(별도 빌드 플래그 없음). no-pivot factor + selinv라 이 보정이 필수다.
- mid는 `BlockUpdateTf32Tc`, big은 `TrailingUpdateTf32Tc`, B=1 big은 전용 `FactorSingleBigTrailTf32`가 같은 mma를 쓴다.

## 3. 비-TF32 커널엔 TC 코드가 없다 — `template<bool UseTC> + if constexpr`

요구사항: **TF32 선택 시 TC 경로가 살아있고, FP32/FP64면 TC 코드가 아예 없어(추가 레지스터 0)** 한다. 이건
런타임 `if`가 아니라 **컴파일타임 분기**로 달성된다.

```cpp
template <typename T, bool UseTC>
__global__ void FactorMid(...) {
  ...
  if constexpr (UseTC) {            // UseTC=false 인스턴스에선 이 블록이 통째로 소거됨
    if (nc <= kTensorCorePivotColumnCap && uc <= kTensorCoreUcCap)
      FactorizeFrontBlockedTf32(...);   // TF32 mma 경로
  }
  ... // scalar 경로
}
```

디스패처(`DispatchFactorMid`)가 정밀도로 인스턴스를 고른다:

```
FP64 → FactorMid<double, false>   ┐  UseTC=false →  if constexpr(UseTC) 블록이
FP32 → FactorMid<float , false>   ┘  컴파일 단계에서 제거 → TC 코드/레지스터 0
TF32 → FactorMid<float , true >   →  TC 경로 활성
```

`if constexpr`이므로 FP32/FP64 커널 바이너리엔 TF32 mma 코드가 **물리적으로 들어가지 않는다**(런타임 `if`였다면
dead-path 레지스터를 예약했겠지만 그렇지 않다). `FactorBig`(`big.cuh`)도 동일한 `if constexpr (UseTC)` 패턴이다.
즉 정밀도별 분리는 [10] 같은 매크로가 아니라 이 템플릿 + `if constexpr`이 한다.

## 4. TF32 적격 — 모양이 staging 예산에 맞는가 (`kTensorCoreUcCap`)

TC trailing은 padded L/U를 shared에 staging하므로 모양이 예산에 맞아야 한다. 게이트는 **두 조건뿐**:

```cpp
nc <= kTensorCorePivotColumnCap (=32)   &&   uc <= kTensorCoreUcCap (=512)
```

- `kTensorCoreUcCap`은 매크로가 아니라 `types.hpp`의 **HW 유도 constexpr**다: whole-front staging
  `(2·uc_pad·nc_pad + 4·nc_pad)·4B`가 99 KiB opt-in shared(`kDynamicSharedMemoryOptInBytes`)에 들어가야 하므로
  `uc_pad·nc_pad ≤ ~12.6K`. nc cap 32에서 512가 안전. **shared 예산이 바뀌면 재유도해야 한다**(주석 명시).
- 과거의 `fsz>32`/`nc≥1`/`uc≥1` 하한 게이트는 제거됐다(mid는 이미 fsz∈[33,64], nc≥1; uc=0 front는 TC 경로에서
  trailing 0회 no-op라 scalar와 동일). 정리 이력은 [03 §4](03-api-config-build.md).

## 5. 단일 시스템(B=1)의 TF32

B=1 big trailing은 배치 커널을 재사용하지 않고 전용 `FactorSingleBigTrailTf32`(shared-staged Ozaki mma)를 쓴다. front당
1 block 융합 체제([08 §4](08-runtime-and-batching.md))에 맞춰진 형태다. small/mid는 B>1과 같은 mma 프리미티브를 공유한다.
