# Sub-warp tiling for the small tier (factor + solve)

**작성일**: 2026-06-08
**범위**: small-tier kernel 내부 lane 할당 변경 (factor + solve). tier 선택 / plan / etree 무수정.
**결과**: case8387 FP64, B≥64 에서 **F+S −14~16%**, B=1 중립, 정확성 보존. default-on.
**관계**: `23-fsz-band-split-dispatch` (tier-split) 와 **직교 + compose**. `22-fp16-ptx-default` 와 무관 (다른 tier).

## 문제

small tier (max_fsz ≤ SMALL_THRESH=32) 는 panel 수의 대부분을 차지한다. case8387 (cap=8) 분포:
fsz ≤ 16 = 7134 panel (96.0%), fsz ≤ 48 = 99.7%, fsz > 48 = 0.26%.

기존 `factor_small<T>` / `solve_*_small<T>` 는 **1 warp = 1 front** (32 lane). fsz=8 front 는 64 element 를
32 lane 으로 처리 → lane 절반 이상이 idle. 게다가 small tier 는 **memory-latency bound**
(docs/16: scoreboard 203%, DRAM 23~32% = 대역폭 아닌 latency).

### 천장 측정 (nsys, 결정성 etree, no-graph)

| | factor_small | solve_*_small | 합산 F+S 비중 |
|---|---|---|---|
| B=1 | 18% of factor | 13% of solve | **16%** |
| B=64 | 24% of factor | **43% of solve** | **31.6%** |

small tier 는 FLOP 의 9% 인데 **wall 의 16~31%** → **2~3.5배 비효율**. 이 초과분이 회복 대상.

## 해결: sub-group lane packing

1 warp 를 **SG-lane sub-group 으로 쪼개 32/SG 개 front 를 동시에** 처리한다.
- SG = level 의 max_fsz 로 선택: `max_fsz ≤ 8 → 8` (4 front/warp), `≤ 16 → 16` (2 front/warp), `else → 32` (기존 1 front/warp).
- 효과: (a) lane 100% 활용 (idle 제거), (b) warp 당 독립 front 가 FPW=32/SG 개 → memory-level
  parallelism FPW 배 → latency 숨김.

### occupancy gate (B=1 회귀 차단)

packing 은 warp 수를 FPW 배 줄인다 → grid 가 작으면 underfill. B=1 에서 +4.5% 회귀 측정.
`factor_warp_fill()` = SMs × (maxThreadsPerSM/32) (GPU 를 꽉 채우는 warp 수, RTX 3090 ≈ 3936)
기준으로 **packed warp 수가 GPU 를 채울 때만 packing**, 아니면 SG=32 유지:

```
SG_cand = pick by max_fsz
if  (level_size × B) / FPW  <  warp_fill   →  SG = 32   (packing 안 함)
```

이 gate 는 tier-split (doc 23) 의 occupancy gate 와 **같은 HW 양** 을 재사용한다. B<4 에서 자동으로 SG=32.

## 결과 (case8387, FP64, production parallel-ND, median)

**per-kernel A/B** (결정성 etree, factor_mid / 일반 solve sanity ±0.1% → 귀속 신뢰):

| | factor_small | solve_*_small |
|---|---|---|
| B=64 | **−13%** | **−27%** |

**end-to-end F+S** (tier-split ON 위에서, base = SG=32):

| B | factor | solve | **F+S** |
|---|---|---|---|
| 1 | 중립 (gate→SG32) | | **~0% (노이즈)** |
| 64 | −6.5% | **−22%** | **−14.2%** |
| 256 | −9% | **−25%** | **−15.2%** |

정확성: relres fp64 4~7e-14, fp32 ~3e-5 — baseline 과 동일 (전 B·정밀도).

### 왜 solve 가 더 큰가

solve 의 small 경로 (substitution + `bwd_cb_subtract`) 가 factor 보다 lane idle 이 심해
packing 회복분이 약 2배. 고-B 에서 solve_*_small 이 solve 의 43% 라 lever 가 더 크다.

### tier-split 과 compose

tier-split (doc 23) 은 **launch 단위** 로 front 를 size-homogeneous range 로 분리한다.
sub-warp 는 **warp 내부** lane 할당을 고친다. tier-split 이 동질 small sub-tier 를 주면 SG 선택이
더 타이트해진다 (tiny sub-tier → SG=8). 위 F+S 측정이 tier-split ON 위에서 나온 추가 win 이다.
직교 lever 라 둘 다 켜고 쌓인다.

## 구현

- `factorize/phases.cuh` `lu_small_warp<FT, SG>` : stride 32→SG, `__syncwarp(mask)`.
- `factorize/kernels.cuh` `factor_small<FT, SG>` : sub-group 인덱싱 (sg = lane/SG, sl = lane%SG, mask).
- `solve/phases.cuh` `fwd_substitute<T,SG>` / `bwd_substitute<T,SG>` / `bwd_cb_subtract<T,SG>` :
  broadcast / reduction 의 shfl width 를 SG 로, sub-group mask 사용. SG=32 default 는 기존 full-warp 와 동일.
- `solve/kernels.cuh` `solve_fwd_small<T,SG>` / `solve_bwd_small<T,SG>` : sub-group 인덱싱.
- `factorize/dispatch.cuh`, `solve/dispatch.cuh` : `*_small_sg()` (SG + gate) + `launch_*_small_t()` 헬퍼.
- `multifrontal.cu` : `factor_small<FT,SG>` 6 instance smem attribute 등록.

SG=32 인스턴스는 원본과 bit-동일 → `-DCLS_SMALL_SG32_ONLY` 로 A/B baseline 빌드 가능.

## 학습 / 남은 것

- **"occupancy 올린다" 는 잘못된 framing 이었다**: small tier 는 이미 occupancy ceiling (docs/18 EXP-D 가
  warp 늘려도 회귀 확인). 진짜 lever 는 resident warp 수가 아니라 **per-warp lane 효율 + MLP**.
- docs/11·12 의 폐기된 small packing 은 **launch 폭증** 이 원인이었다. sub-warp 는 같은 grid/launch 안
  intra-warp 변경이라 그 실패를 피한다.
- case8387 단일 검증. 구조적 lever (tiny-front packing) 라 모든 power-grid 에 일반화 예상이나
  case9241 / 13659 / USA 로 확인 권장.
