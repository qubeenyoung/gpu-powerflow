# Multi-block big-front trailing for underfilled (B=1) levels

**작성일**: 2026-06-08
**범위**: big-tier 의 scalar(FP64/FP32) 경로를 underfill level 에서 panel/trailing/extend 3-kernel 로 분리.
  kernel internal 수학 / plan / etree / TC 경로 무수정.
**결과**: B=1 factor — USA FP64 **−58%**, 25K FP64 −25%, 8387 FP64 −12%, USA FP32 −23%. 정확성·batched 무영향.

## 문제: B=1 deep level 의 underfill

B=1 single-system 에서 factor 가 per-iteration 비용을 지배하고 (analyze 는 one-time),
factor 의 지배 tier 는 케이스마다 다르다 (nsys, FP64):

| case | small | mid | big |
|---|---|---|---|
| 8387 | 14% | 86% | 0% |
| 25K | 5% | 43% | **51%** |
| USA | 3% | 5% | **91%** |

USA 의 deep big level 은 front 가 극소수다 (= B=1 에서 block 수):

| level | front 수 | maxfsz |
|---|---|---|
| L20–L30 | 2–6 | 169–242 |
| L31–L39 | **1** | 6–149 |

RTX 3090 = 82 SM 인데 deep level 은 **1–6 block** 만 → SM 76–81 idle.
`factor_big` 는 1 front = 1 block 으로 panel LU + **trailing GEMM** + extend 를 융합하므로,
FLOP 이 큰 trailing 이 그 소수 block 에 직렬화된다.

precision 스윕 (USA B=1 factor: FP64 9.6 → FP32 2.6 → TF32 2.1) 이 4.5× 차이 → **trailing-compute 지배**
(panel-LU sync 아님) 를 확인. 즉 trailing 을 SM 에 펼치는 게 lever.

## 해결: trailing 을 multi-block 으로 분리

underfill level (`level_size × B < num_SMs`) 에서 fused `factor_big` 대신 3 kernel:

```
factor_big_panel<T>      Phase 1 (panel LU) + Phase 2 (U-solve), 1 block/front.
                         fsz ≤ 48 front 는 lu_small_front 로 융합 처리 (trailing kernel 이 skip).
factor_big_trailing_mb   Phase 3 trailing 을 blockIdx.z element-tile 로 분할.
                         grid (front, batch, tiles), tile = elems_per_block(=2048) C 원소.
factor_big_extend<T>     Phase 4 extend-add, 1 block/front.
```

race-free 근거: 각 C 원소 (ii,jj) 는 정확히 한 tile 만 write, L/U 는 read-only,
kernel-launch 경계가 panel → trailing → extend 순서 보장.

**gate 는 per-level**: filled level (batched 의 대부분) 은 fused kernel 그대로 → batched 무영향.
deep level (B=1 전부, 또는 batched 의 cnt=1 spine-근처 big level) 만 multi-block.

## 결과 (B=1 factor_ms, 5-trial median, no-MB baseline 대비)

| case | FP64 | FP32 |
|---|---|---|
| 8387 | **−12.3%** | +0.4% (big tier 미사용) |
| 25K | **−25.1%** | −1.3% |
| USA | **−58.3%** | **−23.4%** |

- FP64 가 win 이 큰 이유: 8-byte double 이라 mid shared budget 초과로 fsz 더 작은 front 도 big tier 로
  fall-through → big 경로 사용 빈도 높음. + scalar FP64 trailing 이 느려 underfill 영향이 큼.
- FP32 는 big tier 를 genuinely big front (USA) 에서만 사용 → USA 만 −23%.
- batched (B=64) 무영향~소폭 win: 25K −6.8%, USA −1.9% (cnt=1 deep level 이 B=64 에서도 underfill 로 잡힘).
- 정확성: relres FP64 ~e-13/e-14, FP32 ~e-3/e-5 — baseline 과 동일. `-DCLS_NO_BIG_MB` 로 A/B baseline 빌드.

## 구현

- `factorize/kernels.cuh`: `factor_big_panel<T>` / `factor_big_trailing_mb<T>` / `factor_big_extend<T>`.
- `factorize/dispatch.cuh`: `launch_factor_big_mb<T>()` + big tier FP64/FP32 분기에 `big_underfill` gate.
- dynamic smem 없음 → 등록 불필요.

## 남은 lever

- **TC 경로 multi-block**: 현재 tf32/fp16 big (factor_big_tf32_ptx / fp16_ptx) 은 multi-block 미적용.
  USA B=1 은 tf32(fused) 가 이미 2.1ms 라 scalar-MB(1.95) 와 비슷 — 둘을 합치면(tf32 trailing 을 MB 로)
  추가 win 가능. PTX trailing 의 tile 분할 필요.
- **panel kernel**: trailing 이 빨라진 뒤 deep level 의 panel LU (1 block, nc 순차) 가 상대적으로 커질 수 있음.
  multi-block panel 은 cross-block sync 필요 → 난이도 높음, 보류.
- case8387/25K/USA 검증. 다른 케이스 일반화 확인 권장.
