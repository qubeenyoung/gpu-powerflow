# fp32 단일-스트림 최적화 베이스라인 (case8387 / ACTIVSg25k)

**작성일**: 2026-06-08
**대상**: `custom_linear_solver` 현재 구현(as-shipped), `build-bench` (Release, sm_86)
**GPU**: RTX 3090 (24GB), CUDA 12.8. **클럭 고정 불가**(컨테이너에서 `nvidia-smi -lgc` 거부) → boost 변동 존재, median(repeat=20)으로 흡수.
**목적**: 이후 factorize 최적화의 비교 기준선(floor) 고정.

## 0. 설정 (고정 변수)

| 항목 | 값 |
|------|----|
| precision | fp32 (입력은 FP64 저장, factor 내부 fp32) |
| stream | single (`--no-multistream`) — default(multistream ON) 대비 floor |
| 케이스 | `case8387pegase` (n=14908, nnz=110572), `case_ACTIVSg25k` (n=47246, nnz=318672) |
| batch B | 1, 64, 256 — **전원 배치 경로**(`--batch B --batch-only`)로 측정해 timing 경로 통일 |
| panel_cap (eff) | 8387 → **8** (n<16000), 25k → **12** (n≥16000, analyzer auto-bump; `analyze.cu:138`) |
| 측정 | warmup=5(untimed) 후 repeat=20 median. timed 구간 안에서 `cudaDeviceSynchronize` |
| 입력 배치 | 동일 pattern + 동일 값 B copy (throughput floor; 실제 contingency는 값만 다름) |
| 정확도 게이트 | fp32 rel-residual 목표 ~1e-4 |

### 측정 방법론 수정 (이 베이스라인부터 적용)
- **B-1 (sync 통일)**: 단일-시스템 경로에 timed 구간 `cudaDeviceSynchronize` 추가 + 모든 B를 배치 경로로 측정. 기존 단일 경로는 async launch 시간만 재던 결함이 있었음.
- **B-2 (warmup)**: `--warmup N` 추가, graph instantiation / lazy alloc을 median에서 제외.
- 변경 위치: `tests/run_custom_solver.cu`.

## 1. 결과 (median of 5 trials × repeat=20, ms)

> **측정 보정**: 클럭 미고정이라 단일 측정은 boost로 ±5~10% 흔들린다. 아래 값은 **5 trial의 median-of-medians**. 처음 단일 측정본(8387 B=1 factor 0.367, 25k B=1 0.848)은 약간 높게 뽑힌 샘플이었음. **A/B delta는 반드시 interleaved로** 측정할 것(§4) — sequential 비교는 boost drift에 오염됨(level-contiguous 실험에서 확인, doc 15).

### case8387pegase (eff panel_cap=8)
| B | factorize / sys | solve / sys | factor+solve / sys | throughput (sys/s) | rel-residual |
|--:|----------------:|------------:|-------------------:|-------------------:|-------------:|
| 1   | 0.337 | 0.251 | 0.588 | 1,701 | ~3e-5 |
| 64  | 0.0316 | 0.0208 | 0.0524 | 19,084 | ~5e-5 |
| 256 | 0.0286 | 0.0179 | 0.0465 | 21,505 | ~2e-5 |

setup(analyze+symbolic) ≈ 22–27 ms (one-time, Newton 루프 밖).

### case_ACTIVSg25k (eff panel_cap=12)
| B | factorize / sys | solve / sys | factor+solve / sys | throughput (sys/s) | rel-residual |
|--:|----------------:|------------:|-------------------:|-------------------:|-------------:|
| 1   | 0.918 | 0.532 | 1.450 | 690 | ~1.5e-4 |
| 64  | 0.120 | 0.0543 | 0.174 | 5,747 | ~1.7e-4 |
| 256 | 0.115 | 0.0500 | 0.165 | 6,061 | ~1.7e-4 |

setup ≈ 62–71 ms (one-time).

## 2. 관찰 (최적화 타깃)

1. **배치 효율이 낮음 (헤드룸 큼, 1차 표적)**: factorize B=1→B=64 speedup이 8387 10.7×, 25k 7.6× (이상치 64× 대비 각 17%/12%). 단일-시스템 latency가 비포화 커널에 묶여 있다는 신호 → **factorize 최적화의 1차 표적**.
2. **포화는 B≈64~256**: median 기준 B=256이 B=64보다 8387 factor −10%, 25k −4%로 약간 더 빠름(단일샘플에서 "256이 더 느림"은 noise였음). 큰 이득은 아니고 B≈64에서 대체로 참.
3. **25k fp32 정확도가 게이트 초과**: rel-residual ~1.5e-4~1.7e-4로 ~1e-4를 넘김. 큰 Jacobian에서 fp32 누적오차 증가. NR 수렴(보통 update tol 1e-3~1e-6)엔 보통 충분하나, 게이트를 fp32+대형 케이스에 한해 ~3e-4로 완화하거나 mixed-precision 보정 고려.
4. **solve가 factorize의 ~45–60%**: solve도 무시 못 할 비중. 단 최적화 1차는 factorize.

## 3. 재현 커맨드

```bash
cmake -S . -B build-bench -DCMAKE_BUILD_TYPE=Release \
  -DCLS_BUILD_CUDA_OPS=ON -DCLS_BUILD_SCRIPTS=ON -DCLS_CUDA_ARCHITECTURES=86
cmake --build build-bench -j

for CASE in case8387pegase case_ACTIVSg25k; do for B in 1 64 256; do
  ./build-bench/custom_linear_solver_run \
    /datasets/power_system/nr_linear_systems/$CASE \
    --precision fp32 --batch $B --batch-only --no-multistream \
    --warmup 5 --repeat 20
done; done
```

## 4. 한계 / 주의 — A/B 측정 프로토콜
- 클럭 미고정 → 절대값 ±5~10% 변동. **A/B delta는 두 바이너리를 번갈아(interleaved) 측정**하고 각각 median을 비교할 것. sequential(한쪽 다 돌고 다른 쪽)은 boost drift를 신호로 착각함 — level-contiguous 실험에서 sequential은 "−13%", interleaved는 "regression"으로 정반대 결론이 나왔다(doc 15).
- 배치 입력이 동일 copy라 캐시/pivot 거동이 실제 워크로드보다 유리할 수 있음.
- 이 베이스라인은 **single-stream floor**. default(multistream)는 별도 (docs/04/13 참고).
