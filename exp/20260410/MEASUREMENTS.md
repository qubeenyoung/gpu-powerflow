# 2026-04-10 Timing Measurements

수정한 `ScopedTimer` collector를 사용한 측정 결과 메모.

## 기준 run

- 신뢰할 기준 run:
  - `exp/20260410/results/timing_probe_118_9241_r3_gpu2_20260410/`
  - warmup `1`, measured repeats `3`
  - cases: `118_ieee`, `9241_pegase`
  - GPU: `CUDA_VISIBLE_DEVICES=2`

- 세부 breakdown 요약:
  - `exp/20260410/results/timing_probe_118_9241_r3_gpu2_20260410/TIMING_BREAKDOWN.md`
  - 단위는 `ms`

## 참고 run

- smoke check:
  - `exp/20260410/results/timing_smoke_20260410/`
  - parser / raw JSON / summary.csv 형식 확인용

- noisy reference:
  - `exp/20260410/results/timing_probe_118_9241_r3_20260410/`
  - GPU `3`에서 수행
  - 측정 후 `nvidia-smi` 확인 결과 GPU `3`에 다른 compute process가 상주하고 있었음
  - 따라서 반복 편차가 커서 기준값으로 쓰기보다 참고용으로만 보는 편이 맞음

## 빠른 관찰

- `TIMING_BREAKDOWN.md`는
  - `exp/20260410/scripts/write_timing_breakdown.py`
  - 로 `summary.csv`에서 재생성 가능

- TODO 1 (`analyze`) 관점:
  - warm run 기준으로 CUDA `analyze`는 `cudssAnalysis` 비중이 가장 큼
  - `118_ieee`에서는 약 71%
  - `9241_pegase`에서는 약 83%

- TODO 3 (`REFACTORIZATION` vs `SOLVE`) 관점:
  - CUDA `solveLinearSystem` 내부에서는 작은 케이스(`118_ieee`)는 `REFACTORIZATION`과 `SOLVE`가 비슷함
  - 큰 케이스(`9241_pegase`)는 `REFACTORIZATION`이 더 큼
  - CPU optimized는 큰 케이스에서 `factorize`가 거의 대부분을 차지함
