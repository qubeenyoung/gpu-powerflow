# cuPF Profiling Rules

- 작성일: 2026-04-10

## Timer Label Rules

- stage timer label은 고정 문자열을 쓴다.
- iteration index를 label 문자열에 직접 붙이지 않는다.
- 권장 label prefix는 `NR.`로 시작한다.

현재 사용 label:

- `NR.analyze.total`
- `NR.analyze.jacobian_builder`
- `NR.analyze.storage_prepare`
- `NR.solve.total`
- `NR.solve.upload`
- `NR.solve.download`
- `NR.iteration.total`
- `NR.iteration.mismatch`
- `NR.iteration.jacobian`
- `NR.iteration.linear_solve`
- `NR.iteration.voltage_update`

## Timing Rules

- wall-clock timer는 `ScopedTimer`를 사용한다.
- profiling build에서는 `ENABLE_TIMING=ON`으로 켠다.
- 로그를 켠 경우 출력 형식은 `[cuPF][timer] label=... elapsed_ms=...`를 유지한다.
- `timingSnapshot()`과 `resetTimingCollector()`로 누적 통계를 수집할 수 있다.

## Nsight Rules

- Nsight Systems / Compute 사용 자체에는 timer 내부 NVTX 의존이 필요하지 않다.
- NVTX range는 `utils/nvtx_trace.hpp`에서만 관리한다.
- profiling build에서 NVTX가 필요하면 `ENABLE_NVTX=ON`으로 켠다.
- 커널 variant 비교가 필요하면 kernel 이름과 stage label을 안정적으로 유지한다.
- timer와 NVTX는 서로 독립적으로 유지한다.

## Python Binding Rule

- profiling, timer, kernel 분석 흐름이 정리되기 전에는 Python 바인딩을 확장하지 않는다.
