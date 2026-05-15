# cuPF 단계별 파이차트 점검 보고

- 대상 케이스: `case13659pegase`
- 반복: CPU reference 10회, cuPF GPU 10회
- CPU reference: `exp/20260511/benchmarks/utils.py`의 Python/SciPy Newton 경로를 stage별로 계측
- cuPF GPU: `cuPF/build/bench-operators/benchmarks/cupf_case_benchmark`, `cuda_mixed_edge`, `ENABLE_TIMING=ON`

## 결론

파이차트는 만들 수 있다. 다만 현재 20260511 벤치마크에서 직접 측정되는 것은 `MATPOWER` 기준 경로와 `cuPF` GPU 경로 두 끝점이다. 중간 단계인 `선형계 GPU 가속`, `선형계 + 자코비안 GPU 가속`은 같은 케이스에서 측정한 CPU/GPU component timing을 한 항목씩 교체한 합성 추정치다.

진짜 ablation으로 논문용 수치를 만들려면 현재 20260511 흐름에 `cuda_wo_cudss`, `cuda_wo_jacobian` 같은 전용 측정 경로를 추가하는 편이 맞다.

`outer / transfer`는 파이차트와 단계별 총 시간에서 제외했다. 대신 `Other NR ops`를 `Ibus`, `Mismatch`, `Mismatch norm`, `Voltage update`로 펼쳤다.

## 단계별 결과

| 단계 | 총 시간 | CPU reference 대비 |
| --- | ---: | ---: |
| MATPOWER | 189.281 ms | 1.00x |
| 선형계 GPU 가속 | 53.323 ms | 3.55x |
| 선형계 + 자코비안 GPU 가속 | 9.295 ms | 20.36x |
| cuPF 전체 구조 | 6.571 ms | 28.80x |

## component 평균

| 경로 | 선형계 | 자코비안 | Ibus | Mismatch | Mismatch norm | Voltage update | 제외된 outer/transfer | 측정 solve |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MATPOWER | 141.762 ms | 44.110 ms | 0.000 ms | 1.515 ms | 0.000 ms | 1.895 ms | 0.258 ms | 189.540 ms |
| cuPF | 5.803 ms | 0.082 ms | 0.340 ms | 0.053 ms | 0.221 ms | 0.070 ms | 0.908 ms | 7.480 ms |

## 산출물

- `stage_pies_case13659pegase.png`
- `stage_pies_case13659pegase.pdf`
- `stage_pie_data.csv`
- `cpu_reference_timing.csv`
- `gpu_timing.csv`

그림은 4개 파이를 가로로 배치했고, 전체 타이틀은 제거했다. 각 파이의 단계 레이블은 상단 박스, 실행시간은 하단 박스에 배치했다. 파이차트 크기와 그림 크기는 유지했고, 레이블/실행시간/범례/% 텍스트는 1.5배 키웠다. 작은 조각의 `%`는 큰 글자에서 겹치지 않도록 5% 이상일 때만 표시한다.
