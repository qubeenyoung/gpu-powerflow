# cuPF Solve Speedup 요약

작성일: 2026-04-23

## 측정 범위

- 지표: solve-only 평균 시간, 단위는 ms.
- 가속비: `CPU reference solve 평균 / 성공한 GPU cuPF 중 가장 빠른 per-case solve 평균`.
- CPU reference와 CPU cuPF는 batch 1에서 10 repeats.
- GPU cuPF는 3 warmups, 10 repeats. MATPOWER는 batch 1, 4, 16, 64, 256 완료.
- 이 문서의 GPU 측정은 cuDSS MT auto를 켠 결과를 기준으로 한다.
- Texas 결과는 partial이다. CUDA/NVML이 내려가기 전 GPU batch 1, 4까지만 완료됐다.

## 버스 크기별 요약

### MATPOWER

| bus bin | used / total | CPU ref ms | CPU cuPF ms | GPU b1 ms | GPU best multi ms | GPU best overall ms | best overall batch counts | speedup vs CPU ref |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| <100 | 40/41 | 0.4679 | 0.0584 | 0.4857 | 0.0434 | 0.0434 | b256:40 | 10.77 |
| 100-999 | 10/10 | 2.6148 | 0.4753 | 0.6996 | 0.0989 | 0.0989 | b256:10 | 26.43 |
| 1k-9,999 | 22/22 | 38.9101 | 10.4338 | 2.2222 | 0.8502 | 0.8502 | b256:22 | 45.77 |
| 10k-49,999 | 3/3 | 231.4322 | 56.6472 | 6.632 | 3.8255 | 3.8255 | b256:3 | 60.5 |
| >=50k | 2/2 | 2081.8823 | 596.9388 | 29.1801 | 21.6482 | 21.6482 | b256:2 | 96.17 |

MATPOWER used count에서 제외된 케이스: case16am.

### Texas Partial

| bus bin | used / total | CPU ref ms | CPU cuPF ms | GPU b1 ms | GPU best multi ms | GPU best overall ms | best overall batch counts | speedup vs CPU ref |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| 100-999 | 3/3 | 4.5446 | 1.1942 | 0.7236 | 2.7478 | 0.7236 | b1:3 | 6.28 |
| 1k-9,999 | 4/4 | 66.9688 | 22.9091 | 3.1193 | 6.226 | 3.1193 | b1:4 | 21.47 |
| 10k-49,999 | 3/3 | 301.4345 | 86.959 | 7.1484 | 14.7617 | 7.1484 | b1:3 | 42.17 |
| >=50k | 2/2 | 2228.4733 | 641.6774 | 28.3675 | 50.2171 | 28.3675 | b1:2 | 78.56 |

## 가속비 Top 5

각 케이스에서 성공한 GPU batch 중 가장 빠른 값을 사용했다. Texas는 현재 batch 1 또는 4만 포함된다.

| rank | dataset | case | buses | best GPU batch | CPU ref ms | GPU cuPF ms | speedup |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | MATPOWER | case_ACTIVSg70k | 70000 | 256 | 1970.7911 | 19.411052 | 101.53x |
| 2 | MATPOWER | case_SyntheticUSA | 82000 | 256 | 2192.9736 | 23.88532 | 91.81x |
| 3 | Texas | Base_Eastern_Interconnect_515GW | 78478 | 1 | 2487.4715 | 29.8495 | 83.33x |
| 4 | Texas | case_ACTIVSg70k | 70000 | 1 | 1969.4751 | 26.8855 | 73.25x |
| 5 | MATPOWER | case_ACTIVSg25k | 25000 | 256 | 405.3352 | 5.666703 | 71.53x |

## MATPOWER 전용 Top 5

MATPOWER는 현재 cuDSS MT auto 재측정에서 모든 GPU batch가 완료된 데이터셋이다.

| rank | dataset | case | buses | best GPU batch | CPU ref ms | GPU cuPF ms | speedup |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | MATPOWER | case_ACTIVSg70k | 70000 | 256 | 1970.7911 | 19.411052 | 101.53x |
| 2 | MATPOWER | case_SyntheticUSA | 82000 | 256 | 2192.9736 | 23.88532 | 91.81x |
| 3 | MATPOWER | case_ACTIVSg25k | 25000 | 256 | 405.3352 | 5.666703 | 71.53x |
| 4 | MATPOWER | case9241pegase | 9241 | 256 | 166.0975 | 2.478963 | 67x |
| 5 | MATPOWER | case13659pegase | 13659 | 256 | 179.1767 | 3.348489 | 53.51x |

## 원본 파일

- `matpower_comparison_mt_20260423/solve_wide.csv`
- `texas_comparison_mt_partial_20260423/solve_wide.csv`
