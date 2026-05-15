# 2026-04-23 Experiment Index

오늘 실행은 크게 보면 최종 논문용 2개, 참고용 1개, 보관용 중간 실행들로 나뉜다.

## 논문에 의미 있는 실행

| 구분 | 결론 | 볼 파일 |
|---|---|---|
| MATPOWER solve speed | CPU reference / CPU cuPF / GPU cuPF를 solve-only로 비교한 최종 속도 실험. GPU는 cuDSS MT auto를 켠 재측정 결과이며 batch 1, 4, 16, 64, 256을 모두 포함한다. | `paper/solve_speedup_report_20260423.md`, `paper/speed_matpower_final/matpower_comparison_mt_20260423/solve_bin_summary.md` |
| MATPOWER precision | GPU batch 1에서 FP32 / Mixed / FP64를 공정하게 다시 잰 최종 정밀도 실험. timing은 fixed 10 Newton updates, mismatch는 tolerance 1e-8 convergence run 기준이다. | `paper/precision_matpower_final/matpower_precision_fp32_mixed_fp64_fair_analysis_20260423/fair_precision_bin_summary.md` |
| Texas speed partial | Texas 데이터에 대한 참고 실험. CPU와 GPU batch 1, 4는 있으나 CUDA/NVML 문제로 GPU 전체 batch가 끝나지 않아 paper main 결과로 쓰기에는 partial이다. | `paper/speed_texas_partial/texas_comparison_mt_partial_20260423/solve_bin_summary.md` |

## 최종 raw run 위치

- MATPOWER speed raw:
  - `paper/speed_matpower_final/matpower_cpu_ref_cpu_b1_w0_r10_20260423`
  - `paper/speed_matpower_final/matpower_gpu_mt_b1_w3_r10_20260423`
  - `paper/speed_matpower_final/matpower_gpu_mt_b4_w3_r10_20260423`
  - `paper/speed_matpower_final/matpower_gpu_mt_b16_w3_r10_20260423`
  - `paper/speed_matpower_final/matpower_gpu_mt_b64_w3_r10_20260423`
  - `paper/speed_matpower_final/matpower_gpu_mt_b256_w3_r10_20260423`
- MATPOWER precision raw:
  - `paper/precision_matpower_final/matpower_precision_fp32_mixed_fp64_fixed10updates_b1_w3_r10_20260423`
  - `paper/precision_matpower_final/matpower_precision_fp32_mixed_fp64_convergence_b1_tol1e-8_maxit10_w3_r10_20260423`
- Texas partial raw:
  - `paper/speed_texas_partial/texas_cpu_ref_cpu_b1_w0_r10_20260423`
  - `paper/speed_texas_partial/texas_gpu_mt_b1_w3_r10_20260423`
  - `paper/speed_texas_partial/texas_gpu_mt_b4_w3_r10_20260423`
  - `paper/speed_texas_partial/texas_gpu_mt_b16_w3_r10_20260423`

## 보관용 실행

- `archive/superseded_speed_no_mt`: cuDSS MT auto를 켜기 전 GPU speed 실행. 최종 speed 비교에서는 쓰지 않는다.
- `archive/precision_diagnostics`: FP32/FP64 초기 측정, Mixed 추가 측정, operator probe 등 정밀도 실험 설계 중 확인한 중간 산출물. 최종 precision 표에는 쓰지 않는다.

## 논문 표에 바로 쓸 핵심 파일

- Speed bin summary: `paper/speed_matpower_final/matpower_comparison_mt_20260423/solve_bin_summary.md`
- Speed top cases: `paper/solve_speedup_report_20260423.md`
- Precision bin summary: `paper/precision_matpower_final/matpower_precision_fp32_mixed_fp64_fair_analysis_20260423/fair_precision_bin_summary.md`
