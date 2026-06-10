# cuPF × custom solver — FP32 / TF32 / FP16 end-to-end (Newton-Raphson)

**작성일**: 2026-06-09
**범위**: custom_linear_solver 를 cuPF NR 백엔드로 연동, **fp32/tf32/fp16** 전체 전력조류
수렴까지 측정 + **70K 에서 cuDSS 와 직접 비교**. identical replicas (`CUPF_BENCH_SCALE_STEP=0`,
cuDSS 레퍼런스와 동일). 빌드 `-DCUPF_ENABLE_CUSTOM_SOLVER=ON -DBUILD_EVALUATORS=ON`.
custom = no-MB default(note 30) + LDB fix. 러너 `cupf_graph_bench <dump> <cudss|custom> <B,…>`.

> **수렴 판정**: `iters < 30`. 30 = max-iter cap 도달 = **발산**(relmis 가 tol 1e-8 에 못 미침).

## 0. 연동 상태
옛 `cls::batched` API 참조로 빌드 불가였던 cuPF 어댑터를 현재 `cls::Solver` phase API 로 **재작성**
(commit 33fc273) → cuPF 가 custom solver 로 빌드·실행·수렴.

## 0b. ⚠️ B=256 발산은 perturbation 탓
graph_bench 기본값 `scale=1+0.001·b` → B=256 마지막 replica load **+25.5%**, USA 는 해 없어
**모든 백엔드(cuDSS 포함) 발산**. `CUPF_BENCH_SCALE_STEP=0`(identical) 로 재측정 — 전부 identical.

## 1. 수렴 iters (B=1·16·64·256) — fp32 / tf32 / fp16
| case | fp32 | tf32 | fp16 |
|---|---|---|---|
| 3012 | 4·4·4·4 | 4·4·4·4 | 4·4·4·4 |
| 6468 | 4·4·4·4 | 4·4·4·4 | 4·4·4·4 |
| 8387 | 4·4·4·4 | 4·4·4·4 | 4·4·4·4 |
| 25K | 6·6·5·5 | 5·5·5·5 | 5·6·6·6 |
| USA | 8·9·8·8 | 30·30·30·30 ❌ | 12·9·9·9 |

- **3012·6468·8387**: 3정밀도 동일 수렴(4). **25K**: 모두 수렴(5~6).
- **USA**: **tf32 발산**(30 iters, tol 미달) — 진짜 정밀도 floor. fp16 수렴(fp32 보다 iter 더 듦).

## 2. end-to-end ms_per_sys
| case | B | fp32 | tf32 | fp16 | tf32/fp32 | fp16/fp32 |
|---|---|---|---|---|---|---|
| 3012 | 1 | 1.449 | 1.285 | 1.491 | -11.3% | +2.9% |
| 3012 | 16 | 0.140 | 0.135 | 0.145 | -3.6% | +3.5% |
| 3012 | 64 | 0.085 | 0.082 | 0.087 | -2.5% | +2.3% |
| 3012 | 256 | 0.066 | 0.067 | 0.066 | +0.5% | +0.0% |
| 6468 | 1 | 1.955 | 1.863 | 1.864 | -4.7% | -4.7% |
| 6468 | 16 | 0.232 | 0.228 | 0.227 | -1.8% | -2.4% |
| 6468 | 64 | 0.157 | 0.157 | 0.169 | -0.1% | +7.2% |
| 6468 | 256 | 0.146 | 0.148 | 0.145 | +1.2% | -1.0% |
| 8387 | 1 | 2.389 | 2.475 | 2.541 | +3.6% | +6.4% |
| 8387 | 16 | 0.337 | 0.333 | 0.326 | -1.2% | -3.2% |
| 8387 | 64 | 0.237 | 0.244 | 0.236 | +3.2% | -0.2% |
| 8387 | 256 | 0.217 | 0.218 | 0.218 | +0.9% | +0.5% |
| 25K¹ | 1 | 6.088 | 6.313 | 6.498 | +3.7% | +6.7% |
| 25K¹ | 16 | 1.081 | 1.103 | 1.116 | +2.0% | +3.2% |
| 25K¹ | 64 | 0.909 | 1.063 | 1.024 | +17.0% | +12.7% |
| 25K¹ | 256 | 0.866 | 1.016 | 1.016 | +17.3% | +17.3% |
| USA | 1 | 24.490 | 95.819✗ | 36.634 | —발산 | +49.6% |
| USA | 16 | 6.660 | 21.593✗ | 6.421 | —발산 | -3.6% |
| USA | 64 | 5.252 | 18.653✗ | 5.651 | —발산 | +7.6% |
| USA | 256 | 5.005 | 19.062✗ | 5.614 | —발산 | +12.2% |
(`✗` = 미수렴, 시간 무의미. ¹25K = **10-run median** — 단일-run 변동 큼, §5 참조.)

## 3. 결론 (custom 내부, end-to-end)
- **25K**: **정밀도 간 차이 없음(median)** — fp32≈tf32≈fp16, B≥64 는 tf32/fp16 이 오히려 +12~17%(iter 1 더). 기존 단일-run 의 "tf32 승"은 **noise**(§5). **3012·6468·8387**: ±노이즈.
- **USA**: **tf32 발산**(3~4배 느림), fp16 수렴하나 fp32 보다 +8~50%.
- → factor-level TC 우위(note 30)는 **case-dependent**, ill-conditioned 에선 수렴 리스크가 압도.

## 4. 70K (case_ACTIVSg70k, n=70000) — cuDSS vs custom (ms_per_sys, identical)
| backend | B=1 | B=16 | B=64 | B=256 | iters | 수렴 |
|---|---|---|---|---|---|---|
| cuDSS(fp32) | 21.32 | 13.05 | 13.02 | 12.93 | 7·7·7·7 | ✅ |
| custom fp32 | 22.16 | 4.64 | 4.55 | 4.41 | 7·7·8·8 | ✅ |
| custom tf32 | 102.97✗ | 19.54✗ | 16.88✗ | 16.56✗ | 30·30·30·30 | ❌발산 |
| custom fp16 | 30.67 | 5.62 | 5.14 | 6.24 | 10·9·9·11 | ✅ |

**custom fp32 / cuDSS** batch: B=16 0.36× · B=64 0.35× · B=256 0.34× → **custom fp32 가 cuDSS 대비 ~2.8× 빠름**.

- **custom fp32 가 cuDSS 를 batch 에서 압도** (~4.5 vs ~13 ms/sys) — uniform-batch 분할 이득. 수렴 동일(7~8).
- **custom tf32 발산**(30 iters) — 70K 도 tf32 정밀도 부족. **TC 는 큰 케이스에서 또 실패.**
- **70K 실이득은 fp32 의 batch 속도지 TC 가 아님.** (fp16 수렴하나 fp32 보다 느림.)

## 5. ⚠️ 25K 변동성 (run-to-run) & median 갱신

25K 는 수렴 iter 경계(fp32 5↔6, tf32 5–8)에 있어 **run 마다 iter 가 흔들린다** → ms_per_sys 도 같이 흔들림.

- **원인은 parallel ND 가 아니다**: serial ND(`CUPF_CUSTOM_SERIAL_ND=1`)로도 B>1 이 5↔6 흔들림. 비결정성은 **GPU 수치 연산**(multifrontal extend-add 의 `atomicAdd` 누적 순서가 run 마다 달라 factorization 이 비트 단위로 미세히 변함)에 있고, 25K-batch 가 경계라 유독 민감.
- 단일-run 기존값은 10-run median 대비 **−20%~+20%** 빗나감. 옛 "25K tf32 −15~22% 승"은 그 run 이 우연히 fp32=6, tf32=5 iter 였던 **noise**.
- **§1·§2 의 25K 는 10-run median 으로 갱신** (init/solve source `*_med10`, per-op `*_med5`). median 에선 3 정밀도 tied (B≥64 는 fp32 가 오히려 약간 빠름).
- 기존 단일-run(오차 크나 일부 빨랐던) 기록은 보존: `data/cupf_25k_custom_singlerun_noisy_archive_{init_solve,ops_ms}.csv`.
- 다른 케이스는 robust(3012/6468/8387 iter 4 고정, USA 발산 명확, 70K 7~8 안정)라 **단일-run 유지**.

## 6. 재현
```bash
python3 -c "import sys;sys.path.insert(0,'gpu-powerflow/python');from tests.matpower_data import load_case,save_dump_case;save_dump_case(load_case('/datasets/power_system/matpower/case_ACTIVSg70k.m'),'dumps')"
cmake -S gpu-powerflow/cuPF -B build -DCMAKE_BUILD_TYPE=Release -DWITH_CUDA=ON -DCUPF_ENABLE_CUSTOM_SOLVER=ON -DBUILD_EVALUATORS=ON -DCUPF_CUDA_ARCHITECTURES=86 && cmake --build build --target cupf_graph_bench -j
CUPF_BENCH_SCALE_STEP=0 build/tests/cupf_graph_bench <dump> cudss 1,16,64,256 1
for p in fp32 tf32 fp16; do CUPF_BENCH_SCALE_STEP=0 CUPF_CUSTOM_PRECISION=$p build/tests/cupf_graph_bench <dump> custom 1,16,64,256 1; done
```
원자료(통합): init/solve = `data/cupf_mixed_cudss_fp32_mt_auto_scale1_maxiter10_init_solve.csv` (source=custom_*/cudss_*_scale0_maxiter10), per-op(B=1) = `..._ops_ms.csv` (source=*_native_b1).
