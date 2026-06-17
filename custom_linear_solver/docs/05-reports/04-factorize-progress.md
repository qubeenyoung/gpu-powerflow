# Factorize 가속 진행 — B=1 / non-GEMM 레버

> **상태**: reference   **갱신**: 2026-06-10
> **한 줄**: B=1·배치 factorize 가속(채택 변경·tier-split gate·non-GEMM 예산·구조적 한계)의 측정 종합.

원본 진행 보고서(`08`, `09`, `07`, `11`, `12-investigation`, `12-spine`)를 하나로 병합한 reference 문서.
관련: [`01-final-report.md`](01-final-report.md), [`02-comprehensive-sweep.md`](02-comprehensive-sweep.md), [`../03-optimization-notes/01-kernel-engineering.md`](../03-optimization-notes/01-kernel-engineering.md).

기준선: baseline commit `7394c28`, RTX 3090(sm_86, 82 SM), CUDA 12.8/12.x, fp32/fp16, `--single-precision fp64`. skip-trailing·incomplete factorization·solve로의 작업 이동은 채택 수치에 포함하지 않는다.

측정 프로토콜: `factor`는 `batch_factor_per_sys_ms`. wall 분해는 measurement-only 빌드 플래그(`CLS_PROFILE_NO_TRAILING`, `CLS_PROFILE_NO_EXTEND`)로 — factor가 의도적으로 틀려지므로 wall partition 전용. GPU 클럭 미고정이라 A/B는 interleaved median, 3% 미만 delta는 noise로 취급. 동시 실행 금지(병렬 trial은 GPU contention으로 폐기). repeat-21 single-process는 noise 폭이 더 크고, 핵심 점(13K B=256 −30.1%)은 repeat-301 median으로 재확인.

## 1. 목표와 결과 요약

목표: Newton power-flow `factorize`를 (a) B=1 단일 시스템에서, (b) B≥64 포화 배치에서 가속한다. B≥64는 non-GEMM(scatter/front-init/small·mid·big panel/stage·writeback/extend-add) 부분 30% 감소를 직접 타깃으로 삼았다.

| 영역 | 목표 | 실제 달성 |
| --- | --- | --- |
| 배치 non-GEMM 30% | bus ≥ 10K 전반 | 한 점만 달성: `case13659pegase B=256` direct no-GEMM `0.047062 → 0.032911`, **−30.1%**(repeat-301 median, margin 극소) |
| 25K+ 배치 non-GEMM | 30% | 미달. 재생성 `case_ACTIVSg25k B=256` direct no-GEMM **−12.4%**, 70K/USA B=64/256 **−13.9%..−18.4%** |
| B=1 20% target | bus 전반 best-mode 20% | **미달**(아래 §5) |
| B=1 10% target | <10K·>10K 각 1개 best-mode 10% | 달성: `case3012wp` 11.8%, `case13659pegase` 13.8%(vs best `min(fp32,fp16)`) |

요지: B≥64에서 측정 wall은 이미 non-GEMM이 지배(`case13659pegase B=256` 95.1%, `case_ACTIVSg25k B=64/256` ~92%, 최대 케이스도 83–87%). 따라서 tensor core는 이 타깃의 주 레버가 아니다(§4). B=1은 etree 깊이 직렬성이라는 구조적 한계에 묶인다(§5).

## 2. 채택된 변경

`08`(B=1 exact-path)과 `12-spine`(spine fusion)에서 default로 채택된 변경. 모두 exact numeric factorization 범위.

| 변경 | 적용 범위 | 효과 / non-GEMM 성격 |
| --- | --- | --- |
| `scatter_values_unique` | 전 B, `a_pos` unique 시 | analyze가 destination 중복 없음을 증명하면 scatter atomicAdd → store. 중복 시 atomic fallback. NR 케이스 전반에서 불필요 atomic 제거 |
| B=1 full factor graph replay | B=1 | scatter/init + factor level을 한 graph에 캡처, launch 오버헤드 감소 |
| `factor_mid_single` | B=1 mid tier | small-front 처리에서 batch indexing/modulo 제거 |
| FP16 B=1 scalar mid (`n < 24000`) | B=1 fp16 mid | mid front에서 FP16 WMMA staging 회피, 큰 fp16 개선 + 잔차 향상 |
| `factor_spine_chain` (spine-chain fusion) | B=1 single-panel top spine, 저-B 게이트 | cnt=1 spine 사슬을 1 커널로 fuse. **8387 B=1 −3%(wall)/−7.5%(GPU), B=4 −10.7%**. 25k는 spine front가 shared 96KB 초과로 미발동(무회귀) |
| narrow cap18 policy | B=1, `5000 ≤ n < 8000` | 정확 factorization, amalgamation width만 변경. `case3012wp`를 strict best-mode 10% 위로 |
| plain extend gate | B=1 fp32 `n ≥ 80000` main-stream `level_size==1` | 동시 부모 갱신 sibling이 없을 때 atomic 회피 |

`08`의 best-mode 검증(3-process median, candidate=fp16):

| case | bus class | baseline best | candidate | reduction vs best | relres |
| --- | --- | ---: | ---: | ---: | ---: |
| `case3012wp` | <10K | 0.222217 | 0.196111 | 11.8% | 1.96e-04 |
| `case8387pegase` | <10K | 0.333045 | 0.314470 | 5.6% | 2.17e-05 |
| `case9241pegase` | <10K | 0.344407 | 0.316234 | 8.2% | 2.62e-06 |
| `case13659pegase` | >10K | 0.421932 | 0.363716 | 13.8% | 1.16e-04 |

mode-specific fp16 비교(`08`)는 더 큼: `case3012wp` 30.7%, `case8387pegase` 28.5%, `case9241pegase` 31.5%, `case13659pegase` 37.3% — 단 mid trailing을 FP16 WMMA에서 scalar로 바꾼 것을 포함하므로 순수 panel/sync/memory 결과로 계산하지 않는다.

`09`의 fresh repeat-21 B=1 fp32 full-factor(band-split 전):

| case | baseline | candidate | delta |
| --- | ---: | ---: | ---: |
| `case_ACTIVSg10k` | 0.357962 | 0.323337 | −9.7% |
| `case13659pegase` | 0.409158 | 0.344106 | −15.9% |
| `case_ACTIVSg25k` | 0.844436 | 0.787558 | −6.7% |
| `case_ACTIVSg70k` | 2.949510 | 2.722230 | −7.7% |
| `case_SyntheticUSA` | 2.795450 | 2.349520 | −16.0% |

USA no-trailing fraction(61.8%)을 proxy로 쓰면 B=1 fp32 −9.7..−16.0%는 estimated non-GEMM **−15.7..−25.9%**에 해당해 30% 미만이다.

## 3. Tier-split dispatch (occupancy gate)

`11`의 최종 형태. storyline 의 A1 tier routing 운영 형태에 해당한다.

문제(`doc 10` 데이터, case8387 n=14908, fronts=7431, levels=28): factorize dispatch가 tier(small/mid/big 커널)를 **level의 max_fsz**로 정한다. 한 level에 큰 front 하나만 있어도 그 level의 작은 front 전부가 큰 커널로 승격(case8387에서 small→mid 승격 = **1760 fronts = 23.68%**). mid 커널은 front당 256-스레드 블록이라 fsz≤8 front엔 과한 스레드.

설계: tier를 level-max가 아니라 **front별**로. 같은 level front는 etree상 독립(부모는 상위 level)이므로 level을 tier-동질 sub-range로 쪼개 순차 launch해도 정확. analyze 1회 비용으로 `h_plcols_tier`(level-major, tier-contiguous) + per-(level,tier) CSR 경계 생성. case 상수 없음 — tier 경계는 커널 능력(SMALL_THRESH=32 / MID_THRESH=128).

**B 의존성과 게이트**: 무조건 split하면 B에 따라 정반대(8387 ungated B=1 **+82%**, B=256 −18%). 저 B는 latency 영역이라 mid 커널의 많은 스레드가 빈 GPU에서 유리, 고 B는 warp-packing throughput이 유리. 그래서 occupancy 게이트:

```
warp_fill = num_SMs × (max_threads_per_SM / 32)   // 순수 HW값, 3090 = 3936
split  ⇔  level이 mixed  ∧  small_cnt × B ≥ warp_fill
else   →  whole-level dispatch (= 승격, pre-split 동작)
```

B=1은 게이트가 split을 끄므로 BASE와 동일 커널(regression 0). 결과(interleaved median, factor delta):

| case | B=1 | B=16 | B=64 | B=256 |
|------|----:|-----:|-----:|------:|
| case8387 factor | +1.9%* | −1.4% | **−11.2%** | **−13.0%** |
| ACTIVSg25k factor | −4.4%* | +0.1% | **−3.6%** | **−3.9%** |

`*` B=1은 split 비활성 → 순수 noise. solve dispatch도 동일 tier 배열·동일 occupancy 게이트 적용(case8387 solve B=64 −8.9%, B=256 −15.8%). factor+solve 합산 per-system은 **case8387 B=256 기준 ~−14%**. multistream도 (subtree,level) cell 단위로 동일 분해+게이트(case8387 B=256 factor+solve −4.3%, 25k −3.5%). 정확성 불변(relres 8387 ~2e-5, 25k ~1.3e-4, fp64 3e-14).

기각: finer 5-class 분해(small을 1–8/9–16/17–32로 추가 분할) — 고 B에서 +2% 미미, 25k B=1 +6.5% 악화. 3-tier split 유지.

## 4. non-GEMM 예산 분석

다중 배치 full-factor best-mode delta(`09`, repeat-21, band-split 전)는 broad 3–6%에 한 점만 −12%:

| case | B=64 best | B=256 best |
| --- | ---: | ---: |
| `case_ACTIVSg10k` | −5.0% | −3.5% |
| `case13659pegase` | −5.3% | −5.1% |
| `case_ACTIVSg25k` | −4.9% | −5.1% |
| `case_ACTIVSg70k` | −5.8% | −5.9% |
| `case_SyntheticUSA` | −6.2% | −3.5% |

front-size tier 분포(`f2`=front arena `fsz²` share, `f3`≈dense-factor `fsz³` proxy share)가 케이스별 레버 차이를 설명한다. 13K는 `fsz>128` front이 0개(small/mid 지배), 25K는 mid 지배(`33..128` front이 panel 1.6%지만 f3 87.8%), 70K는 56개 big front이 f3의 51.2%를 차지:

| case | tier | count | f2 share | f3 share |
| --- | --- | ---: | ---: | ---: |
| `case13659pegase` | `fsz ≤ 32` | 12266 | 68.3% | 34.6% |
| `case13659pegase` | `33..48` | 93 | 18.7% | 31.4% |
| `case_ACTIVSg25k` | `fsz ≤ 32` | 22381 | 45.5% | 12.3% |
| `case_ACTIVSg25k` | `65..96` | 92 | 23.6% | 43.2% |
| `case_ACTIVSg70k` | `fsz ≤ 32` | 62712 | 41.0% | 6.6% |
| `case_ACTIVSg70k` | `fsz > 128` | 56 | 20.4% | 51.2% |

따라서 25K는 mid-front 경로, 70K/USA는 big/front-memory 경로 — 13K-tuned small/mid 단일 최적화로는 25K+ 전체를 못 움직인다. setup-time fsz band-split(`16000 ≤ n < 40000`은 B≥128, `n ≥ 40000`은 B>1에만 발동, B=1은 canonical order 유지)이 이 분리를 구현해 25K/70K/USA 다중 배치를 추가로 끌어내렸다(§1 수치).

`09`의 measurement-only `CLS_PROFILE_NO_TRAILING` / `CLS_PROFILE_NO_EXTEND` 빌드(factor를 의도적으로 틀리게 만들어 wall 분해만). candidate fp32 direct non-GEMM(baseline vs candidate no-trailing):

| case | B | direct non-GEMM delta | candidate non-GEMM fraction |
| --- | ---: | ---: | ---: |
| `case13659pegase` | 256 | −26.7% (band-split 후 **−30.1%**) | 95.1% |
| `case_ACTIVSg25k` | 64 | −2.2% | 92.1% |
| `case_ACTIVSg25k` | 256 | +6.8% (band-split 후 −12.4%) | 92.0% |
| `case_ACTIVSg70k` | 64 | −5.8% (band-split 후 −18.4%) | 82.9% |
| `case_ACTIVSg70k` | 256 | −4.2% (band-split 후 −16.3%) | 84.0% |
| `case_SyntheticUSA` | 64/256 | band-split 후 −13.9%/−16.3% | 82.8%/86.5% |

graph-node 프로파일(no-trailing): 잔여 wall은 `factor_mid<float>` + `factor_mid<float>`가 지배.

- 25K B=256(5 calls 합): `factor_mid` 53.48 ms > `factor_mid` 39.23 ms > `scatter_values_unique` 12.69 ms.
- 70K B=256(5 calls 합): `factor_mid` 148.42 ms > `factor_mid` 100.53 ms > `factor_big` 68.60 ms > front arena `cudaMemsetAsync` 45.46 ms(~7.91 GB/call) > scatter 36.57 ms.

front-memset은 실재 병목이지만 sparse 대체 불가: sorted-`a_pos` 보완 범위 zeroing은 25K에서 dense memset 바이트의 최대 12.8%, 70K 11.5%만 절감하고 fill run이 26–30 slot로 짧아 최적화 memset보다 느리다. front-memory 개선은 dense arena 미물질화(fusion/layout)가 필요.

**extend-add 예산**(`CLS_PROFILE_NO_EXTEND`, B=256 no-trailing 대비):

| case | extend-add 예산 |
| --- | ---: |
| `case13659pegase` | 12.9% |
| `case_ACTIVSg25k` | 20.4% |
| `case_ACTIVSg70k` | 18.7% |

tier별: 13K는 거의 mid-tier(no small extend −10.1%), 25K는 small+mid 분할(−11.9%/−9.8%), 70K는 small+big(−12.3%/−6.2%). tier 수치는 graph node 지속시간·stream overlap이 측정마다 달라 비-가산적이며, tier별 parent-update 재설계의 상한으로만 쓴다.

contiguous-extend audit: contiguous extend slot은 ~29–36%(13K small ~26%/mid ~36–42%, 25K mid ~49%, 70K big ~68–71%). exact contiguous `asm_local` fast path는 budget을 일부 커버하나 per-front branch/metadata가 full factor를 회귀시켜 revert(13K no-trailing 0.033284 > 0.032911).

**parent-update는 contended atomic이 아니다**: exact destination-collision audit(`--analyze-info`, level scope)에서 colliding writes는 **4–11%**뿐(13K 10.3–10.4%, 25K 4.5–5.0%, 70K 4.0–4.4%)인데 no-extend는 13–20% 예산을 보인다. 즉 비용은 고-multiplicity atomic이 아니라 **mostly-uncontended write traffic + update 경로 자체**. plain-safe 분리(level scope 69–79% 커버)는 budget × safe_share ≈ 9–16% no-trailing에 그쳐 30%를 닫지 못한다.

## 5. B=1 구조적 한계

`12-investigation`과 `07`이 같은 결론. B=1 병목은 **깊은 narrow level이 1블록·~0.3% GPU util·순차 pivot chain에 묶인 구조적 latency**다.

프로파일(8387 B=1, graph-node trace, busy ~310µs ≈ wall ~340µs):

| kernel | us/iter | 비중 |
|--------|--------:|----:|
| factor_mid (22 launch) | 239 | 77% |
| factor_mid (6) | 66 | 21% |
| scatter_values | 4.5 | 1.5% |

`factor_mid` 22 launch가 front 708개 level이나 1개 level이나 ~8–15µs로 비슷. ncu(`factor_mid` fp32, 256 threads):

| level 유형 | grid | SM throughput | barrier stall |
|-----------|------|--------------:|--------------:|
| deep narrow (front 1개) | (1,1)×256 | **0.3%** | 1.5 |
| shallow (front 709/400) | (709/400,1)×256 | 9–18% | 4.9–5.8 |

deep narrow level은 1블록이 빈 GPU(81 SM idle)에서 실행 → ~16개 narrow level × ~10µs ≈ factor의 절반 이상. shallow level은 `lu_small_front`의 pivot당 `__syncthreads`에 barrier-bound. block-scale 시도: ungated(무조건 scale)은 high B 개선(8387 B=256 −5%, 25k −3%)이나 **B=1 +21% 악화** — 빈 GPU의 1블록은 스레드가 많을수록 intra-block 병렬 이득이라 serialization 패널티가 barrier 절감보다 큼. gated는 B=1 회귀는 막지만 결과가 noise 수준(8387 B=16 +6.5%)이라 "특정 케이스 미세 최적화 금지" 원칙에 위배 → 되돌림. 결론: **intra-kernel small/mid 트윅으로는 B=1이 개선되지 않는다(측정 확인)**.

실효 레버는 (a) 구조적 fusion(narrow level을 device-sync로 묶기, 복잡·리스크) 또는 (b) **batching으로 B↑**(doc 14 B≈64 포화 — GPU를 채우면 narrow level도 B개 front로 병렬화). spine fusion(§2)은 phase-1로 채택됐으나 spine ~98µs 대부분이 launch가 아닌 single-block compute라 이득이 작다(fusion은 inter-launch gap + re-staging만 제거). phase-2 후보(본 변경 범위 밖): narrow non-spine level fusion(cross-front 의존으로 더 복잡), 대형 그리드용 global-resident spine 커널(25k처럼 spine front가 shared 초과하는 경우).

**20% target 미달(`07`)**: incomplete-factor+GMRES, deferred-spine은 factor 작업을 solve로 옮기므로 기각됨. honest exact `b1_fast` 최종 baseline(`CLS_B1_DEFER_SPINE_LEVELS=0`, repeat-25)은 11개 케이스 모두 target factor ms 미달:

| case | target ms | exact b1_fast ms | pass |
| --- | ---: | ---: | --- |
| case1197 | 0.055666 | 0.058200 | no |
| case3012wp | 0.197102 | 0.264798 | no |
| case8387pegase | 0.293573 | 0.346842 | no |
| case13659pegase | 0.326235 | 0.455176 | no |
| case_ACTIVSg25k | 0.581672 | 0.714604 | no |
| case_ACTIVSg70k | 1.666872 | 1.914160 | no |
| case_SyntheticUSA | 1.856112 | 2.181070 | no |

## 6. 폐기 실험 목록

`09` 다중 배치 시도(타깃 `case13659pegase B=256 fp32`, 모두 30% 미달 후 revert):

| 실험 | 결과 | 판정 |
| --- | ---: | --- |
| B>1 full graph 캡처(scatter/init+level) | full 0.037697, no-trailing 0.037464 | revert — B>1 launch 비-병목 |
| 2D small-front batch grid(`% level_size` 제거) | full 0.041045 | revert — 1D warp 패킹이 우수 |
| tiled `scatter_values_unique` (B tile 4/2) | no-trailing 0.036604 / 0.047776 | revert — q-contiguous 패턴 깨짐 |
| scatter block 256→512 threads | no-trailing 0.035031 | revert |
| `__launch_bounds__(256,2)` (small / mid) | no-trailing 0.036172 / 0.035972 | revert |
| size-scaled mid block(64/128/256 by max_fsz) | no-trailing 0.034575 | revert |
| `__ldg` metadata in `factor_mid{,_single}` | full 0.053341 | revert |
| child-count 기반 plain extend | full 0.037989 | revert |
| `--panel-cap` override sweep(cap 8–20) | cap8 0.036663(repeat-101) | revert |
| fill-in-only front zero(`front_zero_pos`) | no-trailing 0.043412 | revert — memset보다 느림 |
| 25K finer fsz-band / small-packing(4·16 warp) | coarse 대비 악화/noise | revert |
| contiguous `asm_local` extend fast path | full 0.093278(25K) > 0.092086 | revert |
| `CLS_EXTEND_PLAIN_SAFE`(plan-wide unique 비-atomic extend) | 13K +5.0%, 70K +3.6% no-trailing | default OFF — atomic 교체는 25K+ 레버 아님 |
| 70K big-tier block 1024→512 | 0.313462 > 0.284919 | revert |

`08`/`07` B=1 기각: `fsz 33..48` split, FP32 mid dynamic block 64/128/256, mid tier 4/16 warp, reciprocal-multiply pivot, `ROWFUSED_NC_MAX=8`, `factor_mid __launch_bounds__(256,2)`, full tail-chain fusion(48 panels, 과직렬화), `CLS_B1_SMALL`/`TIER_SPLIT`/`SMALL_COOP`/`SUBTREE_COOP`/`MIXED_SMALL`/`WARP_USOLVE`/`INIT_MAP`, TF32·TF32 WMMA, no-atomic small extend(잔차 깨짐). incomplete-factor+GMRES·deferred-spine은 solve로 작업 이동 → hard reset.

### 문헌 인용

- Rennich, Stosic, Davis, "Accelerating Sparse Cholesky Factorization on GPUs" (IA3 2014): 작거나 불규칙한 dense 연산은 GPU 가속이 어렵다 — batching·concurrent kernel·subtree GPU 잔류로 launch/PCIe 절감.
- Boukaram et al., "Batched sparse direct solver ... in SuperLU_DIST" (IJHPCA 2024): elimination-tree level별 batched dense + batched Scatter 커널 — level batching·scatter layout이 1급 설계 대상.
- Abdelfattah, Tomov, Dongarra, "Progressive Optimization of Batched LU on GPUs" (HPEC 2019): 매우 작은 LU는 level-3 BLAS만으로 안 되고 size-aware kernel fusion 필요 — small-front TC/cuBLAS 패킹 음성 결과와 일치.
- Abdelfattah et al., "Batched One-Sided Factorizations of Small Matrices" (JOCS 2018): size ≤32는 vectorization·메모리 traffic·register blocking·concurrency가 핵심, ordinary GEMM 아님.
- Volkov, Demmel, "LU, QR and Cholesky ... Vector Capabilities of GPUs" (UCB/EECS-2008-49): dense factorization은 blocking 후에야 높은 GEMM-rate, panel 작업은 별도 scheduling 필요.
- NVIDIA CUDA C++ Programming Guide(Warp Matrix Functions), cuBLAS `cublasComputeType_t`(`CUBLAS_COMPUTE_32F_FAST_TF32` 등), A100/Ampere whitepaper: TC는 dense tile MMA/trailing GEMM에 매핑되며 scalar panel·scatter·memset·extend-add에는 직접 적용 안 됨. 측정상 `case8387 B=256` trailing fraction `f=0.21` → 2× trailing 가속해도 wall 상한 `1.12×`. 따라서 30% non-GEMM 타깃은 level batching·scatter/front layout·sync·memory·extend 변경이 필요하지 TC 미세 튜닝이 아니다.

## 7. 남은 증명 / 방향

- **near-miss를 넘기기**: `case13659pegase B=256`이 −26.7%(band-split 후 −30.1%, margin 극소). 13K는 big tier가 없으므로 small/mid proof point로만 취급하고 25K+/70K 일반 증명으로 쓰지 않는다.
- **multi-batch small/mid**: B=64/256은 launch 레버 소진. partial front init·front arena layout 압축·zero/scatter를 first-use stage-in과 fusion하되 q-contiguous warp 패턴은 보존(tiled scatter 실패가 한정).
- **multi-batch big tier**: 70K/USA를 guardrail로. `fsz>128` front이 f3 51.2% 소유. 단순 resident CTA 증가(512-thread)는 잘못된 개입 — 알고리즘 재구조화/메모리 볼륨 감소 필요.
- **B=1**: 구조적 fusion(narrow level device-sync/megakernel) 또는 batching이 유일한 실효 레버.
- 남은 증명: 타깃 10K+(이상적으로 25K+ multi-batch) 케이스에서 direct no-trailing ≤ baseline의 70%, 그리고 성공 변경 후 serial full/no-trailing 표 재측정.
