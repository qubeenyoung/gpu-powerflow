# cuPF 성능 병목 분석 보고서

- 작성: perf/cupf-bottleneck-analysis 브랜치
- 환경: NVIDIA RTX 3090, CUDA 12.8, Release 빌드, `ENABLE_TIMING=ON`
- 대상: `BackendKind::CUDA`, cuDSS 선형 솔버
- 케이스: pandapower `case118 / case1354pegase / case2869pegase / case9241pegase`
  (평균 차수 nnz/row ≈ 3.5–4.1, 전형적인 전력망 희소도)
- 재현: `perf/dump_cases.py`(케이스 생성), `cupf_cpp_evaluate`(단일), `cupf_batch_bench`(배치),
  `perf/aggregate.py`(집계)

> 결론 요약: **선형계(factorize+solve)가 전체의 64–88%로 압도적 1순위 병목**이며 n에
> 따라 비중이 커진다. 배치로 가면 ibus(SpMV)도 절대시간은 커지지만 비중은 ~6%로 안정적이고,
> 오히려 **host↔device 전송(upload/download)이 큰 배치에서 ~22%까지** 증가한다. 그 외
> 여러 구조적·구현적 비효율(배치 공통 수렴, 매 iteration 동기화, SpMV lane 낭비, FP64
> jacobian의 ibus 중복 계산)이 확인됐다.

---

## 1. 단일 케이스 stage 분해 (실험 A)

`NR.solve.total` 대비 stage 비중 (5회 평균, FP64/cuDSS):

| stage | case118 | case1354 | case2869 | case9241 |
|---|---|---|---|---|
| factorize | 37.3% | 51.7% | 52.5% | **63.2%** |
| solve | 30.0% | 31.3% | 30.8% | 24.3% |
| **linear_solve 합** | **67.3%** | **83.0%** | **83.3%** | **87.5%** |
| jacobian | 7.8% | 3.7% | 3.2% | 2.4% |
| mismatch_norm | 6.7% | 3.4% | 3.3% | 2.1% |
| ibus | 3.5% | 2.2% | 2.5% | 2.2% |
| voltage_update | 4.1% | 2.0% | 1.7% | 0.8% |
| mismatch | 3.3% | 1.5% | 1.2% | 0.6% |
| upload / download | 4.3 / 1.2% | 2.7 / 0.6% | 3.4 / 0.7% | 3.7 / 0.4% |

**관찰**
- 선형계 비중이 n과 함께 67%→88%로 증가. factorize 단독이 최대 stage.
- ibus/mismatch/jacobian/voltage는 단일 케이스에서 각각 작다(대부분 고정 launch 오버헤드 성격).
- Mixed는 FP64 대비 factorize가 싸져 total이 약 30% 빠르지만 **비중 구조는 동일**(선형계 81%).

---

## 2. 배치 스케일링 (실험 B, case2869, Mixed)

| B | total(us) | per-case(us) | factorize | solve | ibus | upload+download |
|---|---|---|---|---|---|---|
| 1 | 2,316 | 2,316 | 47% | 30% | 3.7% | 7% |
| 4 | 3,843 | 961 | 51% | 26% | 4.9% | 11% |
| 16 | 10,845 | 678 | 53% | 24% | 6.0% | 13% |
| 64 | 42,720 | 668 | 49% | 21% | 5.7% | 21% |
| 256 | 168,460 | 658 | 49% | 21% | 5.7% | **23%** |

작은 케이스(case118, Mixed)는 per-case가 B=1→256에서 **784→29us (27×)** 개선되나 B≈256에서 평탄화.

**관찰**
- **배치는 "고정 오버헤드 amortization"이 본질**이다. 작은 시스템(case118)은 단일 solve가 GPU를
  못 채워 배치 이득이 크지만, 큰 시스템(case2869)은 B≈16에서 이미 per-case가 평탄화(2316→658us, 3.5×).
- **ibus(SpMV)는 절대시간이 B에 비례해 커지지만 비중은 ~6%로 안정적**. 사용자 가설("멀티배치면
  ibus가 커진다")은 절대량 측면에서 맞으나, 여전히 선형계(~70%)가 지배적.
- **upload+download가 큰 배치에서 ~23%까지 증가** — 새로 부상하는 2순위 병목.
- factorize+solve는 모든 B에서 ~70% 유지. cuDSS uniform-batch가 per-case 비용을 거의 그대로
  쌓는다(배치 내 추가 가속 거의 없음). 동일 sparsity 패턴 B개를 푸는데도 economy of scale 부재.

---

## 3. 병목 및 비효율 지점 (우선순위순)

### P1. 선형계 — 지배적 병목 (64–88%)
- factorize가 매 NR iteration 수행된다. cuDSS 심볼릭 분석은 `initialize()`에서 1회만 하고
  (좋음), 이후 REFACTORIZATION 경로를 탄다(`cuda_cudss.cpp::factorize`). 구조 자체는 합리적.
- **개선 여지**
  - **자코비안 재사용/지연 갱신(dishonest/chord Newton)**: 수렴 후반부엔 J가 거의 안 변하므로
    factorize를 매 iteration 하지 않고 N iteration마다 또는 norm 감소율 기준으로만 재인수분해.
    이미 `IterationContext.jacobian_age` 필드가 있으나 현재 미사용 — 이 정책을 붙이면 factorize
    호출 횟수를 직접 줄일 수 있다. (가장 효과 큰 단일 개선 후보)
  - **iterative refinement + 저정밀 factorization**: Mixed가 이미 FP32 factorize로 이득을 보지만,
    FP32-factor + FP64-residual 정제를 명시적으로 도입하면 정확도 유지하며 더 공격적으로 저정밀화 가능.
  - cuDSS 배치에서 economy of scale이 없으므로, 동일 패턴 다수 시스템은 **단일 심볼릭 + 배치
    numeric**가 제대로 작동하는지 cuDSS 옵션(UBATCH) 재검토 가치.

### P2. host↔device 전송 — 큰 배치의 2순위 (최대 ~23%)
- `CudaFp32Storage::upload` / `CudaMixedStorage::upload`는 **호스트에서 원소별 루프로
  double→float 캐스팅한 std::vector를 만든 뒤 H2D 복사**한다. download도 대칭으로 float→double를
  호스트 루프로 변환. 비용이 O(B·n_bus)로 배치에 비례해 커진다.
- **개선**: (a) 캐스팅을 디바이스 커널로 이전(원본 FP64를 그대로 H2D 후 GPU에서 변환 —
  이미 `prepare_rhs` 커널이 유사 패턴), (b) 핀드(pinned) 호스트 메모리 + 비동기 복사로 오버랩,
  (c) 입력이 batch에 걸쳐 공유되는 부분(예: 동일 Ybus 패턴)은 재업로드 회피.

### P3. 배치 공통 수렴 — 구조적 낭비 (이종 배치에서)
- `newton_solver.cpp::run_iteration_stages`의 루프는 **단일 `ctx.converged`로 break**한다
  (line 431). 배치의 mismatch_norm은 **배치 전체의 최댓값 norm**으로 수렴을 판정하므로
  (`cuda_mismatch.cu` CudaMismatchNormOp), 일부 케이스가 먼저 수렴해도 **가장 느린 케이스가
  끝날 때까지 모든 케이스가 factorize/solve/ibus를 계속 수행**한다.
- 동질 배치(본 실험처럼 거의 동일한 부하)에선 안 보이지만, 실제 이종 배치(서로 다른 계통/부하
  시나리오)에선 상당한 낭비.
- **개선**: per-case 수렴 마스크를 도입해 수렴한 케이스의 stage 연산을 스킵(또는 압축). 최소한
  voltage_update/jacobian/solve에 active-mask를 전달.

### P4. 매 iteration 블로킹 동기화 — 지연 → **보류 (구현 안 함)**
- `CudaMismatchNormOp::run`이 매 iteration `d_normF`를 호스트로 복사(`copyTo(&ctx.normF,1)`)해
  수렴을 검사한다 → iteration마다 D2H 블로킹 동기화.
- **재분석 결과 — 이 sync는 제거하면 안 된다.** `run_iteration_stages`(newton_solver.cpp:425–444)
  구조상 norm 검사(`if (ctx.converged) break;`, line 431)는 **지배적 비용인 factorize 직전**에 있다.
  즉 이 sync는 "비싼 factorize를 하기 전에 수렴 여부를 확인해 불필요한 마지막 factorize를 건너뛰는"
  early-exit를 보장한다. sync를 지연/제거(파이프라이닝)하면 수렴 판정이 update 이후로 밀려 **매 solve마다
  factorize를 1회 더** 하게 되고, factorize가 ~65%이므로 sync 지연(수십 µs)을 아끼려다 factorize 1회
  (수 ms)를 더 내는 **순손해**가 된다. sync가 의미 있는 소형 문제는 절대 시간이 이미 작다.
- **결론**: 현재 mid-iteration sync는 합리적 설계. launch 오버헤드를 줄이려면 early-exit를 유지한 채
  **CUDA graph로 반복을 캡처**하는 별도 작업이 필요(범위 큼). 단순 sync 완화는 권장하지 않음.

### P4b. CUDA Graph 반복 캡처 (옵션) — 타당성 검증됨
- **실험(perf/cudss_capture_probe.cu)**: cuDSS의 `REFACTORIZATION + SOLVE`가 **stream capture
  가능**함을 확인(노드 19개 캡처, 그래프 재생 결과 정확). 즉 cuDSS를 그래프 밖으로 뺄 필요 없이
  반복 본체를 통째로 캡처 가능.
- **설계(옵션, 기본 off, 빌드 플래그 `CUPF_ENABLE_CUDA_GRAPH` + 비-timing 빌드 전제)**: early-exit를
  유지하기 위해 그래프를 둘로 나눈다 —
  - `graph_pre` = ibus → mismatch → norm-reduce (수렴 판정 입력 생성), 이후 host에서 norm sync+check.
  - `graph_post` = jacobian → prepare_rhs → factorize/refactorize → solve → voltage_update (cuDSS 포함).
  - factorize 위상(첫 iter=FACTORIZATION vs 이후=REFACTORIZATION)이 다르므로 iter0은 eager 실행 후
    iter1에서 `graph_post`를 lazy 캡처하여 iter2+ 재생.
- **제약/wrinkle**: (1) 캡처 중 `cudaDeviceSynchronize` 금지 → timing 빌드(ENABLE_TIMING)와 norm의
  `copyTo` sync는 캡처 구간 밖이어야 함(그래서 pre/post 분리, norm sync는 eager). (2) **non-default
  stream 필수**(legacy default stream은 캡처 불가) → 기능 on시 전용 스트림 생성. (3) Jacobian 패턴/배치
  shape 변경 시 그래프 재캡처.
- **기대효과**: 작은 커널들(ibus/jacobian/voltage 등)의 **launch 오버헤드 제거** → 소형 케이스·소배치에
  유효. 대형/대배치는 cuDSS 연산이 지배적이라(그래프는 cuDSS 연산 자체를 가속하지 않음) 효과는 작음.

### P5. ibus SpMV — lane 활용도 낭비 (구현)
- `compute_ibus.cu` 커널은 **row당 32-lane warp**로 누적하는데, 전력망 평균 차수는 **~3.5–4
  nnz/row**다. 즉 누적 루프에서 **32 레인 중 약 4개만 일하고 ~88%가 유휴**.
- 현재 ibus 비중은 ~6%라 즉효는 작지만, 멀티배치에서 절대시간이 커지는 stage이고 비효율이 명확.
- **개선**: (a) row당 레인 수를 평균 차수에 맞춰 축소(예: 4 또는 8)하고 한 warp가 여러 row 처리,
  또는 (b) merge-based / CSR-stream SpMV로 부하 균형, (c) Jacobian fill과 동일한 edge-parallel
  (nnz 1개당 1 thread + atomic) 패턴으로 통일.

### P6. FP64 jacobian의 ibus 중복 계산 (구현)
- `fill_jacobian_gpu.cu`의 `CudaJacobianOp<double>::run`은 `use_cached_ibus=false`로 호출한다
  (line ~226). 그래서 **대각 항에서 bus 전류주입(Ibus)을 커널 안에서 행 전체를 다시 돌며 재계산**한다
  — 직전 ibus stage가 이미 `d_Ibus_re/im`을 계산했는데도. (FP32/Mixed 경로는 `true`로 캐시 사용.)
- jacobian이 ~2–3%라 영향은 작지만 명백한 중복. **개선**: FP64 경로도 캐시된 Ibus를 전달.

### P7. mismatch_norm 리덕션 (구현, 단일 케이스)
- FP64 단일 경로의 norm 리덕션은 **단일 블록(grid=1)**으로 dimF(최대 ~16k) 전체를 strided로 읽어
  리덕션한다(`reduce_mismatch_norm.cu` 호출부, FP64는 grid=1). 단일 케이스에서 ~2–7%를 차지.
- **개선**: 다중 블록 + 2단계 리덕션, 또는 mismatch 계산과 융합(fused norm).

---

## 4. 권고 (효과/난이도/상태)

| 항목 | 기대효과 | 난이도 | 상태 |
|---|---|---|---|
| **P1 자코비안 지연 갱신(jacobian_age 활용)** | 큰 (factorize 호출 수↓) | 중 | 보류 (알고리즘 수정, 사용자 요청으로 연기) |
| P2 전송 디바이스화 | 큰(대배치) | 중 | **적용 완료** (commit 3b955c0) |
| P3 per-case 수렴 마스크 | 큰(이종 배치) | 중상 | 보류 |
| P4 동기화 완화 / CUDA graph | 소 (대케이스), 중(소) | 중상 | **보류 (구현 안 함)** — §3 P4 참조: sync가 early-exit를 보장 |
| P5 ibus SpMV scalar화 | 중(대배치) | 중 | **적용 완료** (commit 90de5fe) |
| P6 FP64 jacobian 캐시 ibus | 소 | 하 | **적용 완료** (commit 7856063) |
| P7 norm 멀티블록 리덕션 | 소 | 하 | 미적용 (선택) |

---

## 5. 구현 결과 (P2 / P5 / P6 적용)

모두 매 변경마다 **WITH_CUDA=ON 빌드 + 수렴 정확도 + stage timing** 검증. 동작/수치 동일.

### 항목별 측정

| 항목 | 측정 | before | after | 개선 |
|---|---|---|---|---|
| P6 | jacobian stage (case9241, FP64) | 213us | 123us | 1.7× |
| P5 | ibus stage (case2869, Mixed, B=64) | 2,440us | 407us | 6.0× |
| P5 | ibus stage (case2869, Mixed, B=256) | 9,632us | 1,348us | 7.1× |
| P5 | ibus stage (case9241, FP64 단일) | 192us | 144us | 1.3× |
| P2 | upload (case2869, Mixed, B=256) | 20,728us | 2,521us | 8.2× |
| P2 | download (case2869, Mixed, B=256) | 20,094us | 2,445us | 8.2× |

### 누적 (case2869, Mixed, solve_total)

| B | baseline | P2+P5+P6 | 개선 |
|---|---|---|---|
| 16 | 10,845us | 9,283us | 1.17× |
| 64 | 42,720us | 32,655us | 1.31× |
| 256 | 168,460us | 121,257us | **1.39×** |

- 대배치일수록 효과 큼: 제거 대상(전송 호스트 루프, ibus lane 낭비)이 배치에 비례해 커졌기 때문.
- **단일 케이스(case9241, FP64)는 거의 불변**(~8.8ms): 선형계(factorize ~65%)가 지배적이라 P5/P6의
  비-선형계 절감이 묻힌다. → 단일·대형 케이스의 추가 개선은 P1(선형계) 영역.

### P4 미적용 사유
§3 P4 참조 — 매 iteration norm sync는 factorize 직전 early-exit를 보장하므로, 제거 시 매 solve마다
factorize 1회를 추가로 내는 순손해. CUDA graph 기반 접근(early-exit 유지 + launch 오버헤드 제거)은
범위가 커 별도 과제로 분류.

---

## 부록: 재현 방법

```bash
# 케이스 생성
python3 perf/dump_cases.py perf/cases
# 타이밍 빌드
cmake -S . -B build/perf-gpu -DWITH_CUDA=ON -DBUILD_EVALUATORS=ON \
  -DENABLE_TIMING=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/usr/local/lib/python3.10/dist-packages/torch/share/cmake
cmake --build build/perf-gpu -j --target cupf_cpp_evaluate cupf_batch_bench
# 단일 케이스 stage 분해
./build/perf-gpu/cupf_cpp_evaluate --case-root perf/cases --output-dir perf/out_fp64 \
  --backend cuda --compute fp64 --warmup 2 --repeats 5
python3 perf/aggregate.py perf/out_fp64/timing.csv
# 배치 스케일링
./build/perf-gpu/cupf_batch_bench perf/cases/case2869pegase mixed 1,4,16,64,256 5
```
