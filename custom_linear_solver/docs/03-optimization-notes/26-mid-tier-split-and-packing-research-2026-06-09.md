# Mid tier 분할 + front-per-block packing 연구

**작성일**: 2026-06-09
**범위**: mid tier dispatch bucket 세분화(반영 완료) + factor_mid latency 진단 + packing 연구 방향(미구현, 추후 대상).
**상태**: mid-split은 commit `073ab65` 반영. packing은 이론·근거 정리 단계(구현 보류).

---

## 1. 반영: mid tier를 mid-low / mid-high bucket으로 분할 (commit 073ab65)

`front_bucket()` (src/plan/solver_constants.hpp)가 analyze-time tier ordering을 세분화한다.
mid tier(`kSmallFrontMax < fsz <= kMidFrontMax`)를 `kMidSplitFrontMax=64`에서 둘로 쪼개
heterogeneous mid level이 작은 mid front(fsz≤64) 다수를 **tighter `fsz_cap`**으로 dispatch한다.
shared front-staging `Fs ~ fsz_cap²`이므로 cap이 작아지면 점유율이 오른다.

- 커널 SELECTION은 `classify_front_tier`(small/mid/big) 그대로 — 두 mid bucket 모두 `factor_mid`를
  자기 `fsz_cap`으로 실행. dispatch/kernel signature 무변경, bucket 수만 `kNumTiers` 3→4.
- `CLS_NO_MID_SPLIT`로 3-bucket A/B 복원.

### 측정 (ACTIVSg25k fp32 B=64)
- mid front 이질성: cap≥100 level에 작은 mid front가 몰려(25K 106개, USA 438개), level-max cap이
  작은 mid front의 Fs를 38–43% over-allocate. (front CSV `--dump-fronts` 분석.)
- split이 mid-low launch 점유율을 **28% → 73%** (shared 29KB→10KB)로 올림.
- **그러나 factor_mid 총 wall time 불변**: baseline 83.0ms vs split 82.5ms (nsys, no-graph 빌드).
- end-to-end A/B: fp64 B=1 underfill만 win(USA −8.6%, 25K −3.9%), B=64는 노이즈(±3%).

→ 점유율은 올랐으나 속도는 그대로. **유지하는 이유**: 손해 없음 + shared headroom 확보가
packing(아래)의 전제. 점유율↑ 자체가 목적이 아니라 packing을 위한 발판.

---

## 2. 진단: factor_mid는 latency-bound (점유율 무관)

factor_mid는 B=64 factor의 **~37%** (지배 커널, nsys). ncu (25K fp32 B=64):

| 지표 | 값 | 해석 |
|---|---|---|
| SM throughput | 22% | compute-bound 아님 |
| DRAM throughput | 16% | bandwidth-bound 아님 |
| sm__warps_active | 73% (mid-low) | 점유율 충분 |
| **stall long_scoreboard** | **8.4 /issue** | **global load 지연 (지배)** |
| stall barrier (__syncthreads) | 4.3 | per-pivot sync |
| stall wait (FMA dep) | 2.6 | |
| global **load** sectors/req | 1.83 | **합착됨** |
| global store sectors/req | 6.84 | 비합착(extend_add scatter) |
| L2 sector hit rate | 42% | load 58%가 cold→DRAM |

핵심: **점유율 73%인데도 long_scoreboard 8.4 잔존.** warp 병렬이 충분한데 메모리 지연이
안 숨겨진다 → "독립 warp 추가"(occupancy = naive packing의 본질) 레버는 **소진**.

### 보조 실험 (음성)
- filled scalar-mid block 256 → front-sized(64/128): factor_mid 82.5→80.2ms (≈3%, 노이즈).
  thread 수만 줄이는 건 무효. lever는 thread 수가 아니라 front 간 overlap.

---

## 3. 이론: phase-aligned staging stall

`stage_in_async`(src/factorize/phases.cuh)는 cp.async로 front를 shared `Fs`에 올린 뒤
`__pipeline_wait_prior(0)`로 **front 전체 staging 지연을 동기 barrier로 노출**한다. load는
합착됐지만 cold(L2 42%)라 DRAM 지연이 크다.

모든 block이 커널 진입 직후 stage_in에서 **동시에 같은 위상**으로 메모리를 기다린다. 그래서
occupancy(다른 block의 연산)가 이 구간을 못 채운다 — 다들 staging 중이라 채울 연산이 없다.
이것이 "warp를 더 줘도(occupancy/naive packing) 안 숨겨지는" 이유다.

> 일반화: staged front 커널(mid + big)과 batched-dense 직접해법 전반에 적용되는
> *Little's law / 위상정렬 잠복 은닉* 문제. 단일 케이스 튜닝이 아니라 구조적 성질.

---

## 4. 추후 연구 대상: pipelined front-per-block packing

packing의 진짜 레버는 occupancy가 아니라 **software-pipelined staging**이다 — 한 block이
front A를 factorize하는 동안 front B를 prefetch하여 위상 정렬을 깨고 staging 지연을 연산과
overlap한다.

### Falsifiable / 판별 예측 (실험을 숫자놀이가 아니게 만드는 핵심)
- **pipelined packing → long_scoreboard 감소 + factor_mid 단축.**
- **naive packing(F front 한꺼번에 stage→한꺼번에 factor) → 효과 없음** (위상 정렬 유지,
  occupancy와 동일 → 이미 음성으로 측정됨).

오직 pipelined만 위상을 깨므로 이 예측이 occupancy와 packing을 구분한다. 구현 전 검증할
근거: long_scoreboard를 source 귀속(lineinfo + ncu source counters)해 `stage_in`의 wait가
지배함을 확정. stage_in이 지배이면 pipelined-packing 프로토타입이 이론의 직접 테스트.

### 설계 스케치
- mid-low front F개를 한 block에 묶음, 각 front를 warp-group(sub-block)이 처리.
- per-front sub-barrier(cooperative-group/named barrier)로 block-wide `__syncthreads` 회피.
- staging 파이프라인: stage(f+1) ∥ factorize(f). shared 예산 F×Fs — **mid-split이 확보한
  headroom 사용**.
- 동일 원리를 big tier에도 확장 가능(global-resident이라 staging 패턴은 다름, 별도 검토).

### 보조 관찰
- store 비합착(6.84 sec/req)은 extend_add의 parent scatter. long_scoreboard(load)와 별개지만
  store 트래픽 1.7× 낭비 — extend-add 레이아웃/fusion은 독립적인 후속 레버.

---

## 5. 결론
- mid-split은 점유율↑ + shared headroom 확보로 반영(중립~소폭, packing 전제).
- factor_mid 병목은 점유율이 아니라 **phase-aligned staging 잠복 지연** — 이론·근거 확립.
- 다음 실험은 **pipelined vs naive packing** 판별로, 위 이론을 직접 검정(일반화 가능).
