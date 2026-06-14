# 260612 목표 (신규) — 멀티배치 factorize 를 compute-bound 로 전환해 1.2–1.4× 가속

> **상태**: planning   **날짜**: 2026-06-12   **GPU**: RTX 3090 (sm_86, 82 SM)
> **한 줄**: B=16/64 fp32 factorize 는 지금 **compute-bound 가 아니라 occupancy(shared-residency 1 block/SM)·일부 memory bound** 다(SM compute throughput 7–31%). mid·big 커널을 재설계해 **연산 유닛을 채워(compute-bound 화)** B=16·64 에서 **1.2–1.4× 가속**을 달성한다. 커널·스케줄링 재설계 허용.

---

## 0. 배경 (왜 이 목표가 성립하나)

[`exp_260612/04`](exp_260612/04-b1-impossibility-analysis.md)·[`05`](exp_260612/05-batch-leverage.md) 에서 확정:
- **B=1 가속은 구조적 불가**(under-fill 벽). 회수 가능한 곳은 **배치 throughput**.
- 배치 fill 로 per-system 이 이미 5–13× 빨라지나, **그 위 B=16/64 자체가 비효율**이라 추가 여지가 크다.

---

## 1. 베이스라인 — 현재 fp32 factorize (per-system, B≥16; ordering 은 배치서 washes out 이라 seed 무관)

| case | B=16 /sys (ms) | B=64 /sys (ms) |
|---|---:|---:|
| 13K (case8387pegase) | 0.0386 | 0.0239 |
| 25K (case_ACTIVSg25k) | 0.1142 | 0.0970 |
| 70K (case_SyntheticUSA) | 0.4383 | 0.4041 |

> 측정: `--precision fp32 --batch {16,64}`, `batch_factor_per_sys_ms`. 정확한 13659/70k 파일 부재 → 대체 case(문서 표기).

## 2. 진단 — 지금은 compute-bound 가 아니다 (ncu, B=64 fp32)

| 지배 커널 | grid | warps_active | **sm__throughput** | DRAM | 병목 |
|---|---:|---:|---:|---:|---|
| `factor_mid_blocked` (25K) | 370 | 18–33% | **7–11%** | 5–8% | whole-front **shared-residency → 1 block/SM**, narrow block warp slot 낭비. latency/occupancy bound |
| `factor_big` (70K) | 2546 | 81%(상위)→18%(심부) | **17–31%** | 34%(상위)→17% | 상위 memory-bound 접근, **심부 레벨 occupancy 저하** |

→ **연산 유닛이 비어 있다**(throughput 7–31%, 로ofline 한참 아래). compute-bound 로 끌어올릴 헤드룸이 곧 가속 여지.

## 3. 목표 · 성공 기준

- **타깃**: **mid·big 커널** (+ 동반 스케줄링).
- **수치 목표**: B=16 **및** B=64 에서 per-system factorize **1.2–1.4×**(case 별).
- **인과 기준**: wall-time 개선을 ncu 로 **메커니즘 입증** — `sm__throughput`(또는 TC pipe active) 상승, occupancy(warps_active)·블록/SM 증가, DRAM 트래픽 감소 중 *무엇으로* 이겼는지 명시.
- **compute-bound 전환 증거**: 개선 후 지배 커널이 SM/TC 파이프 한계(또는 DRAM roofline)에 근접함을 보일 것.

## 4. 허용 범위 (방법론 자유 — 상세 방향은 실험에서 결정)

커널 재설계·스케줄링 재설계 **전면 허용**. 예시(구속 아님): whole-front shared-residency 의 1 block/SM 벽 깨기(부분 staging·register blocking·2-blocks/SM layout), 시스템 차원을 trailing GEMM 의 **batched-GEMM** 으로 접기, TF32/Tensor-Core trailing(이미 B=64 +9–15% 확인 — 출발점이지 상한 아님), mid/big tier 경계 batch-aware 재조정, 심부 레벨 스케줄링. **어느 것이든 측정으로 채택/기각.**

## 5. 측정 프로토콜 (정직성 — notes 54/55)

- **best-vs-best**: A·B 외 변수 고정(같은 batch·precision·case·seed). cap/baseline-inflation 금지.
- **regime 전수**: B∈{16, 64}(필요시 32) 모두. "어느 batch 에서 이기나" 명시.
- **ncu 인과 동반**: §3 인과 기준대로 메커니즘 입증 없이 wall-time 만으로 주장 금지.
- **correctness**: relres + 단위테스트. tf32 채택 시 정확도 손실 동반 보고.

## 6. 산출물

- 실험·발견 → `docs/exp_260612/` (신규 노트 추가).
- 베이스라인·진단 근거 → 본 문서 + [`exp_260612/05`](exp_260612/05-batch-leverage.md).
- 채택분 → `docs/storyline.md` B 축 편입.

## 재현 (베이스라인)
```bash
BIN=build/custom_linear_solver_run; C=/datasets/power_system/nr_linear_systems/case_ACTIVSg25k
$BIN $C --precision fp32 --batch 16 --repeat 15 --warmup 5 --serial-nd --metis-seed 7   # B=16 /sys
$BIN $C --precision fp32 --batch 64 --repeat 15 --warmup 5 --serial-nd --metis-seed 7   # B=64 /sys
# ncu 병목: build-prof(graph off) 로 factor_mid_blocked<float>/factor_big<float> 의 sm__throughput·warps_active 측정
```
