# TF32 TC-eligibility 제한 완화 (mid/big 거의 전부 TC 경로로)

> **상태**: reference   **갱신**: 2026-06-11
> **한 줄**: TF32 trailing 의 적격 gate(`nc≤32 && uc≤256`, mid 의 `fsz>48 && uc≥32 && nc≥8`)를 완화해 거의 모든 mid/big front 를 텐서코어 경로에 올린 실험 — **uc>256 spine front 의 scalar fallback(70K L25 spike)을 제거**, 정확성 유지, factorize ~1.02–1.06×.

이 노트는 [`03-tensor-core-investigation.md`](03-tensor-core-investigation.md) 의 후속이다. TC ceiling 분석은 [`../02-design-analysis/04-gemm-fraction-tc-ceiling.md`](../02-design-analysis/04-gemm-fraction-tc-ceiling.md), occupancy 병목(왜 TC 활용도가 낮은가)은 lab-meeting `small-tier-no-tensorcore.md` §5(d) ncu 진단 참조.

## 동기 — 모든 mid/big 이 TC 를 타지 않는다

TF32 mma trailing 의 적격 조건(완화 전):
- **big** (`factor_big`, big.cuh): `tc = (nc <= 32 && uc <= 256)`. 아니면 `trailing_update_scalar`.
- **mid** (`factor_mid_blocked`, mid.cuh): `fsz > 48 && uc >= 32 && nc >= 8 && nc <= 32 && uc <= 256`. 아니면 scalar(fsz≤48 은 `lu_small_front` fused).

→ 빠지는 front: **uc>256 spine front**(소거트리 root 근처 separator), thin-nc(<8) / 작은 mid(fsz 33–48, uc<32). 특히 70K L25 의 uc=261 front 가 `uc≤256` 에서 탈락해 scalar 로 돌면서 **factorize 의 10%** 를 먹는 spike 였다(EXP_260611 per-level, [lab `etree-characteristics.md`](../20260612_lab_meeting/etree-characteristics.md) §3).

## 방법 — 제한을 매크로로 빼고, 완화값을 **기본으로 승격**

`types.hpp` 에 튜너블 매크로. **2026-06-11 기준 기본값을 완화값으로 승격**(아래 "신" 컬럼) — 검증 후 shipped:

| 매크로 | 구(shipped 전) | **신(기본)** | 의미 |
|---|---:|---:|---|
| `CLS_TC_UC_CAP` | 256 | **512** | TF32 trailing 적격 uc 상한 |
| `CLS_TC_NC_MIN` | 8 | **1** | mid: 최소 nc(K) |
| `CLS_TC_UC_MIN` | 32 | **1** | mid: 최소 uc |
| `CLS_TC_FSZ_MIN` | 48 | **32** | mid: 최소 fsz (mid 는 fsz≥33) |

→ 데이터상 max uc=261·max nc=16 이므로 이 기본값이면 **모든 mid/big 이 TC**(small tier fsz≤32 만 scalar — 원리상 TC 불가). A/B 는 `-DCLS_TC_*` 로 덮어쓰기.

**shared 한계**: whole-front staging `(2·uc_pad·nc_pad + 4·nc_pad)·4 ≤ 99 KiB` → `uc_pad·nc_pad ≤ ~12.6K`. nc 상한 32 에서 uc 는 384 까지가 절대 안전; **UC_CAP=512 는 nc≤24 에서 안전**(전력망 nc≤16 → ~35% 여유). nc 상한(`kTensorCorePivotColumnCap=32`)은 유지 — 더 키우면 shared 초과.

## 버그 — `max_uc` 클램프가 TC cap 과 묶여 있었다

uc cap 만 풀면 70K 가 **`factorization failed`**. compute-sanitizer: `factor_big<float,true>` 의 **shared OOB write**.

원인: `scan_front_range`(`front_range_caps.hpp`)가 TF32 shared 사이징용 `max_uc` 를 **`uc<=256` 으로 클램프**하고 있었다. dispatch 는 `uc_pad_max = round16(max_uc)` 로 shared 를 잡는데, uc=261 front(uc_pad=272)를 TC 로 보내면 staging 이 256 기준 버퍼를 넘는다.

→ 수정: 클램프를 cap 과 **연동**.
```cpp
// front_range_caps.hpp
if (uc > m.max_uc && uc <= CLS_TC_UC_CAP) m.max_uc = uc;   // was: uc <= 256
```
이 둘은 항상 lock-step 이어야 한다(cap 을 풀면 shared 사이징도 따라가야 OOB 없음).

## 결과 (serial-ND 1588, tf32 Ozaki, 클럭 고정, repeat 21)

factorize per-system (ms), baseline = 기본 cap:

| case | B=1 base→tc | 배율 | B=64 base→tc | 배율 |
|---|---|---:|---|---:|
| 3xxx | 0.185 → 0.186 | 0.99× | 0.00826 → 0.00812 | 1.02× |
| 8xxx | 0.312 → 0.312 | 1.00× | 0.0258 → 0.0253 | 1.02× |
| 25K | 0.596 → 0.595 | 1.00× | 0.0852 → 0.0823 | 1.03× |
| **70K** | 1.935 → **1.818** | **1.06×** | 0.366 → 0.356 | 1.03× |
| usa | 1.887 → 1.887 | 1.00× | 0.406 → 0.395 | 1.03× |

- **정확성 유지**: relres 전 케이스 동급(예 70K B1 0.009→0.016, 같은 ~1e-2), compute-sanitizer clean.
- **70K L25 spike 10.0% → 5.5%** (per-level): uc=261 spine front 이 scalar→TC.

## 해석

- 큰 이득은 **uc>256 spine front 를 TC 로** 돌린 것 → 70K B1 1.06%·L25 spike 반감. spine 이 없는 케이스(8387·25K·usa)는 B1 ~1.00×.
- 작은 mid(fsz 33–48, thin-nc)를 TC 로 돌린 건 **거의 중립**(작아서 TC≈scalar overhead).
- **B=64 전 케이스 ~1.02–1.03×**: 더 많은 front 가 TC 라 배치(GPU fill)서 일관된 소폭 이득.
- 상한은 여전히 occupancy: coverage 는 늘었지만 TC 활용도(ncu tensor-op ~2–5%)는 1블록/front under-util 이 좌우 → 더 큰 이득은 occupancy 레버(멀티블록 trailing)와 함께라야.

## 코드 (shipped)

- `src/internal/types.hpp`: `CLS_TC_{UC_CAP,NC_MIN,UC_MIN,FSZ_MIN}` 매크로. **기본값 = 완화값(512/1/1/32)** 로 승격.
- `src/factorize/big.cuh` · `src/factorize/mid.cuh`: gate 가 매크로 참조.
- `src/internal/plan/front_range_caps.hpp`: `max_uc` 클램프를 `CLS_TC_UC_CAP` 과 연동(클램프–cap lock-step 버그 수정).

이제 **기본 빌드에서 모든 mid/big 이 TC**. 구 동작(uc≤256)으로 되돌리려면 `-DCLS_TC_UC_CAP=256 -DCLS_TC_NC_MIN=8 -DCLS_TC_UC_MIN=32 -DCLS_TC_FSZ_MIN=48`.

## 검증 (2026-06-11, 7 case, serial-ND 1588)

전 케이스 정상(factorization 실패 0, compute-sanitizer clean), relres 동급. 70K B1 1.06×·L25 spike 10.0%→5.6%, B=64 전반 1.02–1.03×. small/mid 지배 케이스(3xxx·8xxx·25K·usa)는 B1 ~1.00×(spine 없음).
