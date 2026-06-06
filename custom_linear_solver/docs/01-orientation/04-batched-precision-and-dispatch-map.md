# Batched 솔버의 정밀도 모드 + factor / solve 커널 dispatch 맵

*`custom_linear_solver` 의 `--batch B` (uniform-batch) 경로 한정. 정밀도 5개 (FP64 / FP32 / Mixed / TC / TC32) 가 front-size tier 와 곱해져 어떤 커널로 dispatch 되는지, 그리고 각 env / CMake lever 가 어디서 어떤 분기를 바꾸는지 한 곳에 정리한다.*

> 본 문서는 *재구성 (refactoring) 의 베이스라인* — 코드 위 어떤 결정이 어디서 일어나는지 단일 표로 합치고, 중복 / 사문화된 분기를 식별한다.

---

## 0. 오너 결정 / 코멘트 (Owner Notes)

> 이 섹션은 사용자가 직접 채우는 공간입니다. AI 는 사용자의 명시적 지시가 있을 때만 수정합니다.


### 0.1 사용자 지시 — 2026-06-05

> 아래 항목은 `custom_linear_solver` 재구성 작업의 사용자 지시다. 본 문서의 기존 dispatch map / refactor 후보 설명과 충돌하는 경우, 이 지시를 우선한다.

1. 삭제 대상 코드는 바로 흩어두지 말고 `deprecated/` 폴더 아래에 정리한다.
2. 파일명과 커널 이름은 과감히 변경한다. 커널 이름은 구현 세부사항보다 알고리즘이 잘 드러나도록 추상화해서 짓고, 지나치게 긴 이름은 피한다. 특히 Tensor Core 를 쓰지 않는 커널 이름에 `tc` 가 남아 있으면 제거한다.
3. `src/batched/factorize` 는 일반 batched 경로이고, `src/factorize` 는 `B=1` 전용 경로다. `src/solve` 도 같은 맥락에서 중복 구조를 점검한다. 일관성과 직관성을 우선하며, 필요하면 `src/factorize` / `src/solve` 를 삭제하고 `src/batched` 아래 코드를 상위 구조로 승격한다.
4. `src/` 전체 구조는 과감히 정리한다. 디렉터리 이름과 소스 배치는 읽는 사람이 바로 책임 영역을 이해할 수 있게 직관적으로 만든다.
5. 지원 정밀도는 `fp64`, `fp32`, `tc` 세 가지로 확정한다. `mixed` 는 제거하고, `tc32` 라는 별도 이름도 제거한다. 새 `tc` 는 기존 `tc32` 경로를 의미한다.
6. 정밀도 선택용 환경변수는 제거한다. 정밀도는 실행 시 환경변수가 아니라 설정 / 구성 값으로 확정한다.
7. factorize dispatch 정책은 코드 구조와 성능을 함께 고려해 정리한다. 불필요한 죽은 분기와 모호한 threshold 는 제거하거나 명확한 정책으로 고정한다.
8. `CLS_PROFILE_NO_TRAILING` 및 `cls_profile_no_trailing` 계열은 실험 코드이므로 삭제한다. 같은 맥락의 실험용 코드, 측정용 clone, 임시 우회 경로도 전부 제거 대상으로 본다.
9. `selinv`, `amalg` 관련 경로와 토글은 삭제한다.
10. 주석은 표준 스타일로 정리한다. 실험 흔적, 로그성 메모, 임시 설명은 제거하고 알고리즘 설명 중심으로 다시 작성한다. 복잡한 흐름은 ASCII 시각화로 설명하고, 커널 내부도 실행 흐름을 따라갈 수 있게 주석을 둔다.
11. 가독성을 우선한다. 변수명 / 함수명 / 커널명은 사람이 읽을 수 있게 일관적으로 정리하고, 논리 블록별로 개행과 배치를 정돈한다.

### 0.2 § 9 의 refactor 후보별 결정

§ 9 의 표에 별도 "결정 / 코멘트" 칸이 있으니 거기에 인라인으로 기록하세요.

### 0.3 미해결 질문 / 추가 조사가 필요한 항목

- [ ] _____
- [ ] _____
- [ ] _____

### 0.4 외부 참조 / 디스커션 링크

| 항목 | 링크 | 메모 |
|---|---|---|
| | | |

---

## 1. 정밀도 모드 (BatchPrecision)

`src/batched/multifrontal_batched.hpp:30`:

```cpp
enum class BatchPrecision { FP64, FP32, Mixed, TC, TC32 };
```

| Mode | front 저장 | working LU | trailing GEMM | extend-add | 정확도 (NR Jacobian) |
|---|---|---|---|---|---|
| **FP64** | FP64 (`d_frontB`) | FP64 | scalar FP64 | FP64 atomic | ~1e-13 |
| **FP32** | FP32 (`d_frontBf`) | FP32 | scalar FP32 (mid_tc32\<false\>) **또는 staged-scalar FP32 (mid_tiled)** ★ | FP32 atomic | ~1e-4 ~ 1e-2 |
| **Mixed** | FP64 master + FP32 working | FP32 | scalar FP32 | FP64 atomic (FP32→FP64 cast) | ~1e-5 ~ 1e-3 |
| **TC** | FP64 master + FP32 working | FP32 | **FP16 WMMA (HMMA)** + FP32 accum | FP64 atomic | ~1e-3 ~ 1e-1 |
| **TC32** | FP32 (NO master) | FP32 | **FP16 WMMA (HMMA)** + FP32 accum | FP32 atomic | ~FP32 (trailing 부분만 FP16 rounding) |

★ FP32 는 *두 단계의 mid-tier 커널* 을 가짐 — 자세한 dispatch 는 § 3.

### 1.1 메모리 영향 (front arena, FP64 USA 기준)

| Mode | per-batch front | B=256 (USA) 총량 | 비교 |
|---|---:|---:|---|
| FP64 | 70.4 MB | 18.0 GB | baseline |
| FP32 | 35.2 MB | 9.0 GB | 1/2 |
| Mixed | 70.4 + 35.2 = 105.6 MB | 27.0 GB | **1.5×** (FP32 working 추가, RTX 3090 24GB 초과 위험) |
| TC | 105.6 MB | 27.0 GB | **1.5×** |
| TC32 | 35.2 MB | 9.0 GB | 1/2 (Mixed/TC 와 다름) |

→ **Mixed/TC 는 *FP64 보다 메모리 더 큼*** — FP64 master + FP32 working 둘 다 유지. 큰 case 에선 메모리 압박이 *FP64 보다 심해질 수 있다*.
→ TC32 는 FP32 와 동일 점유.

## 2. 정밀도 선택 — env var 우선순위

`tests/run_custom_solver.cu:355-362`:

```cpp
// 기본값: n<24000 → Mixed, n>=24000 → FP64
BatchPrecision prec = (n < 24000) ? BatchPrecision::Mixed : BatchPrecision::FP64;
if (std::getenv("MF_NO_MIXED")) prec = BatchPrecision::FP64;
if (std::getenv("MF_FP32"))     prec = BatchPrecision::FP32;
if (std::getenv("MF_MIXED"))    prec = BatchPrecision::Mixed;
if (std::getenv("MF_TC"))       prec = BatchPrecision::TC;
if (std::getenv("MF_TC32"))     prec = BatchPrecision::TC32;
```

순차 if → **나중에 set 된 게 이김**. 충돌하는 두 변수 (e.g., `MF_FP32=1 MF_TC32=1`) → TC32.

### 2.1 default 의 함정

- case8387 (n=14908) — default = **Mixed**. 즉 측정 시 `MF_*` 안 주면 Mixed 가 됨.
- USA (n=156255) — default = **FP64**. 같은 이유.

→ 벤치마크에서 "default" 가 case 마다 달라 비교가 비일관 — 본 dispatcher 의 *first cleanup 후보*.

## 3. Factor dispatch — `issue_factor_level_range` (`multifrontal_batched.cu:68-267`)

per-level 로 max_fsz / max_uc 를 host 에서 계산해 (a) front-size tier, (b) 정밀도 mode 에 따라 분기.

### 3.1 결정 트리

```
issue_factor_level_range(b, e)
├─ small_ok = (FP32 || TC32 || FP64)     # = pure_fp32 || FP64
├─ MID_THRESH = (TC32 ? 128 : 159)
├─ s_skip_trailing = (CLS_PROFILE_NO_TRAILING != 0)
├─ s_use_tiled_fp32 = (CLS_NO_TILED_TRAILING == 0)  # default ON
│
├─ if small_ok:
│   ├─ ① max_fsz ≤ 32  →  mf_factor_small_warp_b<FT>
│   │      • 1 warp / (front, batch), SMALL_WARPS=8 per block
│   │      • shared-staged front, __syncwarp only (no block sync)
│   │      • FT = double (FP64) or float (FP32/TC32 — Mixed/TC 안 옴: small_ok 가 그들 false)
│   │
│   ├─ ② pure_fp32 && max_fsz ≤ MID_THRESH:    # mid-tier float
│   │   ├─ TC32 →  mf_factor_mid_tc32_b<true>     # WMMA trailing
│   │   ├─ FP32 && s_use_tiled_fp32 && max_fsz ≥ 48 && shared ≤ 96KB
│   │   │       →  tc::mf_factor_mid_tiled_b      # Σ.14 staged-shared scalar trailing
│   │   └─ FP32 else
│   │           →  mf_factor_mid_tc32_b<false>    # 같은 kernel symbol, scalar trailing
│   │
│   └─ ③ pure_fp32 && max_fsz > MID_THRESH:     # big float (top levels)
│       ├─ TC32 →  mf_factor_extend_tc32_b       # WMMA trailing (1024 threads)
│       └─ FP32 →  mf_factor_extend_level_b<float>  (1024 threads)
│
└─ else (FP64, Mixed, TC) — switch (prec):
    ├─ FP64  →  mf_factor_extend_level_b<double>           (128 threads)
    ├─ Mixed →  mf_factor_extend_mixed_b                    (128 threads, FP64 master cast)
    ├─ TC    →  mf_factor_extend_mixed_tc_b                 (128 threads, WMMA)
    └─ TC32  →  mf_factor_extend_tc32_b                     (※ unreachable - § 3.3)
```

### 3.2 CLS_PROFILE_NO_TRAILING

`CLS_PROFILE_NO_TRAILING=1` 이면 ①②③ 각각의 *FP32 path* 가 `*_NT_b` clone 으로 swap (trailing 만 skip). 측정용 — 정확도 깨짐. `T_full - T_NT = trailing wall time`.

지원되는 NT clones:
- `mf_factor_small_warp_NT_b`
- `tc::mf_factor_mid_tiled_NT_b`
- `mf_factor_mid_tc32_NT_b`
- `mf_factor_extend_level_NT_b`

(TC32 / Mixed / TC 의 NT 는 없음 — FP32 path 만 instrument.)

### 3.3 dispatcher 의 모호한 곳 (refactor target)

**(a) TC32 의 mid_tc32 vs extend_tc32 경계가 `MID_THRESH=128` 이지만, FP32 의 mid 경계는 `159`** — 같은 kernel symbol 인데 mode 마다 다른 임계값. 의도된 건지 잔재인지 불명.

**(b) FP32 의 mid path 에 *두 커널* 이 존재** (mid_tiled vs mid_tc32\<false\>). dispatcher 가 *shared 96KB 제약* 으로 둘 사이 fallthrough. 두 커널의 *상위 phase (panel LU + U panel solve) 는 동일*, *trailing 만 다름* — staged vs raw scalar. trailing 만 분기시키면 코드 중복 제거 가능.

**(c) `else` switch 안의 TC32 case 는 unreachable** (line 240, `case BatchPrecision::TC32:`) — `is_fp32_front(TC32)=true` 라 위 `if small_ok` 분기에서 처리되고 이리 절대 안 옴. **죽은 코드**.

**(d) `small_ok` 정의가 *"FP64 OR pure_fp32"*** — 즉 Mixed / TC 는 leaf-level small fronts 도 generic extend kernel 로 처리 (block=128, 1 block/(front,batch)). small fronts 가 매우 많은 power-grid 에선 손해 — Mixed/TC 가 *작은 front 에서 FP64 / FP32 보다 느린* 한 원인.

**(e) `s_use_tiled_fp32` static lambda 가 *프로세스 시작 시 1회만 평가*** — 단일 프로세스 안에서 env 를 바꿔도 반영 안 됨. 측정 자동화에서 유의.

### 3.4 selinv 마지막

dispatcher 후 `issue_factor_levels` (line 319-329) 에서 `st.selinv == true` 이면 `mf_invert_pivot_b<FT>` 1번 launch — 각 (front, batch) 의 nc×nc pivot block 을 FP64 inverse (FT 상관없이 코드 내부 FP64) 로 계산해 덮어쓰기.

**Default OFF** (post-Σ.9, line 439-440):
```cpp
st.selinv = (std::getenv("CLS_USE_SELINV") != nullptr) &&
            (std::getenv("MF_NO_SELINV") == nullptr);
```

→ `CLS_USE_SELINV=1` 켜야 invert_pivot 호출됨.
→ `MF_NO_SELINV=1` 은 *CLS_USE_SELINV 가 켜져도* OFF 로 강제 (kill-switch).

## 4. Solve dispatch — `issue_solve_levels` (`multifrontal_batched.cu:332-425`)

per-level 로 max_fsz 만 보고 fwd / bwd 분기.

```
issue_solve_levels()
├─ pure_fp32 = FT가 float (FP32 또는 TC32)
├─ FT = float (pure_fp32) or double (FP64/Mixed/TC)
├─ sel = st.selinv ? 1 : 0     # kernel 내부에서 sel 분기
│
├─ fwd loop (leaf → root):
│   ├─ max_fsz ≤ 32  →  mf_fwd_small_warp_b<FT>           # warp-packed
│   └─ else          →  mf_fwd_level_b<FT>                # block per (front, batch)
│        block size = 64 (mf≤64), 128 (mf≤128), 256 (else)
│
└─ bwd loop (root → leaf):
    ├─ max_fsz ≤ 32  →  mf_bwd_small_warp_b<FT>
    └─ else          →  mf_bwd_level_b<FT>
         dynamic shared = max_cb * sizeof(FT)  (CB row cache for bwd GEMV)
```

→ solve 는 *정밀도와 무관하게 같은 dispatcher* — FP64/Mixed/TC 모두 *FP64 front 를 읽고 FP64 y* 로 동작 (Mixed/TC 의 *master* front 가 solve input).
→ FP32 / TC32 는 *FP32 front + FP32 y*.

### 4.1 solve dispatch 의 모호한 곳

**(a) selinv off / on 에서 kernel 의 *경로* 가 같지만 내부 분기 (`if (selinv)`) 가 있음** — kernel 내부 branch + 컴파일러가 못 최적화. 두 별도 kernel 로 나누면 selinv off 의 코드 hot path 가 더 깨끗.

**(b) bwd 의 dynamic shared `max_cb` 가 *전 level 통일 sized* 가 아니라 *per-level*** — 다른 level 의 kernel 이 다른 shared 점유. 캡처된 graph 에서 launch 마다 attribute 가 다름. (현재 동작은 OK, refactor 시 의식)

## 5. 환경변수 / CMake lever 전체 인벤토리

### 5.1 정밀도 선택 (`tests/run_custom_solver.cu`)

| env | 동작 | 우선순위 | default |
|---|---|---:|---|
| `MF_NO_MIXED` | → FP64 | 1 | unset |
| `MF_FP32` | → FP32 | 2 | unset |
| `MF_MIXED` | → Mixed | 3 | unset |
| `MF_TC` | → TC | 4 | unset |
| `MF_TC32` | → TC32 | 5 | unset |
| (none) | n<24000 → Mixed, else FP64 | — | — |

### 5.2 batched factor / solve toggles (`src/batched/multifrontal_batched.cu`)

| env | 동작 | default | 영향받는 kernel/path |
|---|---|---|---|
| `CLS_USE_SELINV` | selinv ON (invert_pivot 호출) | **OFF** | mf_invert_pivot_b 추가 launch + solve kernel 내부 `sel` 분기 |
| `MF_NO_SELINV` | selinv 강제 OFF (CLS_USE_SELINV 가 켜져도) | unset | (kill-switch) |
| `CLS_NO_MULTISTREAM` | multi-stream subtree dispatch OFF | **ON** | issue_factor_levels: subtree streams 사용 / 미사용 |
| `CLS_NO_TILED_TRAILING` | Σ.14 mid_tiled 사용 안 함 (mid_tc32\<false\> 만) | **mid_tiled ON** | FP32 mid-tier path 에서 두 kernel 중 선택 |
| `CLS_PROFILE_NO_TRAILING` | FP32 factor kernels 의 NT clone 사용 | **OFF** | small_warp / mid_tiled / mid_tc32 / extend_level → `*_NT_b` |
| `MF_EXT_CAPTURE` | runner 가 external capture 모드로 (cuPF 통합 검증용) | unset | tests/run_custom_solver.cu 의 측정 흐름만 |

### 5.3 single-system path 및 symbolic toggle (`src/factorize/`, `src/solver.cpp`, `src/symbolic/`)

본 문서는 batched path 가 중심이지만 영향 있는 것:

| env | 동작 | default |
|---|---|---|
| `CLS_USE_AMAL` | 심볼릭 amalgamation 사용 | unset (off) |
| `CLS_CAP` | panel cap 강제 override | (분석에서 자동 결정) |
| `CLS_AMAL_CAP` | amalgamation cap | unset |
| `CLS_AMAL_MIN_DEPTH` | amalgamation 최소 깊이 | unset |
| `CLS_AMAL_INFO` | dump amalgamation 결정 | OFF |
| `CLS_USE_SMALL_WARP` | single-system small kernel 사용 | OFF |
| `CLS_NO_SHARED_FACTOR` | single-system shared-resident factor OFF | OFF |
| `CLS_WARP_DBG` | warp-level debug print | OFF |
| `CLS_DUMP` | 분석 후 etree / front size 분포 dump | OFF |
| `CLS_PANEL_DUMP` | per-panel (fsz, nc) dump (GEMM 분석용) | OFF |
| `CLS_TREE_INFO` | etree info dump | OFF |
| `CLS_KERNEL_TIME` | 커널 시간 자가 측정 | OFF |

### 5.4 TC dedicated path (`src/tc/multifrontal_tc.cu`) — `MF_TC` 와 다른 경로

TC dedicated path 가 별도 존재 (batched 와 무관, `solver.tc_setup(B)` 로 진입). Power-grid 측정에선 사용 안 함. 관련 env:

| env | 동작 |
|---|---|
| `CLS_USE_CUBLAS` | trailing 을 cuBLAS gemmStridedBatched 로 |
| `CLS_CUBLAS_TF32` | TF32 모드 |
| `CLS_CUBLAS_MIN_FSZ` | cuBLAS 사용 threshold |
| `CLS_USE_REGBLOCK` | trailing_update_regblock (FP32) |
| `CLS_USE_REGBLOCK_H16` | trailing_update_regblock_h16 (FP16 input) |
| `CLS_USE_SPINE` | spine kernel 사용 |
| `CLS_USE_MULTISTREAM` | multi-stream (positive flag, TC path 의 `CLS_NO_MULTISTREAM` 와 반대) |
| `CLS_TC_SETUP_DBG` | TC setup 디버그 |
| `CLS_USE_PIVOTING` | within-pivot-block partial pivoting |
| `CLS_BYPASS_GRAPH` | TC path 에서 graph 우회 |

### 5.5 CMake options (`CMakeLists.txt`)

| option | default | 효과 |
|---|---|---|
| `CLS_BUILD_CUDA_OPS` | ON | CUDA kernel 빌드 |
| **`CLS_INTERNAL_GRAPH`** | **ON** | batched factor / solve 를 *내부 CUDA graph 로 capture* 후 replay. OFF → 외부 stream 에 raw kernel issue (cuPF 의 outer capture 호환). nsys / ncu 측정 시 OFF 권장. |
| `CLS_BUILD_SCRIPTS` | ON | tests/run_custom_solver.cu 빌드 |
| `CLS_BUILD_CUDSS_SCRIPT` | OFF | cuDSS 비교 스크립트 |
| `CLS_ENABLE_NVTX` | OFF | NVTX range 삽입 |

→ 측정 환경에 *graph ON 빌드 (/tmp/clsb)* 와 *graph OFF 빌드 (/tmp/clsb_nograph)* 두 개를 두는 게 표준.

### 5.6 lever 그룹 — refactor 시 정리 후보

1. **명명 비일관** : `MF_*` (정밀도 + selinv kill-switch) vs `CLS_*` (구조 toggle + selinv ON-switch) — 통일 필요.
2. **이중 게이트** : selinv 가 `CLS_USE_SELINV` AND NOT `MF_NO_SELINV` 의 AND — 단일 toggle 로 정리 가능.
3. **죽은 분기**: dispatcher 의 `case TC32:` (§ 3.3-c). 제거 대상.
4. **사문화된 lever**: `MF_NO_MIXED` 가 의미상 *FP64 강제* — `MF_FP64=1` 로 renaming 이 명확.
5. **그룹 분리**: 측정/디버그용 (`CLS_*_DBG`, `CLS_*_DUMP`, `CLS_PROFILE_*`) vs 동작 변경 (`CLS_NO_*`, `CLS_USE_*`) — 두 그룹의 prefix 분리.

## 6. 정밀도별 kernel 표 (한 눈에)

| Front size tier | FP64 | FP32 | Mixed | TC | TC32 |
|---|---|---|---|---|---|
| fsz ≤ 32 (small) | small_warp\<double\> | small_warp\<float\> | (extend_mixed) ✗* | (extend_mixed_tc) ✗ | small_warp\<float\> |
| 32 < fsz ≤ 48 (small-mid) | extend_level\<double\> | mid_tc32\<false\> | extend_mixed | extend_mixed_tc | mid_tc32\<true\> |
| 48 < fsz ≤ MID_THRESH (mid) | extend_level\<double\> | **mid_tiled** (default) or mid_tc32\<false\> (shared > 96KB) | extend_mixed | extend_mixed_tc | mid_tc32\<true\> |
| fsz > MID_THRESH (big) | extend_level\<double\> | extend_level\<float\> | extend_mixed | extend_mixed_tc | extend_tc32 |

\* Mixed/TC 가 small-warp 경로 안 받음 → 작은 front 가 많은 power-grid 에서 손해 (§ 3.3-d).

`MID_THRESH = 128` (TC32), `159` (FP32) — § 3.3-a 불일치.

## 7. solve 의 kernel 표

| Front size tier | FP64/Mixed/TC | FP32/TC32 |
|---|---|---|
| fsz ≤ 32 | fwd/bwd_small_warp\<double\> | fwd/bwd_small_warp\<float\> |
| fsz > 32 | fwd/bwd_level\<double\> | fwd/bwd_level\<float\> |

→ Mixed/TC 의 solve 는 **FP64 path 와 동일** (Mixed/TC 의 *master* front 를 읽음). TC32 만 FP32 solve.

## 8. selinv 의 효과 (FP32 측정)

| Case | B | selinv ON factor | selinv OFF factor (default) | OFF speedup | solve ON | solve OFF |
|---|---:|---:|---:|---:|---:|---:|
| case8387 | 256 | 64.6 μs | **25.5 μs** | **2.53×** | 17.5 μs | 19.4 μs (+11%) |
| USA | 256 | 919.4 μs | **471.7 μs** | **1.95×** | 169.6 μs | 187.6 μs (+11%) |

→ **NR loop / 1-factor-1-solve workload 에선 selinv OFF 가 압도적 이득** (factor 절감 >> solve 손해). default OFF 가 올바른 선택.
→ 다중 RHS / 동일 행렬의 여러 solve 가 있으면 selinv ON 이득 (factor 1회 비용 vs solve 다수 가속).

## 9. 다음 단계 — refactoring 후보 (proposal, 코드 변경 전)

> "결정" 칸: `accept` / `defer` / `reject` / `?` 중 하나. "코멘트" 는 자유 메모. 빈 칸은 미정.

| # | 후보 | 위치 | 위험 / 영향 | **결정** | **코멘트** |
|---:|---|---|---|---|---|
| R1 | 죽은 분기 제거 : `case BatchPrecision::TC32:` | `multifrontal_batched.cu:240` | 무영향 (unreachable). 단순 cleanup. | | |
| R2 | `MID_THRESH` 통일 (128 vs 159) | `multifrontal_batched.cu:78` | TC32 의 WMMA tile fit 과 FP32 의 shared 한도가 본질적으로 다른지 확인 필요 | | |
| R3 | `mid_tiled` / `mid_tc32<false>` 코드 통합 (trailing만 분기) | `tc/trailing_tiled.cuh:192` + `tc/factor_tc.cuh:139` | 두 kernel symbol 유지 vs 단일 kernel + functor — graph capture 영향 확인 | | |
| R4 | small_warp 의 Mixed / TC 지원 | `multifrontal_batched.cu:79` (`small_ok`) | Mixed/TC 가 *FP64 master + FP32 working* 둘 다 staging 필요 — shared 사이즈 |  | |
| R5 | env var 명명 통일 (`MF_NO_MIXED` → `MF_FP64`, selinv 단일 게이트) | `tests/run_custom_solver.cu:358`, `multifrontal_batched.cu:439` | 외부 (cuPF, benchmark scripts) breaking change | | |
| R6 | dispatcher 의 host loop (max_fsz/uc 계산) 을 level metadata 에 흡수 | `multifrontal_batched.cu:91-97`, plan 측 변경 | analyze 측 데이터 늘어남, plan 직렬화 호환 | | |
| R7 | NT clone 을 FP64 / Mixed / TC 에도 추가 | 새 헤더 4개 | 측정 전용 코드 양 증가, 유지보수 비용 | | |
| R8 | doc 09, 10 측정 결과 정정 (selinv OFF default 기준 재측정) | docs only | 시간 비용 (USA 풀 측정 ~1h) | | |
| | | | | | |

### 9.1 새 refactor 후보 추가

| # | 후보 | 위치 | 위험 / 영향 | **결정** | **코멘트** |
|---:|---|---|---|---|---|
| R9 | | | | | |
| R10 | | | | | |

## 10. 관련 문서

- `01-orientation/01-api-and-build-design.md` — cuDSS-like phase API 및 빌드 옵션 개요
- `02-design-analysis/01-why-custom-fast-on-power-grid.md` — D1-D8 설계 분해
- `02-design-analysis/04-multifrontal-layout-and-level-batching-vs-strumpack.md` — level-batching 구조
- `03-optimization-notes/01-fp32-batched-kernel-optimization.md` — FP32 path 의 small/mid/big 라우팅
- `04-benchmarks-profiling/09-batched-membound-case8387-usa-b4-b64-b256.md` — FP64 bound 측정 (selinv ON 구 바이너리 기준 — **본 문서의 default 정정으로 결론 수정 필요**)
- `04-benchmarks-profiling/10-batched-membound-case8387-usa-b4-b64-b256-fp32.md` — FP32 bound 측정 (같은 정정 필요)
