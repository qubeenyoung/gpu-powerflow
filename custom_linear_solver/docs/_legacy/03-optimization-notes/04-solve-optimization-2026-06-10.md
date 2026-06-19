# Solve-path optimization (~1.5× over the prior solve)

> **상태**: reference   **갱신**: 2026-06-10
> **한 줄**: 이미 최적화돼 있던 solve 경로에서 추가로 ~1.4–1.7× 를 끌어낸 레버 모음 — **main 에 통합 완료**.

이 노트는 factorize 의 [`01-kernel-engineering.md`](01-kernel-engineering.md) 에 대응하는 **solve 측 최적화** 기록이다.
대상 코드는 이미 small-front subgroup packing, tier split, subtree multistream, CUDA graph replay, TF32 factor
경로를 갖춘 상태였고, 아래 레버들은 그 위에 얹혔다. CAP/factorize 소스는 변경하지 않았다. 측정 원자료:
[`solve_speedup.tsv`](04-solve-optimization-2026-06-10/solve_speedup.tsv) (baseline = 통합 직전 코드).

## 분석 — 어디가 병목인가

`case_ACTIVSg25k, B=256` Nsight Systems 프로파일: solve 시간은 gather/scatter 나 regular-tier 가 아니라
**small-tier solve 커널**이 지배한다.

| kernel family | total (short profile) |
|---|---:|
| `solve_bwd_small<float,16>` | 9.92 ms |
| `solve_fwd_small<float,16>` | 7.26 ms |
| `solve_bwd_small<float,32>` | 5.97 ms |
| `solve_fwd_small<float,32>` | 5.04 ms |
| regular `solve_bwd<float>` | 5.33 ms |
| regular `solve_fwd<float>` | 4.63 ms |
| `scatter_sol` + `gather_rhs` | 1.79 ms |

> 텐서코어는 solve 의 일반 레버가 아니다 — 배치 각 항목이 **서로 다른 numeric factor 를 가진 독립 1-RHS 삼각해**라,
> batch 를 GEMM 의 RHS 차원으로 보는 건 동일 factor 일 때만 유효한 비일반 지름길이다.

## 통합된 레버 (accepted)

1. **small-tier subgroup packing 을 `level_max_nc` 기준으로** (구: `max_fsz`). 삼각 재귀는 contribution 행 전체(`fsz`)가
   아니라 pivot 열(`nc`) 만큼의 lane 이 필요하다. 6xxx/8xxx 의 small front 대부분은 `fsz>16` 이어도 `nc<=8` 이라,
   warp 당 4개의 8-lane subgroup 으로 돈다. `B=1` 은 occupancy fallback(SG32) 유지.
2. **float fast-divide** — TF32/FP32 front 의 solve 벡터는 float 이므로 backward pivot 나눗셈에 `__fdividef`, double 은 그대로.
3. **`B=1` multistream 비활성** — spine fusion 이후 stream/event 오버헤드가 USA `B=1` 에서도 손해. `B=64/256` 은 유지.
4. **spine fusion** — small/medium spine 은 forward+backward 를 한 커널(`solve_spine_chain`)로; large spine 은 shared/register
   압력 때문에 분리 유지. spine forward 는 sibling writer 가 없어 `atomicAdd` 대신 plain subtract.
5. **narrow regular backward reduction** — large regular front(`width>=128 && cb>=64`)은 row 당 1 스레드 직렬 `cb` 루프
   대신 warp-per-row dot reduction. 임계값을 낮추면 25K 가 회귀해 좁게 유지.
6. **full solve graph 캡처** — `gather + solve levels + scatter` 를 (rhs, sol, perm, iperm, type) 키로 lazy 캡처해 한 번에
   launch. 포인터/타입 변경 시 재캡처. graph launch 의 내부 host sync 제거(소비자가 이미 동기화).
7. **`B=1` fixed-`nc` regular 라우팅** — `level_max_nc ∈ {8,10,14,16,20}` 레벨을 fixed-`nc` 커널로 라우팅해 패널/CB 루프를
   unroll. 나머지 `nc` 는 동적 경로 유지(branch/code-size 페널티 회피). 한 range 의 모든 front 가 `nc∈{8,10,14,20}` 면
   no-fallback exact 커널; `nc=16` 은 exact 가 25K 회귀라 fallback-capable 유지.

부수: fixed-`nc` backward 커널은 `nc==NC` 분기 **후** 로드, `NC=14/16` 은 `cb>=8` 부터 warp-row reduce, scatter 는 `B=1`
도 inverse permutation 사용(`sol[orig]=y[iperm[orig]]` — 연속 store), full graph launch 의 redundant host sync 제거.

## 구현 (핵심 스니펫)

small-tier subgroup 선택 (`level_max_nc`):

```cpp
static int solve_small_sg(int max_nc, long warps_unpacked, int B)
{
    int sg = (max_nc <= 8) ? 8 : (max_nc <= 16 ? 16 : 32);
    const long packed_warps = (warps_unpacked + (32 / sg) - 1) / (32 / sg);
    if (B == 1 && packed_warps < solve_warp_fill() && packed_warps < 800) sg = 32;  // B=1 occupancy fallback
    return sg;
}
```

float fast-divide:

```cpp
__device__ __forceinline__ float  solve_div(float a,  float b)  { return __fdividef(a, b); }
__device__ __forceinline__ double solve_div(double a, double b) { return a / b; }
```

`B=1` multistream off:

```cpp
const bool use_multistream = st.batch_count > 1 && st.num_subtree_streams > 1 &&
                             plan.num_subtrees == st.num_subtree_streams &&
                             !plan.h_subtree_level_off.empty();
```

spine fusion / large regular warp-row reduce / full solve graph 는 각각 `solve/dispatch.cuh` 의 `solve_spine_chain`,
`bwd_level`, 그리고 `solve/solve.cu` 의 lazy graph 캡처에 있다. inverse scatter 는 `solve/permute.cuh` 의
`scatter_sol_inverse[_batched]`.

## 측정 (통합 직전 solve 대비, serial-ND seed 1588, repeat 31)

| case | B | before solve/sys (ms) | after solve/sys (ms) | speedup |
|---|---:|---:|---:|---:|
| case6468rte | 1 | 0.2675 | 0.1886 | 1.42× |
| case6468rte | 64 | 0.01222 | 0.00867 | 1.41× |
| case6468rte | 256 | 0.00901 | 0.00571 | 1.58× |
| case8387pegase | 1 | 0.2655 | 0.1878 | 1.41× |
| case8387pegase | 64 | 0.01569 | 0.01026 | 1.53× |
| case8387pegase | 256 | 0.01233 | 0.00733 | 1.68× |
| case_ACTIVSg25k | 1 | 0.5325 | 0.3794 | 1.40× |
| case_ACTIVSg25k | 64 | 0.03873 | 0.02609 | 1.48× |
| case_ACTIVSg25k | 256 | 0.03367 | 0.02185 | 1.54× |
| case_SyntheticUSA | 1 | 1.0446 | 0.7154 | 1.46× |
| case_SyntheticUSA | 64 | 0.13509 | 0.08007 | 1.69× |
| case_SyntheticUSA | 256 | 0.12786 | 0.07467 | 1.71× |

집계: `B=1` 중앙값 1.42×, `B=64` 1.51×, `B=256` 1.63×, 전체 12셀 1.40–1.71×. 정확도(relres)는 before/after 동일 band.

## 기각된 가설 (재시도 방지 기록)

- broad regular block-wide reduction (`cb>=16`): 25K 회귀 / 무이득.
- small-tier backward reduce 임계 `cb>4nc`→`2nc`: 8387/25K/USA B=256 전부 회귀.
- 동적 `nc<=32` solve 루프 unroll 힌트: 6468 B=1 0.246→0.310 회귀.
- row-unique forward plain-update(레벨/트리 스코프): unique 비율 5–13%뿐, flag 분기 비용이 회귀.
- `B=1` 1024/768/384-thread large-front, `64<fsz<=80` 무조건 512-thread, selective B=1 multistream, cooperative tail fusion,
  fixed-`nc` spine, exact `nc=16`, small-tier fixed-`nc` 분기 — 전부 측정 후 회귀로 reverted.

## 교훈

1. 이미 최적화된 solve 에도 실레버가 있었다: `max_fsz` 는 solve 에 과하게 보수적, `max_nc` 가 삼각 재귀의 정확한 lane 요구.
2. launch-count 정리가 `B=1` 에 실효: full graph 캡처 + spine fusion + redundant sync 제거.
3. regular backward reduction 은 **좁은 large-front gate** 에서만 유효 — 넓히면 즉시 회귀.
4. `B=64/256` 은 모든 케이스에서 ≥1.4× — inverse scatter 가 6xxx B64 의 빠진 조각이었다.
5. 일반적 TC solve 가속은 같은 factor 의 multi-RHS 가 필요해, 독립 시스템 배치 모델에선 비일반.
