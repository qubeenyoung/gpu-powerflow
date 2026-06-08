# Solve: Spine Fusion + Multi-Stream Subtree

**작성일**: 2026-06-07
**목적**: B=1 PF Jacobian 솔브 wall 감축. master 솔브가 launch + memory latency bound 임을 ncu로 확정하고, 그 영역에서 두 lever (spine fusion / multi-stream subtree) 만 net positive로 검증·적용.
**적용 파일**: `src/solve/{dispatch,kernels,phases}.cuh` (+264 / −42 lines)
**측정 GPU**: RTX 3090, CUDA 12.8

## TL;DR

| Case | n | Master baseline (solve_ms) | After (solve_ms) | Δ |
|---|---|---|---|---|
| case_ACTIVSg10k | 18,544 | 0.442 | 0.424 | **−4.1%** |
| case9241pegase | 17,036 | 0.418 | 0.382 | **−8.6%** |
| case13659pegase | 23,225 | 0.504 | 0.467 | **−7.4%** |
| case_ACTIVSg70k | 134,104 | 1.55 | 1.50 | **−3.2%** |
| **평균** | | | | **−5.8%** |

5 trial × 1000 repeat median. residual 4 case 모두 baseline 수준 유지 (1e-12 ~ 1e-15).

## 진단 (ncu)

`solve_fwd` (regular tier):
- Warp Cycles Per Executed Instruction: **44.5**
- L1TEX scoreboard stall: **47%** of warp cycles
- Avg Active Threads Per Warp: **24/32**

`solve_bwd` (regular tier):
- Warp Cycles Per Executed Instruction: 34
- L1TEX stall: 37%
- Avg Active Threads Per Warp: **14.8/32** (`bwd_cb_subtract` 가 nc 개 thread만 활성)

각 kernel grid=(cnt, B) 의 SM throughput:
- grid 152 → 2.19%
- grid 84 → 1.44%
- grid 28 → 0.7%
- grid 1 → **0.05%** (spine, 81 SM idle)

**결론**: solve 는 compute-bound 아님. per-kernel 5–12 μs 대부분이 launch + scheduling + memory latency. 산술이나 SM 점유율은 floor 근처.

## 적용한 두 가지

### 1. Spine Fusion (`solve_*_spine`)

`plan.spine_start_level` 부터 num_plevels-1 까지의 cnt=1 chain (root 영역 5–9 개 panel) 을 단일 persistent kernel 로 융합.

- 한 block per batch 가 `h_spine_panels[]` 를 device-side loop 로 walk
- fwd: bottom → top, bwd: top → bottom
- panel 간 dependency 는 `__syncthreads` (블록 내, atomicAdd CB write 가 다음 panel load 에서 보이도록)
- block size 고정 128 thread (CB 단계가 256-thread 으로 가도 substitute 가 1 warp 이라 ROI 작음)

**왜 cnt=1 만?**: 원래 dispatch 가 그 level 에서 이미 grid=(1, B) — 1 SM 만 쓰던 영역. fusion 은 launch overhead 만 줄이고 SM 자원 사용은 무변화. 안전.

**왜 cnt>1 narrow band 는 안 됨?**: 시도했음 (`cnt ≤ 8`). 모든 case 회귀 (70k +38%, 9241pegase +19%). 원인: PF etree 의 cnt=2..8 영역은 root-area 큰 panel 들. 원래 dispatch 가 cross-SM 으로 분산하던 작업을 한 SM 으로 직렬화 → 손실.

### 2. Multi-Stream Subtree Solve

`factor` 에 이미 있는 multi-stream 패턴 (`State::subtree_streams[8]`, `fork_event`, `join_events[K]`) 을 솔브에 이식.

구조:
```
[main] fork → subtree_streams[0..K-1] forward sweep (levels 0..spine_lo-1)
       join → spine fwd + bwd (main stream)
       fork → subtree_streams[k] backward sweep (levels spine_lo-1..0)
       join
```

각 subtree 스트림이 자기 subtree 의 non-spine 작업을 독립적으로 진행 → 차원에 따라 K 배 SM 동시 occupancy 가능.

**활성 조건**: `st.num_subtree_streams > 1 && plan.num_subtrees == st.num_subtree_streams && !plan.h_subtree_level_off.empty()`. master batched 기본값으로 K ≤ 8 면 자동 활성.

**효과**: B=1 에서 marginal (subtree 당 work 가 작아 stream-event overhead 와 거의 상쇄). 그래도 net positive 또는 free. factor 가 이미 같은 stream 인프라 사용하므로 추가 비용 0.

## 폐기한 시도들

다음은 시도 후 회귀를 확인하고 폐기. 동일 영역 재시도 시 참고.

| 시도 | 결과 | 폐기 사유 |
|---|---|---|
| `SMALL_THRESH` 32→48/64/∞ | mixed, average regression | mid-fsz panel 에서 packed-warp 의 per-warp 직렬 work 가 block-overhead 절감보다 큼 |
| `fwd_cb_update` shared-mem staging | +9~18% 회귀 | strided F access 가 이미 L1 cache 흡수. staging 의 `__syncthreads` + shared 할당이 L1TEX stall 절감보다 큼 |
| `bwd_cb_subtract` cb-parallel regular tier | +11~25% 회귀 | nc=4 cb=10~20 영역에서 warp-reduce overhead 가 nc-parallel 직렬보다 큼. crossover ~cb > 7nc |
| narrow band persistent (cnt ≤ 8) | +7~38% 회귀 | cnt > 1 영역의 cross-SM 병렬 손실 |
| cooperative-grid narrow band | +3~7% 회귀 | grid_sync overhead ≈ per-level launch overhead in graph mode |
| `--single-precision fp32` | solve_ms 동일 | solve 는 compute-bound 아님 확정. precision throughput 차이 무관 |

## 다음 lever 후보 (시도하지 않음)

20% 이상의 추가 절감을 노린다면 다음 방향:

1. **B≥16 batched workload 측정**. B=1 에선 launch-overhead 영역에서 머무름. 16+ 에서 GEMM 패턴 의미. master `--batch 16` 으로 즉시 측정 가능.
2. **cuPF NR-loop 단위 측정**. 50-iter wall 에서 −5% solve 가 −2% wall. 별 의미 없으나, factor wall 과 비교에 따라 우선순위가 바뀜.
3. **다른 GPU**. H100/L40 는 FP64 throughput · L1 BW · async copy 모두 다름. 같은 코드의 절대값이 바뀜.
4. **selinv 부활 + batched GEMM 솔브**. R5 cost model 상 substitution phase 20% 영역만 영향 (CB phase 65% 는 불변). selinv 없이 CB batched-GEMM 이 더 큰 lever.

## 측정 재현

```bash
# Build
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCLS_CUDA_ARCHITECTURES=86
cmake --build build -j

# Measure (5 trial × 1000 rep, median)
for case in case_ACTIVSg10k case9241pegase case13659pegase case_ACTIVSg70k; do
  for trial in 1 2 3 4 5; do
    ./build/custom_linear_solver_run \
      --matrix /datasets/power_system/nr_linear_systems/$case/J.mtx \
      --rhs /datasets/power_system/nr_linear_systems/$case/F.mtx \
      --repeat 1000 | grep "^solve_ms"
  done
done

# ncu warp state (per-kernel stall breakdown)
ncu --kernel-name "solve_fwd" --launch-count 3 --section WarpStateStats \
    --graph-profiling=node ./build/custom_linear_solver_run \
    --matrix ${case}/J.mtx --rhs ${case}/F.mtx --repeat 2

# nsys per-kernel time
nsys profile --output=/tmp/solve --force-overwrite=true --trace=cuda \
     --cuda-graph-trace=node ./build/custom_linear_solver_run \
     --matrix ${case}/J.mtx --rhs ${case}/F.mtx --repeat 20
nsys stats --report cuda_gpu_kern_sum /tmp/solve.nsys-rep
```
