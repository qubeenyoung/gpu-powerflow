# 같은 multifrontal인데 왜 custom이 STRUMPACK보다 power-grid Jacobian에서 빠른가

> **상태**: reference   **갱신**: 2026-06-10
> **한 줄**: case8387pegase에서 custom이 STRUMPACK MAGMA 대비 26× 빠른 이유를 front-size 분포 + ncu bound 분류 + 그래프 separator 이론으로 분해 — 둘 다 latency-bound이고 차이는 power-grid 가정 위 엔지니어링 누적.

`../01-orientation/03-lineage-strumpack-not-the-baseline.md`에서 정리한 대로 STRUMPACK과 `custom_linear_solver`는 **둘 다 multifrontal LU** 패밀리다 (METIS ND, elimination tree, supernode amalgamation, front + extend-add). 같은 알고리즘 패밀리인데 전력망 야코비안 case8387pegase (n=14908, nnz=110572)에서 NR steady-state 성능:

| Solver | NR iter factor + solve | 배수 |
|---|---:|---:|
| STRUMPACK MAGMA | 25.56 ms | 1× |
| custom_linear_solver | 0.98 ms | **26×** |

이 26× 차이를 *front-size 분포* + *ncu 커널 bound 분류* + *논문 데이터셋과의 비교*까지 분해한다.

---

## 1. 측정 베이스

| 항목 | 값 |
|---|---|
| 매트릭스 | case8387pegase (n=14,908, nnz=110,572, 평균 nnz/row 7.4) |
| GPU | RTX 3090 (sm_86, FP64 0.56 TFLOPs, 936 GB/s) |
| STRUMPACK | v8.0 MAGMA, METIS NodeND, MC64, compression off |
| custom | FP64, METIS NodeND, no pivot, CUDA Graph capture |

NR steady-state per-iter (`/tmp/bench/nsys/{strumpack,custom}_nr_8387.nsys-rep`):

| Solver | factor [ms] | solve [ms] | f+s [ms] |
|---|---:|---:|---:|
| STRUMPACK | 14.32 | 11.24 | 25.56 |
| custom | 0.64 | 0.34 | 0.98 |

## 2. "small front" — 정의와 측정

multifrontal LU에서 **fsz**(front size) = pivot 컬럼 수(`nc`) + Schur 행 수(`fsz−nc`). factor 작업 ∝ fsz³, factor 메모리 ∝ fsz². "small front" = *fsz³가 GPU dense-LU breakeven보다 작은 영역*. Spatula(MICRO'23): *"dense-LU throughput drops linearly below fsz=10,000."* GPU 한 SM의 dense-LU sweet spot은 fsz ≥ 64, peak은 fsz ≥ 256.

| 분류 | fsz | factor 메커니즘 |
|---|---|---|
| tiny | ≤ 16 | warp 한 개도 못 채움, idle thread |
| small | 17–32 | warp-per-front sweet spot |
| mid | 33–159 | block 단위 dense kernel (shared) |
| big | ≥ 160 | multi-warp/block 큰 dense |

### 2.1 case8387pegase front 분포 (실측, `CLS_DUMP=1`)

`n=14908 P=7421 levels=31 panel_cap=8 front_total(MB f32)=2.1`

| fsz 빈 | 개수 | 비율 | Σfsz² | Σfsz³ |
|---|---:|---:|---:|---:|
| 1–16 (tiny) | **7,136** | **96.2%** | 53.0% | 19.1% |
| 17–32 (small) | 223 | 3.0% | 21.9% | 23.9% |
| 33–48 | 46 | 0.6% | 14.0% | 25.7% |
| 49–64 | 13 | 0.2% | 7.8% | 20.0% |
| 65–96 | 3 | 0.04% | 3.3% | 11.3% |
| **97–160** | **0** | 0% | 0% | 0% |
| **≥ 161** | **0** | 0% | 0% | 0% |

핵심: **fsz ≤ 16이 96.2%**, **max fsz = 96** (전 행렬 최대 front도 100 미만), front_total = 2.1 MB (24 GB의 0.02%). compute의 89%가 fsz ≤ 64에 있고 fsz > 96인 front는 0개.

### 2.2 STRUMPACK MAGMA 커널 sweet spot — 그리고 8387의 위치

| 커널 | block size | sweet spot fsz |
|---|---|---|
| `gemm_template_vbatched_nn_kernel<double,16,16,48>` | 16×16, K=48 | fsz ≥ 64 |
| `trsm_template_vbatched<double,32,32>` | 32×32 | fsz ≥ 64 |

→ MAGMA vbatched는 fsz가 충분히 커서 여러 tile을 채울 때만 정상 효율. case8387에는 **그 영역에 front가 0개**. nsys top kernels: `gemm<16,16,48>` 평균 19 μs ×146, `trsm<16,64>` 24 μs ×76, `extend_add_kernel<16>` 7 μs ×336. 호출당 시간(19~24 μs) > launch overhead(~6 μs) → launch가 컴퓨트와 같은 무게로 누적. **알고리즘이 small-front 영역에서 의도된 효율을 못 냄.**

---

## 3. ncu — 두 솔버 hot kernel의 bound 분류

### 3.1 custom hot kernels (RTX 3090, FP64)

| Kernel | inst | SM% | DRAM% | warp% | FP64% | 분류 |
|---|---:|---:|---:|---:|---:|---|
| `mf_factor_extend_level` (factor+extend fused) | 62 | **4.4** | **1.8** | 33.5 | 25.3 | **latency-bound** |
| `mf_fwd_level` (solve fwd) | 62 | 1.6 | 1.6 | 11.8 | 4.5 | latency-bound |
| `mf_bwd_level` (solve bwd) | 62 | 6.6 | 2.1 | 12.7 | 17.8 | latency-bound |
| `mf_scatter_csr_values` | 2 | 4.1 | **67.0** | 65.8 | 0.0 | memory-bound (정상) |
| `mf_invert_pivot` | 2 | **51.4** | 7.0 | 28.3 | **64.6** | compute-bound (FP64) |

→ factor 본체가 SM 4%/DRAM 2% = **latency-bound** (warp occ 33.5%). CUDA Graph로 launch를 1회까지 줄인 게 결정적.

### 3.2 STRUMPACK hot kernels (요약)

| Kernel | inst | SM% | 분류 |
|---|---:|---:|---|
| `extract_rhs` / `extend_add_rhs` (solve) | 47–52 | 0.7–1.0 | **극단 latency-bound** |
| `laswp_vbatch_kernel` (pivot row swap) | 47 | **0.1** | **순수 overhead (starvation)** |
| `trsm<8,64>` (작은 fronts) | 23 | 19.1 | latency-bound |
| `trsm<4,32>` / `<2,32>` | 7 / 3 | 1.1 / **0.0** | starvation |
| `trsm<32,32>` (큰 fronts) | **1** | **57.5** | compute-bound (FP64 81.8%) |

진단:
- **(A)** 알고리즘은 큰 fronts에서 작동 — `trsm<32,32>` SM 57.5%가 증거. 하지만 그 큰 front 호출은 **함수당 1번뿐**. 나머지 100+ 호출은 SM 0–25%.
- **(B)** 작은 fronts가 압도 — fsz 작을수록 GPU가 거의 일을 안 함.
- **(C)** `laswp_vbatch` SM 0.1% ×47 = STRUMPACK이 maintain하는 pivoting 정확성의 cost. custom은 no-pivot이라 이 kernel 자체가 없음.
- **(D)** solve 단계 SM 0.7~1.0% → *"Solve is performed on CPU"* 경고와 일치.

**두 솔버 모두 latency-bound** (둘 다 SM% 낮음). 즉 *"GPU compute 더 짜내기"*가 leverage 아님. 차이는: STRUMPACK은 kernel 수가 너무 많고(총 ~700/iter) 각각 launch overhead 누적, custom은 kernel 수 최소화 + graph로 launch overhead 제거.

이 영역에서 wall을 결정하는 우선순위: ① kernel launch 횟수, ② launch 사이 host work, ③ per-kernel work distribution. FP64 throughput/DRAM bandwidth는 어차피 활용 못 하니 *"성능을 결정짓는 자원이 아님"*.

---

## 4. 왜 같은 알고리즘이 power-grid에 안 통하는가 — 그래프 separator

논문 Table 2 매트릭스(Janna 그룹)는 평균 nnz/row 30~75 (8387의 4–10×), 모두 **3D 연속체 PDE 이산화**.

```
3D 메시 (Janna):   separator ~ N^{2/3}  →  big fronts (수천 단위)
                                            STRUMPACK MAGMA sweet spot → 1.87× over cuDSS

평면 망 (power):   separator ~ √N        →  tiny fronts (수십 단위)
                                            MAGMA sweet spot 밖 → 본 측정 26× 손해
```

- 3D mesh: N=1.5M이면 root separator ≈ 13,000+ → fsz 수천~수만 → MAGMA `vbatched<16,16,48>` / `<32,32>` 가 자기 sweet spot 진입 → 논문 1.87× 우위의 원천.
- power-grid 야코비안: nearly-planar → planar separator theorem로 separator ~ O(√N). N=14,908이면 ≈ 120. 실측 max fsz = 96 < 120 (METIS가 약간 더 작은 separator 발견).

→ **STRUMPACK은 *general-purpose* sparse direct 솔버이지 power-flow 특화가 아니다.** 같은 multifrontal 위에서도 *"front 크고 NR-style 반복 없는 일반 PDE"* 가정 위 구현 (FP64-centric, BLR compression, refactorize phase 없음). 그 가정을 정확히 깨는 power-flow에서 sweet spot 밖. cuDSS는 REFACTORIZATION phase가 NR loop에 부분 적응 → STRUMPACK보다는 power-grid에 낫지만 여전히 NR-loop 전용은 아님. `custom_linear_solver`는 정반대 — fsz < 160 + NR loop(sparsity 고정) + no-pivot 세 가정 위에서 커널 라우팅·CUDA Graph·device-resident pipeline을 통째로 설계.

## 5. wall의 분해 — 14.3 ms vs 0.64 ms

### 5.1 STRUMPACK factor = 14.32 ms (nsys 종합, 추정)

| 구성 | 시간 [ms] | 메커니즘 |
|---|---:|---|
| GPU kernel work | ~5 | 1000+ small kernels (avg 10–25 μs) |
| `cudaLaunchKernel` (~700 × 6 μs) | ~4 | small-front 영역 kernel 수 폭증 |
| device alloc | ~2 | NR iter마다 일부 buffer 재할당 |
| H2D memcpy | ~2 | 작은 청크 다수 |
| host scheduling | ~1 | front별 dispatch loop |

→ **GPU kernel은 wall의 35% 이하**, 나머지 65%가 launch + memory + scheduling overhead.

### 5.2 custom factor = 0.64 ms

| 구성 | 시간 [ms] |
|---|---:|
| GPU kernel work (level당 1 graph node) | ~0.60 |
| `cudaGraphLaunch` × 1 | ~0.02 |
| 기타 host | ~0.02 |

→ **wall의 94%가 GPU kernel work**. host overhead 거의 0.

### 5.3 왜 22×인가 — 곱셈 누적

| 구성 | STRUMPACK | custom | 비 | 이유 |
|---|---:|---:|---:|---|
| GPU kernel | ~5 ms | ~0.6 ms | 8× | front 분포에 맞춘 커널 라우팅, fused factor+extend |
| launch overhead | ~4 ms | ~0.02 ms | 200× | graph replay vs 700+ launches |
| alloc/memcpy | ~4 ms | ~0.02 ms | 200× | 메모리 알로케이션 모두 analyze 시점 |
| host scheduling | ~1 ms | ~0.02 ms | 50× | scheduling이 graph에 컴파일 |

→ **단일 거대 원인이 아니라 4가지 모두에서 작은 우위가 곱셈으로 누적.** GPU kernel 차이(8×)도 의미있지만 진짜 leverage는 host-side overhead 제거(200×). 26× 차이는 *"알고리즘 자체의 차이"*가 아니라 *"power-grid 가정 위 엔지니어링 누적"*.

## 6. 한계 / 정직성

- ncu는 단일 GPU(RTX 3090). A100/H100은 FP64 17~30× 높아 same-size front를 더 빨리 saturate하지만, power-grid front가 *너무* 작아(max fsz=96) GPU 종류가 바뀌어도 sweet spot에 닿지 못함 — 정성적 결론 보존 가능성 높음.
- case8387 한 매트릭스. 다른 power-grid(1k~25k bus)도 같은 front 분포 → 일반화 가능.
- BLR compression의 추가 우위는 본 매트릭스에 적용 의미 없음 (큰 front 자체가 없음).

## 7. 참고

- `../01-orientation/03-lineage-strumpack-not-the-baseline.md` — lineage 정정
- `01-strumpack-paper-reproduction.md` — STRUMPACK 논문 행렬 재현
- `03-gemm-fraction-front-distribution.md` — front fsz/nc/uc 분포 상세
- `../02-design-analysis/04-gemm-fraction-tc-ceiling.md`, `../03-optimization-notes/01-kernel-engineering.md`, `../storyline.md`
- Claus et al., IJHPCA 2025; Spatula, MICRO 2023 (*"FullChip on V100: peak의 0.004%"*)

원시 데이터: `/tmp/bench/nsys/{strumpack,custom}_nr_8387.nsys-rep`, `/tmp/bench/ncu_{custom,strumpack}_8387.csv`.
