# Multifrontal 레이아웃 + level-batched 커널 — STRUMPACK 과 무엇이 다른가

> **상태**: reference   **갱신**: 2026-06-10
> **한 줄**: 같은 multifrontal 알고리즘이지만 custom 은 front 를 단일 `fsz×fsz` 덩어리 + level 당 1 융합 커널로 처리하고 STRUMPACK 은 MAGMA vbatched 에 맞춰 front 를 4 조각 + level 당 7+ 호출로 분해 — 이 자료구조/커널 단위 차이가 graph 없이도 case8387 factor ~20× / solve ~21× 우위를 만든다.

이 문서는 [`01-why-custom-fast.md`](01-why-custom-fast.md) 의 "graph 는 보조 최적화, 본질은 알고리즘" 진술을 뒷받침하는 **소스 레벨 분해** 다. nsys 측정에서 CUDA Graph 를 끄고도 (factor 0.74 ms, solve 0.64 ms) STRUMPACK (14.32 / 11.24 ms) 대비 자릿수 우위가 유지되는 것을 봤다. 그 이유는 graph 가 아니라 **데이터 레이아웃 + 커널 단위 (granularity)** 의 설계 차이에 있다.

원본 코드 경로:
- custom: `/workspace/sparse_direct_solver/gpu-powerflow/custom_linear_solver/src/`
- STRUMPACK: `/opt/third_party/src/strumpack/src/sparse/fronts/`

---

## 0. TL;DR — 다섯 축

| 축 | STRUMPACK (CUDA + MAGMA vbatched) | custom_linear_solver |
|---|---|---|
| **Front 레이아웃** | front 1개 = `F11 / F12 / F21 / F22` 4 개 별도 `DenseMW` (각각 다른 leading dim) | front 1개 = `fsz × fsz` 한 덩어리 column-major (`front + front_off[p]`) |
| **Level 단위** | 한 level = **5–7 vbatched 호출** (`getrf → laswp → trsm×2 → getrs(big) → gemm → extend_add`) | 한 level = **단 1 커널** (`mf_factor_extend_level`) — factor + trailing + extend-add 융합 |
| **Front 디스패치** | MAGMA vbatched → 포인터 배열 (`F11[]`, `F12[]`, …) 을 호스트에서 빌드해 디바이스 복사 | `plcols[blockIdx.x]` 한 정수 → `front_off[p]` 로 base 도출, 포인터 배열 없음 |
| **Pivoting** | partial pivoting (in-kernel pivot search + `laswp_vbatched`) | **no pivot** — `if (piv == 0) piv = 1; *sing = 1` 로 표시만 |
| **Extend-add** | 별도 커널 `extend_add_kernel`, 인덱스 배열 `I1/I2` 로 scatter | 같은 factor 커널 안에서 `__syncthreads()` 후 `atomicAdd` 로 부모에 직접 scatter |
| **Solve** | 기본은 **factor 후 D2H → CPU forward/backward** | 전 단계 GPU 레지던트, level-batched `mf_fwd_level / mf_bwd_level` (옵션 selinv) |
| **CUDA Graph** | 없음 | 있음 (옵션). 없어도 위 구조적 이점으로 자릿수 빠름 |
| **Amalgamation 효과** | supernode 폭 ↑ 가 호출당 work 증가로만 전달 (level 당 호출 ~7 고정) | panel 폭 ↑ 가 panel 수 + level 수 양쪽에 곱해져 launch 횟수 자체 ↓ (§6) |

*level-batched 커널* 의 의미가 다르다: STRUMPACK 의 "batch" 는 **MAGMA vbatched API 의 batch 차원** (한 BLAS 연산을 N front 동시 호출). custom 의 "batch" 는 **한 level 의 모든 front 의 모든 연산 (factor + extend-add) 을 단 하나의 커널** 로 묶는 것.

---

## 1. 같은 출발점 — 어디까지가 공통인가

`docs/01-orientation/03-lineage-strumpack-not-the-baseline.md` 와 중복은 짧게: 둘 다 **METIS NodeND** reordering, **elimination tree** 빌드, **supernode amalgamation** (custom 은 panel-based 변형, STRUMPACK 은 supernodal), **frontal matrix** = pivot block + Schur complement (CB), **extend-add** (child CB 를 parent front 로 흘려넣는 multifrontal 본질 연산). 본 문서가 다루는 차이는 *그 다음* — front 를 GPU 메모리에 어떻게 펴고 어떤 커널 단위로 처리하느냐.

---

## 2. 차이 ① — Front 데이터 레이아웃

### 2.1 STRUMPACK: 한 front = 네 개의 별도 `DenseMW`

`FrontGPU.cpp:155-200` 의 `set_factor_pointers` / `set_work_pointers` 가 front 1개를 4 객체로 분리:
- `F11_` (dsep × dsep, ld = dsep) — 대각 pivot 블록
- `F12_` (dsep × dupd, ld = dsep) — U panel
- `F21_` (dupd × dsep, ld = dupd) — L panel
- `F22_` (dupd × dupd, ld = dupd, work 영역) — Schur (CB)

4 개의 `DenseMW` 가 각각 다른 leading dimension 을 가진다. 이를 다시 *level 전체에 걸쳐* 4 개의 포인터 배열로 모은다 (`FrontMAGMA.cpp:704`): `auto F11 = L.dev_F_batch; auto F12 = F11 + Nsmall; ...`. MAGMA 의 `*_vbatched_*` API 가 *array of pointers + array of leading dimensions* 을 받는 형식이라 이 분리가 강제된다.

### 2.2 custom: 한 front = `fsz × fsz` 단일 column-major 덩어리

`multifrontal_plan.cu:576-587` 에서 panel 별 offset 은 fsz² 의 prefix-sum:

```cpp
long total = 0;
for (int p = 0; p < P; ++p) {
    front_off[p] = total;
    const long fsz = front_ptr[p + 1] - front_ptr[p];
    total += fsz * fsz;
}
```

factor 커널은 이 단일 base 에서 직접 인덱싱 (`factorize/multifrontal.cu:43-51`): `FT* F = front + front_off[p];` 한 포인터로 모든 영역 접근. front 안의 모든 영역은 같은 base + 같은 stride (`fsz`):
- L panel: `F[i*fsz + k]`, `i ∈ [k+1, fsz)`, `k ∈ [0, nc)`
- U panel: `F[k*fsz + j]`, `k ∈ [0, nc)`, `j ∈ [nc, fsz)`
- Trailing/CB: `F[i*fsz + j]`, `i, j ∈ [nc, fsz)`

→ 별도 `F11/F12/F21/F22` 객체 없음. **분기는 인덱스에만 있고, 메모리에는 없다.**

### 2.3 무엇이 달라지나

| 항목 | STRUMPACK | custom |
|---|---|---|
| Front 1개당 별도 메모리 객체 | 4 (`F11,F12,F21,F22`) | 1 |
| Level N front 디스패치 메타데이터 | 4 포인터 배열 + 4 leading-dim 배열 + 2 dim 배열 | `plcols[]` 1개, `front_off[]` 1개 (한 번 빌드 후 고정) |
| 커널 내부 메모리 접근 | 4 base pointer + stride 4 종류 | 1 base pointer + 1 stride |

STRUMPACK 이 잘게 쪼개는 이유는 **MAGMA vbatched 가 그렇게 받기 때문**. MAGMA 는 일반 dense BLAS (getrf, trsm, gemm) 의 batch 버전이고 각각 독립 행렬을 받도록 설계 → 한 front 의 L/U/Schur 가 *논리적으로 같은 큰 행렬의 부분* 이라는 정보가 API 경계에서 잘려나간다. 이게 차이 ②를 강제한다.

---

## 3. 차이 ② — Level 의 단위: 5–7 vbatched vs 1 융합 커널

### 3.1 STRUMPACK 의 한 level

`FrontMAGMA.cpp:700-755` 의 한 level 처리는 7+ 커널 launch:
1. `getrf_vbatched` (small fronts, factor pivot block)
2. cuSOLVER getrf loop (big fronts, N개 launch)
3. `laswp_fwd_vbatched` (row swap from pivoting)
4. `trsm_vbatched` (Lower) on F12
5. `trsm_vbatched` (Upper) on F12
6. cuSOLVER getrs loop (big fronts)
7. `gemm_vbatched` (rank-d1 trailing update `F22 -= F21 * F12`)
8. `extend_add_kernel` (left + right children → 2 launches) — **또 별도 커널** (`FrontCUDA.cu:114-148`)

case8387pegase nsys (`cuda_gpu_kern_sum`) 에서 STRUMPACK 한 NR iter 가 ~700 커널 launch 인 것이 이 구조의 직접 결과.

### 3.2 custom 의 한 level

`factorize/multifrontal.cu:34-122` 의 단일 커널 `mf_factor_extend_level` 하나가 한 level 의 **모든 front, 모든 단계 (panel LU + trailing + extend-add)** 처리:
- `fsz <= 48`: 단순 rank-1 루프
- `fsz > 48`: blocked rank-nc — Phase 1 panel LU, Phase 2 U panel TRSM, Phase 3 trailing GEMM
- **`__syncthreads()` 후 extend-add** (atomicAdd to parent):

```cpp
const int par = panel_parent[p];
if (par < 0 || !do_extend) return;
__syncthreads();
FT* Fp = front + front_off[par];
const int pfsz = front_ptr[par + 1] - front_ptr[par];
const int abase = asm_ptr[p];
for (int e = t; e < uc * uc; e += nt) {
    const int a = e / uc, b = e % uc;
    atomicAdd(&Fp[(long)asm_local[abase + a] * pfsz + asm_local[abase + b]],
              F[(long)(nc + a) * fsz + (nc + b)]);
}
```

부모는 더 높은 level 에 있으므로 *현재 launch 에서 동시에 쓰는 다른 block 없음* → race-free. 커널 헤더 주석: *"higher level (not factored this launch), so the atomicAdd is race-free. Halves the graph nodes and removes one inter-kernel sync per level (~72 levels on SyntheticUSA)."*

→ 한 level 에 **1 커널 launch**. case8387 의 31 levels × 2 iter = 62 launch 가 `mf_factor_extend_level` instances=60 (root 제외) 으로 일치.

### 3.3 호출 횟수 비교 (한 NR iter)

| 단계 | STRUMPACK | custom |
|---|---:|---:|
| Factor 본체 | ~700 (level × kernel-type) | ~31 (1 per level) |
| Solve forward | ~50 | 31 |
| Solve backward | ~50 | 31 |
| 합계 | ~800 | ~95 |

이 차이가 nsys 의 *"둘 다 latency-bound 인데 wall-time 은 14×"* 의 직접 원인. **custom 은 launch 횟수가 약 1/8.**

---

## 4. 차이 ③ — Extend-add 의 융합 (custom) vs 분리 (STRUMPACK)

**STRUMPACK**: 별도 커널 `extend_add_kernel` + 인덱스 배열 + grid sweep. 호스트가 미리 `AssembleData<T>` 안에 `CB1, CB2, I1, I2, F[0..3]` 포인터를 채워줘야 함 (`FrontGPU.cpp:301-345`). CB 가 *child 의 F22 영역* 에 살아남아 부모 level extend-add 가 읽어가야 함 → **CB 가 work 메모리에서 차지하는 영역이 child level 끝나도 풀리지 않음**.

**custom**: 같은 커널, atomicAdd, scatter 인덱스는 *symbolic 단계* 에서 `asm_ptr[]` / `asm_local[]` 로 사전 baked (`multifrontal_plan.cu:640-646`). `asm_local[abase + a]` 는 child 의 a 번째 CB 행이 *부모 front 안에서 몇 번 인덱스인지* 미리 계산 (parent-local index). 커널이 하는 일은 atomicAdd 한 줄 — binary search 없음, 자식 CB 메모리 해제 가능, 부모와 같은 column-major arena 에 직접 누적. **CB 라는 별도 객체가 존재하지 않는다.**

**메모리 효과**: STRUMPACK 은 한 level 처리 동안 parent CB (F22) 를 계속 들고 있다가 부모 level 에서 읽어감 → peak 메모리 = 부모 chain 전체 F22 합. custom 은 factor + extend-add 융합 후 자식 CB 영역이 논리적으로 더 안 쓰임. 차이는 *피크 메모리 패턴* 이지 *총량* 은 아니다 (둘 다 etree 전체 front 메모리 보유).

---

## 5. 차이 ④ — Pivoting: partial pivoting + `laswp` overhead vs no-pivot

**STRUMPACK**: `LU_block_kernel` (`FrontCUDA.cu:234-294`) 내부에서 small fronts (NT ≤ 32) 에 in-kernel partial pivoting (각 k 마다 column max search → row swap). big fronts 는 cuSOLVER getrf. F12 (U panel) 도 같은 행 교환 적용 위해 **별도 커널** `laswp_fwd_vbatched` launch. ncu (`/tmp/bench/ncu_strumpack_8387.csv`) 에서 `laswp_vbatch_kernel` SM% = 0.1%, DRAM% = 0.2% — **순수 overhead**, 47 instances/iter.

**custom**: no-pivot (`factorize/multifrontal.cu:54-58`):

```cpp
for (int k = 0; k < nc; ++k) {
    FT piv = F[(long)k * fsz + k];
    if (piv == FT(0)) { if (t == 0) *sing = 1; piv = FT(1); }
    for (int i = k + 1 + t; i < fsz; i += nt) F[(long)i * fsz + k] /= piv;
```

→ pivot 검색 없음, row swap 없음, `piv[]` 배열 없음, `laswp` 커널 없음. 정확성은 power-flow Jacobian 의 주대각 우세 + diagonal-perturbation safe 특성에 의존 ([`02-no-pivoting-proof.md`](02-no-pivoting-proof.md)). 실패 시 `*sing = 1` flag.

**절감**: `laswp_vbatch_kernel` × 47 launches, in-kernel pivot search (각 k 마다 reduction + 2× `__syncthreads()` + 잠재적 row swap), `piv[]` / `dev_ipiv_batch` 메모리 + I/O — custom 은 *전부 0*. pivoting 이 *power-grid Jacobian 의 조건수 특성으로 안전* 하다는 *알고리즘적 결정* 이 이 비용을 제거한다.

---

## 6. 차이 ⑤ — Solve 경로: CPU 폴백 vs GPU 레지던트 + level batching

**STRUMPACK**: 기본 경로는 *factor → D2H 복사 → CPU solve*. factor 후 `gpu::copy_device_to_host_async` (`FrontMAGMA.cpp:745-748`), forward solve `fwd_solve_phase2` (`FrontGPU.cpp:698-712`) 가 `F11_.solve_LU_in_place(bloc, piv_, ...)` — **CPU LAPACK**. "Solve is performed on CPU" 로그의 이유. nsys 의 `extract_rhs_kernel`, `gemvn_kernel_vbatched` 가 SM 0.7-1.0% 인 것도 *GPU 위에 호스트 동기화 + 잘게 흩어진 work 만 남아서*.

**custom**: `mf_fwd_level` (`solve/multifrontal.cu:15-57`) 이 한 level 모든 front 를 한 커널로. `selinv=true` (pivot 블록의 inverse 를 `mf_invert_pivot` 으로 사전 계산) 면 sequential trsv 가 parallel gemv 가 됨. trailing update 도 atomicAdd 로 fused. 백워드도 level-batched `mf_bwd_level` 한 개.

**Solve 의 자릿수 차이가 여기서 나온다** (case8387 nsys):
- STRUMPACK iter2 solve: 11.24 ms
- custom (graph ON): 0.34 ms (**33×**)
- custom (graph OFF): 0.53 ms (**21×**)

직접 원인: STRUMPACK 은 factor 끝마다 D2H + 호스트 per-front LAPACK 으로 도는 동안 GPU 거의 idle. custom 은 GPU 레지던트 + level batching + selinv → 31 levels × 2 kernels = 62 launch 안에 모든 일. **graph OFF 에서도 21× 빠르다 — 본질은 데이터를 GPU 에 둔 채 level batching.**

---

## 7. Amalgamation — `nc`, `cap`, panel 의 의미와 cap sweep 실측

### 7.1 용어

- **Front size `fsz`** — 한 front 행 수 = pivot 컬럼 수 (`nc`) + Schur (CB) 보조 행 수. 메모리 `fsz²`, factor 작업 `~fsz³`.
- **Pivot column count `nc`** — 한 front 에서 동시에 제거되는 컬럼 수 = panel 에 amalgamate 된 컬럼 수. 큰 nc: dense LU 작업당 효율 ↑, launch 수 ↓. 단점: nest 안 하면 빈 슬롯 0 저장 (**padded fill 증가**).
- **Supernode (Liu-Ng-Peyton fundamental)** — below-diagonal pattern 이 정확히 nest (`colcount[jprev] == colcount[j] + 1`) + 유일 자식일 때 병합 (`symbolic/supernode.cpp:23-40`). fill 증가 없음. power-grid 평균 폭 약 2 컬럼 — dense GPU 커널 단위로는 너무 작음.
- **Panel (relaxed amalgamation)** — fundamental supernode 가 작아서 추가로 *etree chain* (`parent[j] == j+1`) 따라 greedy 병합, cap 컬럼까지 (`supernode.cpp:63-82`). colcount 다르면 가장 넓은 것으로 padded → *작은* fill 증가. 안전성 주석: child CB 가 한 parent front 에 nest 해야 하므로 etree chain 만 병합 (cross-subtree merge 는 별도 `MF_AMALG` 경로로 격리).
- **`panel_cap` / `CLS_CAP`** — panel 당 최대 컬럼 수 = nc 상한. 적응적 (`factorize/multifrontal.cu:561`):

```cpp
int eff_cap = n >= 80000 ? 20 : (n >= 16000 ? 12 : panel_cap);  // panel_cap = 8 default
if (const char* ce = std::getenv("CLS_CAP")) eff_cap = std::atoi(ce);
```

cap trade-off: ↑ → panel 수 P ↓, level 수 ↓, panel dense work ↑ → **launch ↓, solve loop 짧아짐**. 단 ↑ → padded fill ↑, nc 가 `MF_REG_NC = 16` (register-partial 경로 상한) 넘으면 fast path 깨짐 → **factor 회귀**.

### 7.2 cap sweep 실측 (graph OFF 빌드, RTX 3090, iter2)

**case8387pegase (n=14908, 기본 cap=8)**

| `CLS_CAP` | P | levels | front 메모리 (MB f32) | factor [ms] | solve [ms] |
|---:|---:|---:|---:|---:|---:|
| 1 (amalg off) | 14908 | 171 | 6.1 | 1.31 | 1.55 |
| 4 | 7638 | 49 | 2.4 | 0.72 | 0.59 |
| **8** (default) | 7418 | 30 | 2.1 | 0.72 | 0.49 |
| 12 | 7321 | 24 | 2.0 | 0.75 | 0.47 |
| 16 | 7311 | 20 | 1.9 | 0.82 | 0.45 |
| 20 | 7350 | 21 | 1.9 | **34.4** (regress) | — |

**case_ACTIVSg25k (n=47246, 기본 cap=12)**

| `CLS_CAP` | P | levels | factor [ms] | solve [ms] |
|---:|---:|---:|---:|---:|
| 1 | 47246 | 281 | 3.30 | 2.97 |
| 8 | 22950 | 47 | 1.48 | 1.00 |
| **12** (default) | 22784 | 33 | 1.40 | 1.15 |
| 16 | 22689 | 31 | 1.67 | 1.08 |
| 20 | 22657 | 28 | **45.1** (regress) | — |

### 7.3 관찰

1. **Amalgamation 본체 효과 = level 수 감소.** case8387 cap=1 → cap=8: levels 171 → 30 (**5.7×**). solve 가 level-batched launch loop 라 *level 수 ≈ launch 수* → solve 1.55 → 0.49 ms (**−68%**).
2. **Factor 도 동반 개선.** case8387 factor 1.31 → 0.72 ms (**−45%**). panel 수 ↓ (14908 → 7418) launch 감소 + dense LU 효율 소폭 상승.
3. **Pareto-crossover 존재.** cap=8 → 16 (case8387): factor 9% 악화 / solve 8% 개선. cap 은 *factor ↔ solve 트레이드* — 매트릭스마다 sweet spot 다른 게 **n-adaptive cap (8/12/20) 의 정당성**.
4. **상한선 실제 위치는 cap ≈ 16.** cap=20 에서 두 case 모두 factor 폭증 (34/45 ms). 원인은 nc > 16 이 `MF_REG_NC=16` fast path 를 깸. n ≥ 80k 큰 매트릭스는 큰 fronts dominant 라 cap=20 도 안전 → *n ≥ 80k 에서만* cap=20 허용.
5. **Padded fill 비용은 작다.** case8387 front 메모리 6.1 MB (cap=1) → 1.9 MB (cap=16) — *더 줄어든다*. amalgamation (P 감소) 이 padding 증가보다 큰 효과. power-grid 의 *etree chain 길고 colcount 거의 일정* 특성이 padding cost 를 작게.

### 7.4 STRUMPACK 과의 대비

STRUMPACK 도 supernode amalgamation 을 한다 (Ashcraft 표준). 그러나:
- amalgamated supernode 가 결국 *MAGMA vbatched 입력* 으로 가서 **다시 4 조각 (F11/F12/F21/F22) 으로 쪼개짐** (§2). "한 덩어리" 정보가 API 경계에서 잘려나감.
- 따라서 amalgamation 효과가 *각 vbatched 호출의 평균 dense work 증가* 로만 전달 — **level 당 호출 수는 supernode 수가 아니라 연산 종류 (~7) 로 고정** → launch 수가 amalgamation 으로 줄지 않음.
- custom 에서는 amalgamation block 이 *그대로 한 커널 한 thread block 의 작업 단위* → **panel 수 감소가 곧 launch 수 감소**.

→ amalgamation 은 같은 알고리즘 결정이지만 *§3 의 level 단위 차이* 가 효과 배율을 다르게 만든다. STRUMPACK 에서 1.5× 개선이 custom 에서는 *panel 수 + level 수* 양쪽에 곱해져 더 큰 개선으로 전달.

---

## 8. 양적 매핑 — nsys 측정과 위 차이의 연결

case8387pegase, iter2:

| 단계 | STRUMPACK [ms] | custom (graph) | custom (no graph) |
|---|---:|---:|---:|
| factor | 14.32 | 0.643 | 0.699 |
| solve | 11.24 | 0.341 | 0.530 |

**Factor 14.32 → 0.699 ms (graph off, 20×)**:
- §3 level 당 ~7+ 커널 vs 1 커널 — 가장 큰 단일 원인. 31 levels × ~7 ≈ 217 launches vs 31. STRUMPACK `gemm_template_vbatched_nn_kernel<16,16,48>` instances=146, `trsm_template_vbatched...` 152 등.
- §5 partial pivoting overhead — `laswp_vbatch_kernel` 47 instances/iter + in-kernel pivot search 동기화.
- §2 4× 디스패치 메타데이터 — MAGMA 가 호스트에서 빌드하는 포인터/leading-dim 배열.
- §3 small-front MAGMA tile mismatch — `gemm_template_vbatched<16,16,48>` sweet spot fsz ≥ 64, 그러나 case8387 max fsz=96, 96.2% 가 fsz ≤ 16 → idle tile 비중 압도.

**Solve 11.24 → 0.530 ms (graph off, 21×)**:
- §6 CPU 폴백 — 가장 큰 단일 비중. ncu SM 0.7-1.0% 가 *"GPU 거의 안 씀"* 증거.
- §6 level batching 없음 + selinv (trsv → gemv).

**Graph (ON vs OFF) — graph 효과의 위치**:

| 단계 | graph ON | graph OFF | graph 효과 |
|---|---:|---:|---:|
| factor | 0.643 | 0.699 | +9% (0.056 ms 절감) |
| solve | 0.341 | 0.530 | +55% (0.189 ms 절감) |

graph 가 잡아주는 건 *kernel launch overhead*. solve 의 커널 한 개 work 가 더 작아 launch overhead 비중이 커 → solve graph 효과 (55%) 가 factor (9%) 보다 큼. **Graph 가 절감한 절대값 (0.245 ms) vs 알고리즘이 절감한 절대값 (24.3 ms) — 약 100× 차이.** "graph 보다 알고리즘이 더 영향 크다" 가 정확히 그 비율로 확인된다.

---

## 9. 정리 — "multifrontal 레이아웃 + level-batched 커널" 의 정확한 의미

custom 의 "*multifrontal 레이아웃*":
1. 한 front = `fsz × fsz` 한 덩어리 column-major (별도 F11/F12/F21/F22 분리 없음, §2)
2. 모든 front 가 단일 device arena 에 prefix-sum 오프셋 배치
3. extend-add scatter 인덱스 (`asm_local[]`) 가 symbolic 단계에서 *부모-로컬 인덱스* 로 미리 변환 (§4)

custom 의 "*level-batched 커널*":
1. 한 level 의 모든 front + 모든 단계를 단일 `mf_factor_extend_level` 처리 (§3)
2. block id 한 정수로 panel id 도출 → 포인터 배열 디스패치 없음
3. partial pivoting 제거 (§5) 로 `laswp` + 동기화 + 메타데이터 제거
4. solve 도 같은 level batching (§6)
5. relaxed amalgamation (§7) 으로 panel 폭 키워 *panel 수 + level 수* 동시 감소

STRUMPACK 은 같은 multifrontal 알고리즘이지만 **MAGMA vbatched 일반 dense BLAS API 에 맞춰 front 를 4 조각으로 쪼개고 한 level 을 7+ vbatched 호출로 분해**. 이 API 일치는 *일반 dense LU* 에서는 합리적이지만 *fsz ≤ 96 의 극단적 small front 영역* (power-grid Jacobian) 에서는 vbatched 한 번의 work 가 launch overhead 수준 / 4 조각 메타데이터 host 부담 / partial pivoting `laswp` overhead / solve CPU fallback 의 host-device sync 폭증 — 따라서 같은 latency-bound 영역인데 wall-time 자릿수 차이. **이건 graph 가 만든 차이가 아니라 custom 이 *small-front 특화 자료구조 + 커널 단위 단일화* 를 적극 선택한 결과.** Graph 는 그 위 *추가* 10-50% 최적화일 뿐.

---

## 부록 A — 본 문서에서 인용한 소스 좌표

### custom_linear_solver
- `src/factorize/multifrontal.cu:30-122` — `mf_factor_extend_level` (factor + extend-add 융합)
- `src/factorize/multifrontal.cu:412` — `MF_REG_NC = 16` (register-partial nc 상한)
- `src/factorize/multifrontal.cu:423-457` — `mf_invert_pivot` (selinv 용 pivot block inverse)
- `src/factorize/multifrontal.cu:561-564` — `eff_cap` n-적응 (cap = 8/12/20, `CLS_CAP` override)
- `src/symbolic/supernode.cpp:5-43` — fundamental supernode (Liu-Ng-Peyton)
- `src/symbolic/supernode.cpp:45-84` — `relaxed_panels` (etree chain greedy merge)
- `src/solve/multifrontal.cu:15-57` — `mf_fwd_level`; `:69-151` — `mf_bwd_level`
- `src/plan/multifrontal_plan.cu:576-587` — front_off (fsz² prefix-sum); `:640-646` — `asm_local[]` (parent-local extend-add 인덱스)

### STRUMPACK
- `src/sparse/fronts/FrontGPU.cpp:155-200` — `set_factor_pointers / set_work_pointers` (F11/F12/F21/F22 4 분리)
- `src/sparse/fronts/FrontMAGMA.cpp:700-755` — 한 level 의 7+ vbatched 호출 시퀀스
- `src/sparse/fronts/FrontCUDA.cu:114-148` — `extend_add_kernel` (별도 패스 scatter)
- `src/sparse/fronts/FrontCUDA.cu:234-294` — `LU_block_kernel` (in-kernel partial pivoting)
- `src/sparse/fronts/FrontGPU.cpp:698-712` — `fwd_solve_phase2` (CPU 폴백 solve)

### nsys / ncu 측정
- `/tmp/bench/nsys/custom_nr_8387.nsys-rep` (graph ON), `custom_nr_nograph_8387.nsys-rep` (graph OFF), `strumpack_nr_8387.nsys-rep` (MAGMA)
- `/tmp/bench/ncu_custom_8387.csv` / `ncu_strumpack_8387.csv` — kernel-bound 분류

### 관련 본 저장소 문서
- `../main-report.md` — 전체 서사 맥락
- [`01-why-custom-fast.md`](01-why-custom-fast.md) — 가속 요인 우선순위 (본 문서 결론과 같은 방향)
- [`02-no-pivoting-proof.md`](02-no-pivoting-proof.md) — §5 no-pivot 안전성 근거
- [`04-gemm-fraction-tc-ceiling.md`](04-gemm-fraction-tc-ceiling.md) — FP32 mixed-precision 변형
- `docs/04-benchmarks-profiling/05-strumpack-vs-custom-multifrontal-case8387.md` — front-size 분포 + ncu bound 분류
- `docs/01-orientation/03-lineage-strumpack-not-the-baseline.md` — 알고리즘 패밀리 동일성
