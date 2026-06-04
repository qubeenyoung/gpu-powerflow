# Multifrontal 레이아웃 + level-batched 커널 — STRUMPACK과 무엇이 다른가

*custom_linear_solver vs STRUMPACK (CUDA/MAGMA) — 같은 알고리즘 패밀리에서 GPU 구현이 어떻게 갈라지는가, 그리고 그 갈림이 case8387pegase의 자릿수 차이 (factor ~20×, solve ~21×) 를 어떻게 설명하는가*

이 문서는 `docs/02-design-analysis/02-acceleration-mechanism-ranked.md` 의 "graph는 보조 최적화, 본질은 알고리즘" 진술을 뒷받침하는 **소스 레벨 분해** 다. 이전 nsys 측정 (`/tmp/bench/nsys/custom_nr_*8387.nsys-rep`, `strumpack_nr_8387.nsys-rep`) 에서 CUDA Graph 를 끄고도 (factor 0.74 ms, solve 0.64 ms) STRUMPACK (14.32 ms / 11.24 ms) 대비 자릿수 우위가 유지되는 것을 봤다. 그 이유는 graph 가 아니라 **데이터 레이아웃 + 커널 단위 (granularity)** 의 설계 차이에 있다. 이 문서는 그 차이를 다섯 축으로 정리한다.

원본 코드는 다음 경로에서 인용한다 (모두 본 저장소 / OS 안):

- custom: `/workspace/sparse_direct_solver/gpu-powerflow/custom_linear_solver/src/`
- STRUMPACK: `/opt/third_party/src/strumpack/src/sparse/fronts/`

---

## 0. TL;DR

| 축 | STRUMPACK (CUDA + MAGMA vbatched) | custom_linear_solver |
|---|---|---|
| **Front 레이아웃** | front 1개 = `F11 / F12 / F21 / F22` 네 개의 별도 `DenseMW` (각각 다른 leading dim) | front 1개 = `fsz × fsz` 한 덩어리 column-major (`front + front_off[p]`) |
| **Level 의 단위** | 한 level = **5–7 개의 vbatched 호출** (`getrf → laswp → trsm×2 → getrs(big) → gemm → extend_add`) | 한 level = **단 1 개의 커널** (`mf_factor_extend_level`) — factor + trailing update + extend-add 까지 융합 |
| **Front 디스패치** | MAGMA vbatched API → 포인터 배열 (`F11[]`, `F12[]`, …) 을 호스트에서 빌드해 디바이스로 복사, MAGMA 가 tile-template 로 분기 | `plcols[blockIdx.x]` 한 정수로 panel id 결정 → `front_off[p]` 로 base 포인터 도출, 포인터 배열 없음 |
| **Small-front 경로** | `factor_block_batch<NT>` (NT ∈ {8,16,24,32}) hand-written + MAGMA vbatched fallback | 같은 커널이 `if (fsz <= 48)` 분기로 small/big 양쪽 처리, NT 분기 없음 |
| **Pivoting** | partial pivoting (in-kernel pivot search + `laswp_vbatched`) | **no pivot** — `if (piv == 0) piv = 1; *sing = 1` 로 표시만 |
| **Extend-add** | 별도 커널 `extend_add_kernel` (`FrontCUDA.cu:114-148`), 인덱스 배열 `I1/I2` 로 scatter | 같은 factor 커널 안에서 `__syncthreads()` 후 `atomicAdd` 로 부모에 직접 scatter |
| **Solve** | 기본은 **factor 끝나면 D2H 복사 → CPU에서 forward/backward** (`FrontGPU.cpp:702-712`) | 전 단계 GPU 레지던트, level-batched `mf_fwd_level / mf_bwd_level` (옵션 `selinv=true` 면 sequential trsv → parallel gemv) |
| **CUDA Graph** | 없음. 매 iter 마다 모든 커널 직접 launch | 있음 (옵션). 없어도 위의 구조적 이점으로 STRUMPACK 대비 자릿수 빠름 |
| **Amalgamation 효과** | supernode 폭 ↑ 가 vbatched 호출 당 평균 work 증가로만 전달 (level 당 호출 수는 ~7 로 고정) | panel 폭 ↑ 가 panel 수 + level 수 양쪽에 직접 곱해져 launch 횟수 자체를 줄임 (§ 7) |

여기서 *level-batched 커널* 의 의미는 두 솔버가 다르다. STRUMPACK 의 "batch" 는 **MAGMA vbatched API 의 batch 차원** — 한 BLAS 연산 (getrf, trsm, gemm) 을 N 개 front 에 대해 동시 호출. custom 의 "batch" 는 **한 level 의 모든 front 에 대한 모든 연산 (factor + extend-add) 을 단 하나의 커널** 로 묶는 것이다. 같은 단어가 가리키는 단위가 다르다.

---

## 1. 같은 출발점 — 어디까지가 공통인가

`docs/01-orientation/03-lineage-strumpack-not-the-baseline.md` 와 중복되는 부분은 짧게만:

- **Reordering**: 둘 다 METIS NodeND (nested dissection)
- **Elimination tree**: 둘 다 빌드
- **Supernode amalgamation**: 둘 다 (custom 은 `panel_cap` 기반의 panel-based 변형, STRUMPACK 은 supernodal)
- **Frontal matrix 정의**: pivot block + Schur complement (CB)
- **Extend-add**: child front 의 CB 를 parent front 의 해당 행/열로 흘려넣는 multifrontal 의 본질 연산

여기까지는 같다. 본 문서가 다루는 차이는 **그 다음** — front 를 GPU 메모리에 어떻게 펴고, 어떤 커널 단위로 처리하느냐 — 에 있다.

---

## 2. 차이 ① — Front 데이터 레이아웃

### 2.1 STRUMPACK: 한 front = 네 개의 별도 `DenseMW`

`FrontGPU.cpp:155-169` 의 `set_factor_pointers`:

```cpp
// FrontGPU.cpp:160 — "F11, F21, F11, F21, ..., F12, F12, ..."
void set_factor_pointers(scalar_t* factors) {
  for (auto F : f) {
    const std::size_t dsep = F->dim_sep();
    const std::size_t dupd = F->dim_upd();
    F->F11_ = DenseMW_t(dsep, dsep, factors, dsep); factors += dsep*dsep;
    F->F12_ = DenseMW_t(dsep, dupd, factors, dsep); factors += dsep*dupd;
    F->F21_ = DenseMW_t(dupd, dsep, factors, dupd); factors += dupd*dsep;
  }
}
```

그리고 `set_work_pointers` (`FrontGPU.cpp:179-200`) 가 F22 (CB) 를 별도 work 버퍼에 배치한다:

```cpp
// FrontGPU.cpp:184-189
auto schur = gpu::aligned_ptr<scalar_t>(wmem);
for (auto F : f) {
  const int dupd = F->dim_upd();
  if (dupd) {
    F->F22_ = DenseMW_t(dupd, dupd, schur, dupd);
    schur += dupd*dupd;
  }
}
```

즉 front 1개에 대해:
- `F11_` (dsep × dsep, ld = dsep)  — 대각 pivot 블록
- `F12_` (dsep × dupd, ld = dsep)  — U panel
- `F21_` (dupd × dsep, ld = dupd)  — L panel
- `F22_` (dupd × dupd, ld = dupd, work 영역)  — Schur (CB)

4 개의 `DenseMW` 가 각각 다른 leading dimension 을 가진다. 그리고 이 4 개를 다시 *level 전체에 걸쳐* 4 개의 포인터 배열로 모은다 (`FrontMAGMA.cpp:704-705`):

```cpp
// FrontMAGMA.cpp:704
auto F11 = L.dev_F_batch;   auto F12 = F11 + Nsmall;
auto F21 = F12 + Nsmall;    auto F22 = F21 + Nsmall;
```

MAGMA 의 `*_vbatched_*` API 는 *array of pointers + array of leading dimensions* 을 받는 형식이라 이 분리가 강제된다.

### 2.2 custom: 한 front = `fsz × fsz` 단일 column-major 덩어리

`src/plan/multifrontal_plan.cu:576-587` 에서 panel 별 offset 은 fsz² 의 prefix-sum:

```cpp
// multifrontal_plan.cu:576-587 — front_off computation
long total = 0;
for (int p = 0; p < P; ++p) {
    front_off[p] = total;
    const long fsz = front_ptr[p + 1] - front_ptr[p];
    total += fsz * fsz;
}
front_off[P] = total;
```

그리고 factor 커널은 이 단일 base 에서 직접 인덱싱 (`factorize/multifrontal.cu:49-56`):

```cpp
// factorize/multifrontal.cu:43-51
const int idx = lbegin + blockIdx.x;
if (idx >= lend) return;
const int p = plcols[idx];
const int s = front_ptr[p];
const int fsz = front_ptr[p + 1] - s;
const int nc = ncols[p];
FT* F = front + front_off[p];   // <-- 한 포인터로 모든 영역 접근
```

front 안의 모든 영역은 같은 base + 같은 stride (`fsz`) 로 표현된다:
- L panel: `F[i*fsz + k]` for `i ∈ [k+1, fsz)`, `k ∈ [0, nc)`
- U panel: `F[k*fsz + j]` for `k ∈ [0, nc)`, `j ∈ [nc, fsz)`
- Trailing/CB: `F[i*fsz + j]` for `i, j ∈ [nc, fsz)`

→ 별도의 `F11/F12/F21/F22` 객체가 없다. **분기는 인덱스에만 있고, 메모리에는 없다.**

### 2.3 무엇이 달라지나

| 항목 | STRUMPACK | custom |
|---|---|---|
| Front 1개당 별도 메모리 객체 | 4 (`F11,F12,F21,F22`) | 1 |
| Level N 개 front 디스패치 시 호스트→디바이스 메타데이터 | 4 개의 포인터 배열 + 4 개의 leading-dim 배열 + 2 개의 dim 배열 (`d1, d2`) | `plcols[]` 한 개 (정수), `front_off[]` 한 개 (한 번 빌드 후 고정) |
| 커널 내부 메모리 접근 | 4 개의 base pointer load 후 stride 4 종류 사용 | 1 개의 base pointer + 1 개의 stride |

STRUMPACK 이 이렇게 잘게 쪼개는 이유는 **MAGMA vbatched 가 그렇게 받기 때문**이다. MAGMA 는 일반 dense BLAS — getrf, trsm, gemm — 의 batch 버전이고, 각각이 독립적 행렬을 받도록 설계되어 있다. 그래서 한 front 의 L/U/Schur 가 *논리적으로는 같은 큰 행렬의 부분* 이라는 정보가 MAGMA API 경계에서 잘려나간다. 이게 다음 차이 ② 를 강제한다.

---

## 3. 차이 ② — Level 의 단위: 5–7 vbatched vs 1 융합 커널

### 3.1 STRUMPACK 의 한 level

`FrontMAGMA.cpp:700-755` 의 한 level 처리:

```cpp
// FrontMAGMA.cpp:707-711 — call 1/7: getrf vbatched (factor pivot block)
gpu::magma::getrf_vbatched_max_nocheck_work
  (d1, d1, L.max_d1_small, L.max_d1_small, L.max_d1_small,
   L.max_d1_small*L.max_d1_small, F11, ld1, L.dev_ipiv_batch,
   L.dev_getrf_err, L.dev_getrf_work, &L.getrf_work_bytes,
   Nsmall, handle);

// FrontMAGMA.cpp:712-716 — call 2/7: scalar getrf for big fronts (cuSOLVER, loop)
for (std::size_t i=Nsmall; i<N; i++)
  gpu::getrf(handle, L.f[i]->F11_, ..., L.f[i]->piv_, L.dev_getrf_err+i);

// FrontMAGMA.cpp:727-729 — call 3/7: laswp vbatched (row swap from pivoting)
gpu::laswp_fwd_vbatched
  (handle, d2, L.max_d2_small, F12, ld1, L.dev_ipiv_batch, d1, Nsmall);

// FrontMAGMA.cpp:730-733 — call 4/7: trsm L (lower) on F12
gpu::magma::trsm_vbatched_max_nocheck
  (MagmaLeft, MagmaLower, MagmaNoTrans, MagmaUnit,
   L.max_d1_small, L.max_d2_small, d1, d2, scalar_t(1.),
   F11, ld1, F12, ld1, Nsmall, handle);

// FrontMAGMA.cpp:734-737 — call 5/7: trsm U (upper) on F12
gpu::magma::trsm_vbatched_max_nocheck
  (MagmaLeft, MagmaUpper, MagmaNoTrans, MagmaNonUnit, ...);

// FrontMAGMA.cpp:738-741 — call 6/7: scalar getrs for big fronts
for (std::size_t i=Nsmall; i<N; i++)
  gpu::getrs(handle, Trans::N, L.f[i]->F11_, L.f[i]->piv_, L.f[i]->F12_, ...);

// FrontMAGMA.cpp:750-754 — call 7/7: gemm vbatched (rank-d1 trailing update F22 -= F21 * F12)
gpu::magma::gemm_vbatched_max_nocheck
  (MagmaNoTrans, MagmaNoTrans, d2, d2, d1, scalar_t(-1.),
   F21, ld2, F12, ld1, scalar_t(1.), F22, ld2, Nsmall,
   L.max_d2_small, L.max_d2_small, L.max_d1_small, handle);
```

그리고 **extend-add 는 또 별도 커널** (`FrontCUDA.cu:114-148`):

```cpp
// FrontCUDA.cu:114-148 — separate kernel pass after factor
template<typename T, unsigned int unroll>
__global__ void extend_add_kernel
(unsigned int by0, unsigned int nf, AssembleData<T>* dat, bool left) {
  ...
  auto CB = left ? f.CB1 : f.CB2;
  auto I  = left ? f.I1  : f.I2;
  ...
  for (int i=0; i<unroll; i++) {
    int x = x0 + i;
    auto Ix = I[x];
    F[Ix >= d1][Ix*ld] += CB[i*dCB];  // scatter
  }
}
```

→ 한 level 에 **7+ 커널 launch**:
1. `getrf_vbatched` (small fronts)
2. cuSOLVER getrf loop (big fronts, N개 launch — 여기가 추가)
3. `laswp_fwd_vbatched`
4. `trsm_vbatched` (Lower)
5. `trsm_vbatched` (Upper)
6. cuSOLVER getrs loop (big fronts)
7. `gemm_vbatched`
8. `extend_add_kernel` (multi-grid pass, left + right children → 2 launches)

case8387pegase 의 nsys (`cuda_gpu_kern_sum`) 에서 STRUMPACK 한 NR iter 가 ~700 개 커널 launch 인 것이 이 구조의 직접적 결과다.

### 3.2 custom 의 한 level

`factorize/multifrontal.cu:34-122` 의 단일 커널 `mf_factor_extend_level` 하나가 한 level 의 **모든 front, 모든 단계 (panel LU + trailing + extend-add)** 를 처리한다:

```cpp
// factorize/multifrontal.cu:34-44
__global__ void mf_factor_extend_level(int lbegin, int lend,
    const int* plcols, const int* front_off, const int* front_ptr,
    const int* ncols, const int* panel_parent, const int* asm_ptr,
    const int* asm_local, FT* front, int* sing, int do_extend)
{
    const int idx = lbegin + blockIdx.x;        // panel id in level
    if (idx >= lend) return;
    const int p = plcols[idx];
    ...
    FT* F = front + front_off[p];
    ...
```

이 커널 안에서:
- `fsz <= 48`: 단순 rank-1 루프 (lines 53-70)
- `fsz > 48`: blocked rank-nc — Phase 1 panel LU, Phase 2 U panel TRSM, Phase 3 trailing GEMM (lines 71-106)
- **`__syncthreads()` 후 extend-add** (lines 108-120):

```cpp
// factorize/multifrontal.cu:108-120 — fused extend-add (atomicAdd to parent)
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

부모는 더 높은 level 에 있으므로 *현재 launch 에서 동시에 쓰는 다른 block 없음* → race-free. 커널 헤더 주석이 이걸 명시한다 (`factorize/multifrontal.cu:30-33`):

```cpp
// higher level (not factored this launch), so the atomicAdd is race-free. Halves the
// graph nodes and removes one inter-kernel sync per level (cuts the level
// serialization on the deep power-grid etrees, ~72 levels on SyntheticUSA).
```

→ 한 level 에 **1 개 커널 launch**. case8387 의 31 levels × 2 iter = 62 launch 가 `cuda_gpu_kern_sum` 의 `mf_factor_extend_level` instances=60 (60 = level 중 root 제외) 으로 정확히 일치한다.

### 3.3 호출 횟수 비교 (한 NR iter 기준)

| 단계 | STRUMPACK | custom |
|---|---:|---:|
| Factor 본체 | ~700 (level × kernel-type) | ~31 (1 per level) |
| Solve forward | ~50 | 31 |
| Solve backward | ~50 | 31 |
| 합계 | ~800 | ~95 |

이 차이가 nsys 에서 본 *"strumpack 의 hot kernel 평균 SM% 0.1~25%, custom 도 비슷한 SM% 인데 wall-time 은 14×"* 의 직접 원인이다. **둘 다 latency-bound 영역에 있지만, custom 은 launch 횟수가 약 1/8.**

---

## 4. 차이 ③ — Extend-add 의 융합 (custom) vs 분리 (STRUMPACK)

### 4.1 STRUMPACK: 별도 커널 + 인덱스 배열 + grid sweep

`FrontCUDA.cu:114-148` 가 위의 `extend_add_kernel`. 이 커널이 동작하려면 *호스트가 미리* `AssembleData<T>` 구조체 안에 `CB1, CB2, I1, I2, F[0], F[1]` 포인터를 채워줘야 한다 (`FrontGPU.cpp:301-345`):

```cpp
// FrontGPU.cpp:309-315 — host-side assembly setup
hasmbl[n].set_ext_add_left(c->dim_upd(), c->F22_.data(), fdIptr);
c->upd_to_parent(&f, fIptr);
fdIptr += c->dim_upd();
hasmbl[n].F[0] = f.F11_.data();
hasmbl[n].F[1] = f.F12_.data();
hasmbl[n].F[2] = f.F21_.data();
hasmbl[n].F[3] = f.F22_.data();
```

이걸 디바이스로 복사하고 (`gpu::assemble(N, hasmbl, dasmbl)`), 별도 stream 에서 launch. CB 가 *child 의 F22 영역* 에 살아남아 있어야 부모 level 의 extend-add 가 읽어갈 수 있다 — **CB 가 work 메모리에서 차지하는 영역이 child level 끝나도 풀리지 않음**.

### 4.2 custom: 같은 커널, atomicAdd, scatter 인덱스는 `asm_local[]` 으로 사전 baked

custom 은 *symbolic 단계* 에서 `asm_ptr[]` 와 `asm_local[]` 를 빌드한다 (`plan/multifrontal_plan.cu:640-646`). `asm_local[abase + a]` 는 child 의 a 번째 CB 행이 *부모의 front 안에서 몇 번 인덱스인지* 를 미리 계산해둔 것 (parent-local index — 부모 front_rows[] 의 시작점 이미 뺀 상태).

→ extend-add 시 커널이 하는 일은:
```cpp
atomicAdd(&Fp[asm_local[abase + a] * pfsz + asm_local[abase + b]],
          F[(nc + a) * fsz + (nc + b)]);
```
- binary search 없음
- 자식 CB 메모리 해제 가능 (extend-add 가 끝났으니까)
- 부모와 같은 column-major arena 에 직접 누적

**CB 라는 별도 객체가 존재하지 않는다.** child front 의 trailing 영역이 곧 CB 고, atomicAdd 한 후에는 *그 메모리를 다시 쓸 일이 없음*.

### 4.3 메모리 효과

STRUMPACK 은 한 level 처리 동안 *parent CB (F22) 메모리를 계속 들고 있다가 부모 level 에서 extend-add 패스로 읽어간다*. peak 메모리 = 부모 chain 전체의 F22 합. 반면 custom 은 한 level 처리 (factor + extend-add 융합) 후 자식 CB 영역이 *논리적으로* 더 안 쓰임 (parent 가 자기 영역에 누적했으니까). 단, 둘 다 etree 전체의 front 메모리는 보유한다 — 차이는 *피크 메모리 패턴* 이지 *총량* 은 아니다.

---

## 5. 차이 ④ — Pivoting: partial pivoting + `laswp` overhead vs no-pivot

### 5.1 STRUMPACK: in-kernel pivot search + `laswp_vbatched`

`FrontCUDA.cu:234-294` 의 `LU_block_kernel` 내부에서 small fronts (NT ≤ 32) 에 대해 partial pivoting 을 수행:

```cpp
// FrontCUDA.cu:248-263 — pivot search + row swap, per pivot column
for (int k=0; k<n; k++) {
    if (j == k && i >= k) cabs[i] = absolute_value(M[i+j*NT]);
    __syncthreads();
    if (j == k && i == k) {
        p = k; Mmax = cabs[k];
        for (int l=k+1; l<n; l++)
            if (cabs[l] > Mmax) { Mmax = cabs[l]; p = l; }
        piv[k] = p + 1;
    }
    __syncthreads();
    if (j < n && i == k && p != k) {
        auto tmp = M[k+j*NT];
        M[k+j*NT] = M[p+j*NT];
        M[p+j*NT] = tmp;
    }
    ...
}
```

big fronts 는 cuSOLVER getrf 가 같은 partial pivoting 을 수행. 그리고 `F12` (U panel) 도 같은 행 교환을 적용받아야 하므로 **별도 커널** `laswp_fwd_vbatched` (`FrontMAGMA.cpp:727-729`) 를 launch.

ncu 측정 (`/tmp/bench/ncu_strumpack_8387.csv`) 에서 이 `laswp_vbatch_kernel` 의 SM% = 0.1%, DRAM% = 0.2% — **순수 overhead**. 47 instances/iter (각 level 당 한 번).

### 5.2 custom: no-pivot, `laswp` 없음

`factorize/multifrontal.cu:54-58` 와 `:75-79`:

```cpp
// factorize/multifrontal.cu:54-58 — small front pivot
for (int k = 0; k < nc; ++k) {
    FT piv = F[(long)k * fsz + k];
    if (piv == FT(0)) {
        if (t == 0) *sing = 1;
        piv = FT(1);
    }
    for (int i = k + 1 + t; i < fsz; i += nt) F[(long)i * fsz + k] /= piv;
```

→ pivot 검색 없음, row swap 없음, `piv[]` 배열 없음, `laswp` 커널 없음. 정확성은 power-flow Jacobian 의 *주대각 우세 + diagonal-perturbation safe* 특성에 의존 (`docs/02-design-analysis/03-no-pivoting-empirical-proof.md` 참고). 실패 시 `*sing = 1` 로 flag.

### 5.3 무엇이 절감되나

| 항목 | STRUMPACK 의 cost (case8387 ncu) |
|---|---|
| `laswp_vbatch_kernel` × 47 launches | SM 0.1%, DRAM 0.2% → wall-time 점유 작지만 누적 launch overhead |
| in-kernel pivot search (LU_block_kernel) | 각 k 마다 reduction (max search) + 2× `__syncthreads()` + 잠재적 row swap |
| `piv[]` 배열 maintenance | 메모리 + 호스트 메타데이터 |
| `dev_ipiv_batch` 디바이스 배열 | 매 level 별 메모리 + I/O |

custom 은 이걸 *전부 0* 으로 만든다. pivoting 자체가 *power-grid Jacobian 의 조건수 특성으로 인해 안전* 하다는 *알고리즘적 결정* 이 이 모든 비용을 제거한다.

---

## 6. 차이 ⑤ — Solve 경로: CPU 폴백 vs GPU 레지던트 + level batching

### 6.1 STRUMPACK: 기본 경로는 *factor → D2H 복사 → CPU solve*

`FrontMAGMA.cpp:745-748` 에서 factor 후:

```cpp
// FrontMAGMA.cpp:745-748 — factors copied back to host after each level
comp_stream.synchronize();
gpu::copy_device_to_host_async<scalar_t>
  (pinned, dev_factors, L.factor_size, copy_stream);
```

그리고 `FrontGPU.cpp:698-712` 의 forward solve:

```cpp
// FrontGPU.cpp:698-712 — per-front CPU work
void fwd_solve_phase2(DenseM_t& b, DenseM_t& bupd,
                      int etree_level, int task_depth) const {
  if (dim_sep()) {
    DenseMW_t bloc(dim_sep(), b.cols(), b, this->sep_begin_, 0);
    F11_.solve_LU_in_place(bloc, piv_, task_depth);   // <-- CPU LAPACK
    if (dim_upd()) {
      if (b.cols() == 1)
        gemv(Trans::N, scalar_t(-1.), F21_, bloc, scalar_t(1.), bupd, task_depth);
      else
        gemm(Trans::N, Trans::N, scalar_t(-1.), F21_, bloc, scalar_t(1.), bupd, task_depth);
    }
  }
}
```

→ "Solve is performed on CPU" 가 STRUMPACK 로그에서 뜨는 이유. nsys 의 `extract_rhs_kernel`, `extend_add_rhs_kernel`, `gemvn_kernel_vbatched` 가 SM 0.7-1.0% 인 것도 *GPU 위에 호스트의 호스트-디바이스 동기화 와 잘게 흩어진 work 만 남아서*.

### 6.2 custom: GPU 레지던트 + level batching + 옵션 selinv

`solve/multifrontal.cu:15-57` 의 `mf_fwd_level` 이 한 level 모든 front 를 한 커널로 처리:

```cpp
// solve/multifrontal.cu:32-49 — selinv=true (parallel gemv) vs sequential trsv
if (selinv) {
    for (int k = t; k < nc; k += nt) {
        YT v = y[fr[k]];
        for (int i = 0; i < k; ++i) v += F[(long)k * fsz + i] * y[fr[i]];
        sh_piv[k] = v;
    }
    __syncthreads();
    for (int k = t; k < nc; k += nt) y[fr[k]] = sh_piv[k];
} else {
    if (t == 0) {
        for (int k = 0; k < nc; ++k) {
            YT v = y[fr[k]];
            for (int i = 0; i < k; ++i) v -= F[(long)k * fsz + i] * sh_piv[i];
            sh_piv[k] = v;
            y[fr[k]] = v;
        }
    }
    __syncthreads();
}
```

그리고 trailing update (lines 52-56):
```cpp
for (int i = nc + t; i < fsz; i += nt) {
    YT upd = YT(0);
    for (int k = 0; k < nc; ++k) upd += F[(long)i * fsz + k] * sh_piv[k];
    atomicAdd(&y[fr[i]], -upd);
}
```

`selinv` (= `mf_invert_pivot` 으로 pivot 블록의 inverse 를 사전 계산) 가 켜져 있으면 sequential trsv 가 parallel gemv 가 된다. 백워드도 마찬가지로 level-batched `mf_bwd_level` 한 개.

### 6.3 Solve 의 자릿수 차이가 여기서 나온다

case8387 nsys:
- STRUMPACK iter2 solve: 11.24 ms
- custom (graph ON) iter2 solve: 0.34 ms (33×)
- custom (graph OFF) iter2 solve: 0.53 ms (21×)

차이의 직접 원인:
- STRUMPACK 은 *factor 끝마다 D2H 복사 + 호스트의 per-front LAPACK* 으로 도는 동안 GPU 가 거의 idle (ncu 의 `extract_rhs_kernel` SM 0.7-1.0% 가 그 증거)
- custom 은 GPU 레지던트 + level batching + selinv 까지 결합 → 31 levels × 2 kernels (fwd + bwd) = 62 launch 안에 모든 일을 끝냄

여기서도 graph 효과는 secondary: graph OFF 에서도 21× 빠르다. 본질은 *데이터를 GPU 에 둔 채 level batching* 한 것.

---

## 7. Amalgamation — `nc`, `cap`, panel 의 의미와 그 효과 (cap sweep 실측)

이 절은 앞 § 들에서 자주 등장한 `nc`, `cap`, panel 의 정의를 묶고, 그 결정이 wall-time 에 실제로 어떤 영향을 주는지 — `CLS_CAP` env 로 cap 만 바꿔가며 같은 매트릭스에서 측정한 결과로 — 확인한다.

### 7.1 용어

**Front size `fsz`** — 한 front 의 행 수 = pivot 컬럼 수 (`nc`) + Schur (CB) 보조 행 수. front 의 dense block 크기. 메모리 `fsz²`, factor 작업 `~fsz³` (rank-nc trailing update). § 2.3 의 column-major arena 인덱싱 단위.

**Pivot column count `nc`** — 한 front 에서 *동시에* 제거되는 컬럼 수 = 그 panel 에 amalgamate 된 컬럼 수.
- `nc = 1` (amalgamation 없음): 한 etree 노드 = 한 panel. panel 수 = n.
- `nc = k`: chain 으로 인접한 k 개 컬럼이 한 dense block 으로 묶임. 그 block 의 LU 는 한 panel 의 작업으로 처리.

큰 nc 의 장점: dense LU 의 작업당 효율 ↑, panel 수와 등치된 launch 수 ↓. 단점: 묶인 컬럼들이 nest 하지 않으면 빈 슬롯도 0 으로 저장해야 함 (**padded fill 증가**).

**Supernode (Liu-Ng-Peyton fundamental supernode)** — 컬럼 j 와 직전 postorder 컬럼 jprev 의 below-diagonal pattern 이 *정확히 nest* (`colcount[jprev] == colcount[j] + 1`) 하고 *유일한 자식 관계* 일 때 둘을 같은 supernode 로 묶음 (`symbolic/supernode.cpp:23-40`). **fill 증가 없음** — 안전한 정확 병합. power-grid Jacobian 에서 fundamental supernode 평균 폭은 약 2 컬럼 — dense GPU 커널 단위로는 너무 작음.

**Panel (relaxed amalgamation)** — fundamental supernode 가 작아서 추가로 *etree chain* (postorder 에서 `parent[j] == j+1` 인 단일-자식 경로) 을 따라 greedy 병합, cap 컬럼까지 (`symbolic/supernode.cpp:63-82`). 묶인 컬럼들의 colcount 가 다르면 dense block 은 *가장 넓은 colcount 로 padded*. *작은* fill 증가가 발생.

> 안전성 주석 (`supernode.cpp:56-58`): *"child's contribution block must nest in one parent front — so relaxed_panels itself only does the safe chain merge below"*. 임의 chain merge (cross-subtree) 는 extend-add 의 nest 가정 (§ 4) 을 깨므로 deep-K amalgamation 은 별도 `MF_AMALG` 경로로 격리. 기본 경로는 *etree chain 만* 병합.

**`panel_cap` / `CLS_CAP`** — relaxed panel amalgamation 의 panel 당 *최대 컬럼 수* = nc 의 상한. 매트릭스 크기에 따라 적응적으로 결정 (`factorize/multifrontal.cu:561`):

```cpp
int eff_cap = n >= 80000 ? 20 : (n >= 16000 ? 12 : panel_cap);   // panel_cap = 8 default
if (const char* ce = std::getenv("CLS_CAP")) eff_cap = std::atoi(ce);  // research override
```

cap 의 trade-off:
- ↑: panel 수 P ↓, panel-etree level 수 ↓, 각 panel dense work ↑ → **launch 수 ↓, solve loop 짧아짐**.
- ↑: padded fill ↑, nc 가 `MF_REG_NC = 16` (작은-front 의 register-partial 경로 상한, `multifrontal.cu:412`) 을 넘으면 fast path 깨짐 → **factor 회귀**.

### 7.2 cap sweep 실측 (graph OFF 빌드, RTX 3090, iter2)

**case8387pegase (n=14908, 기본 cap=8)**

| `CLS_CAP` | P (panel 수) | levels | front 메모리 (MB f32) | factor [ms] | solve [ms] |
|---:|---:|---:|---:|---:|---:|
| 1 (amalg off) | 14908 | 171 | 6.1 | 1.31 | 1.55 |
| 4  | 7638 | 49 | 2.4 | 0.72 | 0.59 |
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

1. **Amalgamation 본체 효과 = level 수 감소.** case8387 에서 cap=1 → cap=8: levels 171 → 30 (**5.7×**). solve 가 level-batched launch loop (§ 6) 라 *level 수 ≈ launch 수* → solve 1.55 → 0.49 ms (**−68%**) 의 직접 원인.

2. **Factor 도 동반 개선.** case8387 factor 1.31 → 0.72 ms (**−45%**). 원인은 panel 수 ↓ (14908 → 7418) 로 인한 launch 수 감소 + dense LU 작업당 효율 소폭 상승.

3. **Pareto-crossover 존재.** cap=8 → cap=16 (case8387) 에서 factor 는 9% 악화 / solve 는 8% 개선. cap 은 결국 *factor ↔ solve 트레이드* — 매트릭스마다 sweet spot 이 다른 게 **n-adaptive cap (8/12/20) 의 정당성**. ACTIVSg25k 에서도 cap=12 가 factor sweet spot, cap=8 이 solve sweet spot.

4. **상한선의 실제 위치는 cap ≈ 16.** cap=20 에서 두 case 모두 factor 폭증 (34/45 ms). 원인은 nc > 16 이 `MF_REG_NC=16` register-partial fast path 를 깨는 것. 코드의 `eff_cap > 64` clamp 보다 실제 안전선이 훨씬 낮음. n ≥ 80k 의 큰 매트릭스는 큰 fronts 가 dominant 라서 cap=20 도 안전 — 그래서 *n ≥ 80k 에서만* cap=20 을 허용.

5. **Padded fill 비용은 작다.** case8387 front 메모리 6.1 MB (cap=1) → 1.9 MB (cap=16) — *더 줄어든다*. amalgamation 자체 (P 감소) 가 padded fill 증가보다 큰 효과. power-grid Jacobian 의 *etree chain 이 길고 colcount 가 거의 일정한* 특성이 padding cost 를 작게 만듦.

### 7.4 STRUMPACK 과의 대비

STRUMPACK 도 supernode amalgamation 을 한다 (Ashcraft 1990s 의 표준 절차). 그러나 차이는:

- STRUMPACK 의 amalgamated supernode 는 결국 *MAGMA vbatched 의 입력* 으로 가서 **다시 4 조각 (F11/F12/F21/F22) 으로 쪼개진다** (§ 2). amalgamation 으로 만든 dense block 의 "한 덩어리" 라는 정보가 API 경계에서 잘려나감.
- 따라서 STRUMPACK 에서 amalgamation 효과는 *각 vbatched 호출의 평균 dense work 증가* 로만 전달 — **한 level 당 호출 수는 supernode 수가 아니라 *연산 종류* (~7) 로 고정** → launch 수는 amalgamation 으로 줄어들지 않음.
- custom 에서는 amalgamation 으로 만든 block 이 *그대로 한 커널 한 thread block 의 작업 단위* → **panel 수 감소가 곧 launch 수 감소** 로 전달. 같은 amalgamation 결정이 두 솔버에서 다른 강도의 효과를 낸다.

즉, amalgamation 은 알고리즘적으로 같은 결정이지만, *§ 3 의 level 단위 차이* 가 그 결정의 효과 배율을 다르게 만든다. STRUMPACK 에서 amalgamation 으로 (예) 1.5× 개선이 나오면, custom 에서는 같은 변화가 *panel 수 + level 수* 양쪽에 곱해져 더 큰 개선으로 전달된다.

---

## 8. 양적 매핑 — nsys 측정과 위 차이의 연결

이전 측정 (`/tmp/bench/nsys/`) 의 case8387pegase, iter2 기준:

| 단계 | STRUMPACK [ms] | custom (graph) | custom (no graph) |
|---|---:|---:|---:|
| factor | 14.32 | 0.643 | 0.699 |
| solve  | 11.24 | 0.341 | 0.530 |

각 차이를 위 § 와 매핑:

### Factor 14.32 ms → 0.699 ms (graph off, 20× 빠름)

- **§3 level 당 ~7+ 커널 vs 1 커널** — 호출 횟수가 가장 큰 단일 원인. 31 levels × ~7 = ~217 launches (factor만) vs 31 launches. STRUMPACK 의 `gemm_template_vbatched_nn_kernel<16,16,48>` instances=146, `trsm_template_vbatched_lN[L,U]_kernel<16,64>` 152 등 합산.
- **§5 partial pivoting overhead** — `laswp_vbatch_kernel` 47 instances/iter, in-kernel pivot search 추가 동기화.
- **§2 4× 디스패치 메타데이터** — MAGMA 가 호스트에서 빌드해야 하는 포인터/leading-dim 배열. host-side overhead 가 launch latency 와 함께 누적.
- **§3 small-front 의 MAGMA tile mismatch** — `gemm_template_vbatched<16,16,48>` 의 sweet spot 은 fsz ≥ 64. case8387 의 max fsz = 96, 96.2% 가 fsz ≤ 16. 호출 한 번에 idle tile 비중 압도. (이건 `docs/04-benchmarks-profiling/05-strumpack-vs-custom-multifrontal-case8387.md` § 2-3 에서 정량화됨.)

### Solve 11.24 ms → 0.530 ms (graph off, 21× 빠름)

- **§6 CPU 폴백** — STRUMPACK 이 forward/backward 를 CPU 에서 돌리는 게 가장 큰 단일 비중. ncu 의 SM 0.7-1.0% 가 *"GPU 거의 안 씀"* 의 직접 증거.
- **§6 level batching 없음** — custom 의 `mf_fwd_level/mf_bwd_level` 가 한 launch 로 처리하는 일이 STRUMPACK 에서는 per-front trsv/gemv (CPU 또는 vbatched).
- **§6 selinv** — pivot 블록을 inverse 로 두면 trsv 가 gemv 가 되어 thread-parallel.

### Graph (ON vs OFF) — graph 효과의 위치

| 단계 | graph ON | graph OFF | graph 효과 |
|---|---:|---:|---:|
| factor | 0.643 | 0.699 | +9% (0.056 ms 절감) |
| solve  | 0.341 | 0.530 | +55% (0.189 ms 절감) |

graph 가 잡아주는 건 *kernel launch overhead*. factor 커널 31 개 × ~2 μs ≈ 60 μs 가 graph 로 1 회 launch + replay 가 되면서 절감. solve 도 같은 원리지만 *커널 한 개의 work 가 더 작아서 launch overhead 의 비중이 더 큼* → 그래서 solve 의 graph 효과 (55%) 가 factor (9%) 보다 큼.

**Graph 가 절감한 절대값 (factor 0.056 ms + solve 0.189 ms = 0.245 ms) vs 알고리즘이 절감한 절대값 (factor 13.6 ms + solve 10.7 ms = 24.3 ms)** — **약 100× 차이**. 사용자 진술 ("graph 보다 알고리즘이 더 영향이 큰 게 아닌가") 가 정확히 그 비율로 확인된다.

---

## 9. 정리 — "multifrontal 레이아웃 + level-batched 커널" 의 정확한 의미

custom 의 "*multifrontal 레이아웃*" 은:
1. 한 front = `fsz × fsz` 한 덩어리 column-major (별도 F11/F12/F21/F22 분리 없음, § 2)
2. 모든 front 가 단일 device arena 에 prefix-sum 오프셋으로 배치
3. extend-add 의 scatter 인덱스 (`asm_local[]`) 가 symbolic 단계에서 *부모-로컬 인덱스* 로 미리 변환되어 저장 (§ 4)

custom 의 "*level-batched 커널*" 은:
1. 한 level 의 모든 front + 모든 단계 (panel LU + trailing + extend-add) 를 단일 `mf_factor_extend_level` 커널이 처리 (§ 3)
2. block id 한 정수로 panel id 도출 → 별도의 포인터 배열 디스패치 없음
3. partial pivoting 제거 (§ 5) 로 그 단계에 필요했을 추가 커널 (`laswp`) + 동기화 + 메타데이터 모두 제거
4. solve 도 같은 level batching 패턴 (§ 6)
5. relaxed amalgamation (§ 7) 으로 panel 폭을 키워 *panel 수 + level 수* 를 동시에 줄임 — STRUMPACK 도 같은 amalgamation 을 하지만 vbatched API 가 그 효과를 launch 수 절감으로 받지 못함

STRUMPACK 은 같은 multifrontal 알고리즘이지만, **MAGMA vbatched 의 일반 dense BLAS API 에 맞춰 front 를 4 조각으로 쪼개고 한 level 을 7+ 개 vbatched 호출로 분해**한다. 이 API 일치는 *일반 dense LU* 에서는 합리적이지만, *fsz ≤ 96 의 극단적 small front 영역* (power-grid Jacobian) 에서는:
- vbatched 한 번의 work 가 launch overhead 와 비슷한 수준
- 4 조각으로 분해된 메타데이터가 host 부담을 키움
- partial pivoting 의 정확성 이득 대비 매 level 당 `laswp` overhead 가 큼
- solve 가 CPU 로 fallback 되며 host-device sync 폭증

따라서 같은 latency-bound 영역에서 도는데도 wall-time 이 자릿수 차이가 난다. **이건 graph 가 만들어낸 차이가 아니라, custom 이 *small-front 영역에 특화된 자료구조 + 커널 단위 단일화* 를 적극적으로 선택한 결과**다. Graph 는 그 위에 얹어진 *추가* 10-50% 최적화일 뿐, 자릿수 자체는 알고리즘 결정에서 나온다.

---

## 부록 A — 본 문서에서 인용한 소스 좌표

### custom_linear_solver
- `src/factorize/multifrontal.cu:30-122` — `mf_factor_extend_level` 커널 (factor + extend-add 융합)
- `src/factorize/multifrontal.cu:412` — `MF_REG_NC = 16` (작은-front register-partial 경로의 nc 상한)
- `src/factorize/multifrontal.cu:423-457` — `mf_invert_pivot` (selinv 용 pivot block inverse)
- `src/factorize/multifrontal.cu:561-564` — `eff_cap` n-적응 결정 (cap = 8/12/20, `CLS_CAP` env override)
- `src/symbolic/supernode.cpp:5-43` — fundamental supernode (Liu-Ng-Peyton)
- `src/symbolic/supernode.cpp:45-84` — `relaxed_panels` (etree chain greedy merge, cap 까지)
- `src/solve/multifrontal.cu:15-57` — `mf_fwd_level`
- `src/solve/multifrontal.cu:69-151` — `mf_bwd_level`
- `src/plan/multifrontal_plan.cu:576-587` — front_off (fsz² prefix-sum) 빌드
- `src/plan/multifrontal_plan.cu:640-646` — `asm_local[]` (parent-local extend-add 인덱스) 빌드
- `src/plan/multifrontal_plan.cu:593-598` — panel-etree level 계산

### STRUMPACK
- `src/sparse/fronts/FrontGPU.cpp:155-200` — `set_factor_pointers / set_work_pointers` (F11/F12/F21/F22 4 분리)
- `src/sparse/fronts/FrontMAGMA.cpp:700-755` — 한 level 의 7+ vbatched 호출 시퀀스
- `src/sparse/fronts/FrontCUDA.cu:114-148` — `extend_add_kernel` (별도 패스 scatter)
- `src/sparse/fronts/FrontCUDA.cu:234-294` — `LU_block_kernel` (in-kernel partial pivoting)
- `src/sparse/fronts/FrontGPU.cpp:698-712` — `fwd_solve_phase2` (CPU 폴백 solve)
- `src/sparse/fronts/Front.hpp:312-324` — `get_level_fronts_gpu` (level 별 front 수집)

### nsys / ncu 측정
- `/tmp/bench/nsys/custom_nr_8387.nsys-rep` — custom (graph ON)
- `/tmp/bench/nsys/custom_nr_nograph_8387.nsys-rep` — custom (graph OFF, 본 세션 측정)
- `/tmp/bench/nsys/strumpack_nr_8387.nsys-rep` — STRUMPACK MAGMA
- `/tmp/bench/ncu_custom_8387.csv` / `ncu_strumpack_8387.csv` — kernel-bound 분류

### 관련 본 저장소 문서
- `docs/04-benchmarks-profiling/05-strumpack-vs-custom-multifrontal-case8387.md` — front-size 분포 + ncu bound 분류 (본 문서의 정량 보강)
- `docs/01-orientation/03-lineage-strumpack-not-the-baseline.md` — 알고리즘 패밀리 동일성
- `docs/02-design-analysis/02-acceleration-mechanism-ranked.md` — 가속 요인 우선순위 (본 문서의 결론과 같은 방향)
- `docs/02-design-analysis/03-no-pivoting-empirical-proof.md` — § 5 의 no-pivot 안전성 근거
- `docs/03-optimization-notes/04-tensor-core-factor-design.md` — 추가 (FP32 mixed-precision 변형)
