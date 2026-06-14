# 문헌조사 — B=1 GPU multifrontal factorization 의 under-occupancy

> **범위**: GPU multifrontal/supernodal LU/Cholesky, 병목이 소거트리 **깊은/root 근처** 레벨의 under-occupancy(작은 dense front 소수, 1 block/SM). power-flow Jacobian, 대칭 패턴, ordering 은 Newton 반복에 amortize.
> **요지**: 주류 GPU sparse-direct 연구는 **3D-PDE 의 *큰* root front**(GPU 친화)를 가정. **단일 행렬 + 작은 root + B=1** 은 인지되었으나 **literature 의 빈 구멍**. 전이 가능한 두 아이디어만 실재: (1) **MAGMA native-mode** 단일행렬 on-GPU 패널 인수분해, (2) **critical-path/tree-height-aware ordering**. 나머지는 batch 병렬을 요구하거나 다른 문제를 푼다. → 본 실험의 결론(ordering 이 유일 레버, scheduling/tiling/amalgamation 무효)과 정확히 일치.

## Q1. critical-path / 병렬성-aware fill-reducing ordering ✅ (본 실험의 핵심)

- **tree height = critical path = 병렬성 1차 예측자, 그러나 최소화는 NP-hard**(treedepth). (Pothen; Bodlaender et al., BIT)
- **가장 on-target**: Kayaaslan & Uçar, *"Reducing elimination tree height for parallel LU,"* **HiPC 2014** (hal-01114413). 재귀 strong-vertex separator 로 BBT 형태 → **tree-height ~28% 감소** vs METIS-ND. *단 분산/공유메모리 LU 측정, GPU occupancy 정량화 아님.*
- **예측 metric(전부 pre-numeric)**: ① tree height(=critical path), ② **top-separator size**(=최대 front + fill 상한 `fill ≤ sep²`; Gilbert), ③ **heaviest root-leaf path 의 front-FLOP 합**(node 수보다 정확 — front 비용 superlinear), ④ front-size 분포. (Karsavuran/Ng/Peyton arXiv:2409.14009: ND 가 AMD 보다 얕고 균형↑ GPU-친화)
- **"best-of-k orderings, 최저 비용 선택"은 표준이나 *잘못된* 목적**: MUMPS/CHOLMOD 는 여러 ordering 시도 후 **fill/sequential-FLOP 으로 선택** — critical-path/batchability 아님. **critical-path-aware 선택은 진짜 빈틈.** (Davis survey; Bollhöfer et al. arXiv:1907.05309)

> **본 실험 연결**: `tail_cube = Σ_{under-filled large level} maxfsz³` 는 위 ③(serial tail 의 front-FLOP 합)의 직접 구현. production 의 fill-기반 선택과 달리 **occupancy/critical-path 기반** → 문헌이 지목한 gap 을 메움. 회의점도 확인됨: min-height 단독은 작은 front 를 양산해 occupancy 를 *악화*시킬 수 있음 → 목적은 "front 가 batchable 한 한도 내 critical-path-FLOP 최소".

## Q2. deep/root dense front 의 GPU occupancy

- **가장 전이 가능 — MAGMA "native mode"**: Abdelfattah/Haidar/Tomov/Dongarra, *J. Comput. Sci. 2016.* 단일 중소 행렬을 **CPU 왕복 없이** 패널-in-shared 융합 커널로 인수분해, "각 launch 가 정확히 필요한 자원만 — idle thread 없음". **crossover ~400: 그 아래는 전체를 단일 blocked 커널**. 행렬<400 에서 **1.12–4×**. *260×260 root 에 정확히 해당.* ← 단, **현 솔버의 mid 커널(shared-resident blocked)·big 커널(1 block/front)이 이미 이 형태** → 추가 이득 제한.
- **주류 multifrontal GPU 솔버는 이 문제가 없음**(root 가 큰 3D-PDE separator): STRUMPACK(작은 front→custom+MAGMA batched, 큰 front→cuBLAS; **subtree batching 으로 병렬 제조** — batch 필요), CHOLMOD-GPU("root 쪽 supernode 는 GPU 이용률 높음" — 정반대), **Karsavuran 2024(주의)**: GPU-only Cholesky **저조**, 작은 supernode 는 **CPU 가 빠름**.
- **subtree-to-subcube/proportional mapping**(SuperLU_DIST 3D; PaStiX): root 가 병렬성-기아 영역임을 확인하나 **process/core 할당**(분산/멀티-GPU) — 단일 GPU thread-block 아님.
- **Ozaki/mixed-precision**: root 의 dense GEMM 가속이나 **260×260 은 split overhead amortize 불가** — occupancy 와 무관.

> **결론**: ~260 root 의 근거 있는 선택지는 (a) **CPU fallback**(Karsavuran/cuDSS hybrid), (b) **MAGMA-native 단일커널**(현 솔버가 이미 근사). naive cuBLAS/cuSOLVER GETRF 가 최악. **B=1 전용 root 전략은 미발표 — open area.**

## Q3. level 간 look-ahead / DAG pipelining ◑

- "parent front 를 children 의 *꼬리*와 overlap" = level 간 look-ahead. SuperLU_DIST look-ahead(Li et al. TOMS 2003, `num_lookaheads`), dense LU look-ahead(Catalán et al. arXiv:1804.07017: **이득이 가용 병렬성에 *반*비례** → under-occupied root 에서 최대지만 절대값은 children 잔여 일감에 bound).
- task-runtime(StarPU/PaRSEC) on PaStiX/qr_mumps: inter-front(tree)+intra-front(node) 병렬 결합 — 단 이득은 **큰 front/멀티-GPU**(고병렬 regime).
- **회의(Li & Liu survey arXiv:2602.14289, 2026)**: single-RHS SpTRSV 류는 O(1) arithmetic intensity, "task 병렬 제한적", **generic runtime 이 cost-effective 한지조차 open**.

> **본 실험 연결**: multifrontal 의 parent assembly 는 **모든 child front 완료(extend-add) 후**에야 시작 → inter-level look-ahead 가 깔끔히 적용 안 됨(이전 노트의 fusion 중립과 일치). intra-front pipelining 은 이미 mid/big 단일커널이 처리. → look-ahead 는 B=1 occupancy 를 단독 구제 못 함.

## Q4. amalgamation / relaxed supernode ✗

- Ashcraft & Grimes(TOMS 1989): 인접 supernode 병합 → BLAS-3 화, **단 3–20% (cache-locality 이득, occupancy 아님), *바닥*의 작은 front 대상.**
- 모든 production 코드는 **leaf 에서만 공격적, root 에선 거부**: CHOLMOD(`nrelax`/`zrelax` 가 커질수록 조임 — 260-col root 는 병합 안 함), SuperLU("공격적 병합은 fill 을 root 로 전파"), GPU-Cholesky 2024(누적 storage +25% cap, 병합은 작은 leaf 만).

> **본 실험 연결**: `--max-panel-width 8→32` sweep 에서 **25k/70k 평탄, 8387 약간 악화** — 문헌과 정확히 일치. root 의 문제는 "한 dense front 를 SM 에 펴는 occupancy"이지 front 를 키우는 게 아님. amalgamation 은 가장 비싼 front 에 FLOP 만 더함.

## Q5. power-flow / circuit Jacobian 반복 인수분해 (GPU)

- **(A) batch-to-fill**(B≥수십, true B=1 무관): Wang et al. SEGAN 2021(symbolic 1회+static-pivot batched refactor, **>100×**), Zhou et al. TPWRS 2017(**76× vs KLU**), SABLE(arXiv:2606.07099, **253× vs CPU 단 linear-solver stage 1.6–3.6×**).
- **(B) 단일행렬 column-level/level-scheduling**(B=1, 본 문제와 동일 under-occupancy): **GLU3.0**(arXiv:1908.00204: column당 block/warp 동적할당 3-mode, 13×), NICSLU-GPU, Gnanavignesh-Shenoy(**Jess-Kees reordering 으로 level 폭 확대·deep-tail 축소** — ordering 1회 재사용).
- **(C) OPF/cuDSS B=1 정직선**: Pacaud et al.(condensed-space, KKT pivot-free → cuDSS, **~4×**), Świrydowicz et al. EPSR 2023("GPU sparse-direct 는 큰 grid 만 경쟁력, **작은 grid 는 종종 *느림* — B=1 under-occupies**" — 이 regime 이 어렵다는 최적 인용).

> **본 실험 연결**: B=1 단일행렬 power-flow 접근은 전부 같은 deep-level under-occupancy 에 부딪히고, **per-level 동적 block/warp 할당 + 병렬성-최대화 reordering(Jess-Kees)** 로 *완화*(제거 아님). 큰 speedup 은 모두 batch-to-fill. → 본 솔버의 ordering best-of-k 는 (B) 계열의 reordering 레버와 같은 방향.

## 종합 권고 (문헌 → 본 프로젝트)

1. **ordering 이 최고 레버** (관측 15–67% 변동과 일치). **critical-path-FLOP/top-separator/tree-height 기반 선택**을 추구 — 본 실험의 `tail_cube` best-of-k 가 이미 그 시작. BBT(Kayaaslan–Uçar, height −28%)가 다음 toolbox.
2. **~260 root**: MAGMA-native 단일커널은 현 mid/big 커널이 근사. CPU fallback 도 후보(Karsavuran).
3. **look-ahead/amalgamation/tiling/더 많은 stream 은 B=1 occupancy 를 단독 구제 못 함** — 본 실험·문헌 공통.
4. **batch-to-fill 은 검증된 대박이나 다수 시스템 필요** — true B=1 엔 무관(NR contingency/time-series 가 있으면 그쪽이 정답).

**확인된 빈틈**: multifrontal 의 **single-RHS(B=1) 작은-root front 전략**은 미발표 — 실제 연구 공백. 본 실험의 `tail_cube`-선택은 이 공백의 ordering 측면에 대한 구체적·검증된 기여.

---
*원자료: deep web 조사(2026-06-12). 핵심 인용 — Kayaaslan&Uçar HiPC'14; Abdelfattah et al. JoCS'16; Catalán et al. arXiv:1804.07017; Ashcraft&Grimes TOMS'89; Karsavuran/Ng/Peyton arXiv:2409.14009; Świrydowicz et al. EPSR'23; Wang et al. SEGAN'21; GLU3.0 arXiv:1908.00204.*
