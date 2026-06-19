# STRUMPACK 격차의 커널-레벨 기전: 매핑 입도 · host 오버헤드 · small/big 라우팅 (ncu/nsys 실측)

> **상태**: 실측 완료(ncu/nsys, native FrontGPU 빌드)   **갱신**: 2026-06-17
> **한 줄**: 공정 튜닝(§08) 후 남는 ~13× 우위의 정체를 커널 레벨에서 규명 — **"small front 1개를 누가 맡느냐(매핑
> 입도)"**가 핵심. STRUMPACK은 front 크기와 무관하게 64-thread(2-warp) 블록 하나로 고정 → occupancy 4%. 우리는
> 크기에 맞춰 매핑(small=sub-group packing, >32=512-thread whole-front+multiblock) → occupancy 33–60%.

선행: [§08 공정튜닝](08-fair-strumpack-tuning-2026-06-17.md), [§06 head-to-head](06-head-to-head-2026-06-16.md).
재현 자산: `build_nomagma`(STRUMPACK MAGMA-off), `strumpack_bench_native`, env `CLS_ND_NPES`/`CLS_METIS_NSEPS`.

---

## 1. MAGMA vs no-MAGMA — 둘 다 GPU 경로다 (정정)

STRUMPACK `FrontFactory.cpp`: `is_GPU` → `#if MAGMA` FrontMAGMA `#elif GPU` **FrontGPU(native)**. **컴파일타임 게이트**
(런타임 스위치 없음). MAGMA 빌드는 항상 FrontMAGMA. native를 보려면 `STRUMPACK_USE_MAGMA=OFF` 재빌드(`build_nomagma`).

| 케이스 | 경로 | factor/sys | solve | MAGMA 우위 |
|---|---|---|---|---|
| 25k | native (FrontGPU, no-MAGMA) | 27.6 ms | 104.9 ms | — |
| 25k | MAGMA (FrontMAGMA vbatched) | 19.6 ms | 19.3 ms | factor 1.4×, **solve 5.4×** |
| USA | native | 64.2 ms | 594.3 ms | — |
| USA | MAGMA | 49.4 ms | 77.3 ms | factor 1.3×, **solve 7.7×** |

**no-MAGMA도 GPU를 쓴다** (nsys: `gpu::LU/Schur/solve_block_kernel_batched` = NT∈{8,16,24,32} 작은-front 전용 커널).
초기 "no-MAGMA=CPU" 추정은 **오류**(FrontMAGMA 전용 로그 `Need MB device mem` 부재를 잘못 해석). MAGMA의 이득은
주로 **solve(5–8×)**; factor는 두 GPU 경로가 비슷(1.3–1.4×). head-to-head가 MAGMA(더 빠른 쪽)를 baseline으로
쓴 것은 STRUMPACK에 유리한 공정 선택.

---

## 2. host-side 오버헤드 — H2D/D2H 아님, alloc churn + launch (nsys, 25k 공정구성)

| 항목 | 값 | 성격 |
|---|---|---|
| **H2D 전송** | **0.64 ms** (12 MB) | 무시 가능 |
| **D2H 전송** | **0.027 ms** (0.6 MB) | 무시 가능 |
| GPU 커널 (실 연산) | **7.4 ms** | — |
| cudaMalloc+Free+MallocHost+FreeHost | **~24.5 ms** | **alloc churn** |
| cudaLaunchKernel (581회) | **10.8 ms** | **launch 오버헤드** |
| cudaDeviceSynchronize (148회) | 105 ms | 대부분 GPU 대기(순수 오버헤드 아님) |

**전송(H2D/D2H)은 병목이 아니다(~0.67ms).** STRUMPACK은 **host 오버헤드 바운드**: GPU 실연산 7.4ms인데 host가
alloc churn 24.5ms + launch 10.8ms ≈ **35ms(연산의 5×)**를 orchestration에 씀. small-front라 매 factor마다 버퍼를
재할당하고 레벨마다 수십 커널 launch. 우리는 GPU-resident CUDA graph로 이 per-iter host 비용을 ~0으로 접음.

---

## 3. 핵심: 매핑 입도 — "front 1개를 누가 맡느냐" (ncu, 25k FP64)

### 3.1 small front (≤32)
| | STRUMPACK `LU/Schur_block_kernel_batched<,8>` | 우리 `factor_small<,32>` |
|---|---|---|
| **front 1개 담당** | **1 block = 8×8 = 64 thread (2 warp)** | **1 sub-group**(워프 일부), 워프당 32/SG front packing |
| block / grid | 64 thread / **4–32 block** | 256 thread / **257–1597 block** |
| **occupancy** | **4.16%** (grid 무관 고정) | **45–60%** |
| lane util (/32) | 25–29 (블록 내 높음) | 13.5–16.4 |
| **SM throughput** | **0.12–1.12%** | **28–60%** |

STRUMPACK은 front당 1블록(64 thread=2 warp) → SM 워프슬롯의 4%만 채우고, grid도 4–32개뿐 → **occupancy 4% 천장**.
우리는 sub-group 분해로 수천 front를 packing → grid 1597 → SM 채움. **이 occupancy 4%→60%(~14×)가 기전.**

### 3.2 >32 front (큰 uc)
| | STRUMPACK `Schur_block_kernel_batched<,8>` | 우리 `factor_mid` |
|---|---|---|
| front 1개 담당 | **64 thread (2 warp) 블록 — 크기 무관 고정** | **512 thread (16 warp)**, whole-front shared |
| **occupancy** | **4.17%** | **33.2%** |
| lane util (/32) | 25–27 | 30.7 (~96%) |
| SM throughput | 0.27% | 1.7–7% |

>32에선 front가 적어 packing 불가 → 둘 다 one-block-per-front. 차이는 **front당 thread 수(64 vs 512 = 2 vs 16
warp)**. STRUMPACK은 큰 uc Schur를 2-warp 블록이 8×8 타일로 직렬 순회 → occ 4%. 우리는 16 warp가 협력 + (big tier)
trailing multi-block으로 한 front의 uc²를 GPU 전체 분산. **STRUMPACK은 front 크기와 무관하게 64-thread 블록 고정.**

---

## 4. STRUMPACK은 nc(F11)로 라우팅 — 연구 표준인가? (소스 + 문헌)

STRUMPACK `FrontGPU.cpp:70`: `if (dsep<=32)` block kernel(NT∈{8,16,24,32}, **F11=nc=separator 차원만 타일**),
`else` cuSOLVER getrf. 즉 **분기 기준 = nc(F11), fsz 아님.**

**문헌 판정 (조사):** F11/nc 기준 분류는 **STRUMPACK 특유의 선택이고 표준 단일 기준이 아니다.**
- 고전 정의(Davis–Duff, UMFPACK): "front size" = order = **nc+uc**(full), F11은 피벗 블록일 뿐. 비용은 trailing
  `O(nc·uc²)` — **uc 지배**.
- 다른 솔버: CHOLMOD GPU(Rennich–Davis)=subtree+supernode 배치, SPQR GPU(Yeralan–Davis, Algo 980)=front 크기
  staging, MAGMA=full 차원. **nc만으로 나누는 건 STRUMPACK 뿐.**
- **F11만 보면 큰 uc front를 과소 매핑** — nc≤32지만 uc 큰 front를 작은 block kernel에 가둠(§3.2의 occ 4%). 우리는
  fsz(full) 기준 + big tier에서 uc² multi-block(비용 지배축 uc에 직접 대응).
- 출처: Ghysels–Synk(Parallel Computing 2022), Claus et al.(IJHPCA 2025), Davis–Duff(UMFPACK), Rennich et
  al.(Parallel Computing 2016), Yeralan–Davis(ACM TOMS Algo 980).

---

## 5. small/big 분포 — STRUMPACK 자신의 nc 라우팅, 실측 커널 시간 (정정판)

> 이전 proxy(우리 pw=64 nc로 nc·uc² 추정)는 **틀렸다** — pw=64가 leaf chain을 nc=64로 과합병해 power-flow를 41–60%
> "big"으로 오분류. **STRUMPACK native를 nsys로 직접 측정**(STRUMPACK 자신의 fundamental-supernode nc로 어느 커널에
> 가는지 + 실 GPU 시간):

| 행렬 | 총 커널 ms | small (block, nc≤32) | **big (cuSOLVER getrf, nc>32)** | assembly |
|---|---|---|---|---|
| case3120sp (PF) | 1.67 | 57.2% | **0%** | 42.8% |
| ACTIVSg25k (PF) | 7.11 | 71.1% | **0%** | 28.9% |
| SyntheticUSA (PF) | 25.3 | 81.0% | **0%** | 19.0% |
| scircuit (회로) | 33.5 | 81.4% | **2.1%** | 16.5% |
| cant (3D FEM) | 561 | 2.1% | **96.1%** (getrf_pivot 34% + trsm 32% + gemm) | 1.7% |
| parabolic (2D FEM) | — | native가 GPU 커널 미생성(CPU 라우팅/bail) | | |

**측정 결론:** **power-flow·회로의 big(getrf, nc>32) = 0~2%.** STRUMPACK은 fundamental supernode라 nc≤32로 유지 →
**cuSOLVER getrf를 아예 안 타고 100% block kernel(nc≤32)로 처리.** 그런데 그 block kernel이 occupancy 4%(§3) →
**STRUMPACK이 power-flow에서 느린 진짜 이유 = 작업 전부가 저점유 small-front 커널에 갇힘.** 3D FEM(cant)만 96%가
getrf(big-front 체제, vendor cuSOLVER).

---

## 6. 종합 한 줄
공정 튜닝(§08) 후에도 남는 ~13×는 **매핑 입도**에서 온다 — STRUMPACK은 front 크기 무관하게 64-thread(2-warp) 블록
하나로 고정(small=packing 부재, >32=thread 부족 → 양쪽 occupancy 4%), 우리는 크기별 매핑(small=sub-group packing
→60%, >32=512-thread+multiblock→33%). power-flow는 STRUMPACK 작업의 100%가 이 저점유 small-front 커널이고
(getrf 0%, 실측), 그것이 격차의 커널-레벨 기전이다.

## 7. 방법론·한계 (정직)
- (a) STRUMPACK 자체 front: native가 실행 중 자기 nc로 라우팅한 커널을 측정(block=nc≤32, getrf=nc>32). 우리 front 아님.
- (b) 실측 GPU 커널 시간(nsys/ncu). nc·uc² 추정식 아님(이전 proxy 폐기).
- ncu: occupancy/lane/SM-throughput, RTX 3090, 클럭 미고정(배율은 견고).
- 미완: parabolic native(CPU 라우팅), TSOPF/bmwcra(analyze bail — 8GB arena/초대형 fill). 우리 솔버 측 실측
  커널-시간 split은 미작성(필요시 추가).
</content>
