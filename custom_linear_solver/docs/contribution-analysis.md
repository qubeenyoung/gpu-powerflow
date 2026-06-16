# Contribution analysis — custom linear solver

> **상태**: canonical (정직 버전)   **갱신**: 2026-06-16 (head-to-head + 문헌조사 후)
> **한 줄**: **신규성 = packing과 full-front fusion의 *입도 배타성*을 sub-group 분해로 해소해 둘을 한 커널에서
> 동시에 달성한 것** ([head-to-head 리포트 §4c](05-reports/06-head-to-head-2026-06-16.md)). MAGMA는 packing-only
> (op 입도), STRUMPACK native는 fusion-only(front당 블록) — 둘 다 occupancy 2%에 갇힌다. 우리만 sub-group
> 입도(SG lane=front 1개, 32/SG front/warp)로 둘을 동시에 → occupancy 2%→30–59%(§5b ncu) 회복이 곧 B=1 ~20×다.
> 문헌조사(18 1차 소스, 3표 검증): 이 *동시* 구조는 survey 범위 내 미보고. 단순 조합이 아니라 — 두 baseline의
> 입도는 *합쳐질 수 없는* 것이고 sub-group은 *세 번째 입도*다. 실증: 동일 FP64·동일 GPU·깊이 매칭 후에도
> STRUMPACK+MAGMA(두 경로 best) 대비 **factor 16–22×, solve 33–66×**(B=1), cuPF 통합 ~3–4×. 개별 조각
> (packing·shared 융합·tiering·cuDSS UBATCH·GPU-resident 개념)은 prior art임을 인정한 위에서의 신규성.

선행연구 조사: [`01-orientation/02-related-work-and-novelty.md`](01-orientation/02-related-work-and-novelty.md).
서사: [`storyline.md`](storyline.md).

---

## 0. 범위 (scope)

- **이 문서는 custom linear solver만 다룬다.**
- **cuPF는 기여에서 제외.** cuPF는 **FP64 NR 외부 루프에서 Jacobian 조립·선형계만 FP32(/TF32)로 푸는
  혼합정밀**이다(나머지는 FP64). 이 혼합정밀 NR의 수렴·정확도는 cuPF의 결과이지 솔버의 주장이 아니다. 솔버는
  그 FP32/TF32 선형-풀이 모드를 제공할 뿐이다.

---

## 1. 솔버가 무엇인가 (코드 사실, 간단히)

배치 멀티프론탈 LU(no-pivot, 대각우세 가정), METIS-ND, B개 시스템이 **하나의 symbolic** 공유, front-major
배치 arena, etree 레벨당 커널 1회(`blockIdx.y=batch`). FP64/FP32/TF32 모드 제공(TF32는 텐서코어 trailing).
front 크기별 4-tier 커널(tiny warp-packed / small whole-front shared / big panel-resident / large
global-resident). 전 과정 GPU-resident(CUDA 그래프). 코드: `analyze/`, `internal/`, `factorize/`, `solve/`.

---

## 2. 정직한 현실 — 개별 기법은 전부 prior art

심사에서 "이거 MAGMA/STRUMPACK/cuDSS 아니냐"에 먼저 항복할 부분을 명시한다. 이걸 novelty라 우기면 깨진다.

| 우리가 쓴 기법 | 선행 | 출처 |
|---|---|---|
| tiny front를 **워프당 여러 개 패킹** | **MAGMA** (32 미만 행렬은 across-matrices 병렬 = 워프당 다중, register/shuffle) | Dong·Abdelfattah·Haidar·Dongarra |
| **front 크기별 티어 라우팅** | **STRUMPACK** (`<32×32` custom, 중간 MAGMA vbatched/irrLU, 큰 front cuBLAS/cuSOLVER) | Ghysels et al. |
| **panel-in-shared / trailing-in-global** 블로킹 | dense-LU 표준 | — |
| 큰 front 텐서코어 + TF32/Ozaki | 표준(big→TC) + published Ozaki | — |
| **배치 + 단일 shared-symbolic** | **cuDSS uniform-batch**(`UBATCH`, v0.6+), SuperLU_DIST, Wang | NVIDIA cuDSS |

따라서:
- **"cuBLAS/cuSOLVER 대신 자체 커널" 자체는 기여가 아니다.** 구현 선택일 뿐.
- **"단일 symbolic 공유"는 cuDSS가 이미 uniform batch로 제공.** 개념 신규성 없음.
- **SABLE(2026.06)도 솔버로는 non-novel** — cuDSS 블랙박스 + 고정패턴 배치(기존 개념). 신규성은 ML
  미분가능 레이어 통합 쪽이지 선형 솔버가 아니다.

---

## 3. 그러면 novelty는 어떤 형태여야 하나

기법이 아니라 **"baseline의 구체적 한계를 이 행렬류에서 드러내고, 특화가 그걸 정량적으로 극복한다"** 는
*실증적·경쟁적* 기여여야 한다. 이는 **반드시 공정 head-to-head**를 요구한다 — 공정 비교 없이는 "더 빠르다"도
"그들이 X에서 약하다"도 주장할 수 없다.

### 검증해야 할 "한계 가설" (아직 가설, 측정 전)

baseline마다 *왜 batched tiny-front power-flow에서 불리할 것인가*를 가설로 세우고, 실험으로 확인/반증한다.

- **STRUMPACK**: (가설) 단일 시스템 → B개를 한 symbolic으로 못 묶음(순차 B회 or B만큼 메모리), 작은 레벨
  under-fill; FP64 중심; 큰 front를 벤더에 위임 → 정밀도/점유율 일관 최적화 불가. **측정으로 확인 필요.**
- **cuDSS**: (가설) uniform-batch는 있으나 FP16/TC factor 모드 없음, 블랙박스라 tiny-front 특화 여부·점유율
  불명. **가장 직접적인 경쟁자 — 반드시 포함.**
- **MAGMA**: dense 배치 *빌딩블록*이라 sparse 솔버로 직접 비교 불가 → STRUMPACK(=MAGMA 사용) 또는
  front-size 분포 위 커널 수준 비교로만 다룸.

**중요**: 위는 전부 *주장이 아니라 검증 대상*이다. 측정에서 baseline이 더 빠르면, 우리 기여는 없는 것으로
보고한다(이 repo의 honesty 원칙).

---

## 4. 공정 비교 설계 (이 작업의 게이트)

- **같은 행렬**: 동일 power-flow Jacobian(MATPOWER/ACTIVSg 케이스, 동일 ordering 입력 또는 각자 최적 ordering 명시).
- **같은 정확도 목표**: relres 동일 기준. 우리 FP32/TF32 vs baseline FP64는 *정확도-정규화* 비교(또는
  baseline도 가능한 최저 정밀도)로, "정밀도를 낮춰 이겼다"가 교란되지 않게.
- **같은 작업단위**: 1회 factor + 1회 solve(전력조류 NR 1스텝). B sweep(1/16/64/256).
- **공정한 baseline 사용**: STRUMPACK/cuDSS의 권장 설정·reuse(symbolic 재사용, refactorization) 켜기.
- **프로파일로 "왜"**: 단순 wall-time이 아니라 ncu로 baseline의 병목(occupancy/launch/DRAM)을 제시 →
  한계의 *기전*을 보여야 기여가 됨.

### 예비 결과 (확정 아님)
cuPF 통합 기준 custom solver 적용 시 약 **3–4×** 관측. 단, 이는 *공정 head-to-head로 재현·정규화하기 전까지*
기여로 확정하지 않는다.

---

## 5. 실험 계획 (이 브랜치)

1. **MAGMA 설치·빌드**(CUDA) — STRUMPACK 의존 + 커널 수준 비교용.
2. **STRUMPACK 설치·빌드**(MPI/METIS/BLAS/LAPACK/CUDA) — 멀티프론탈 baseline.
3. (가능하면) **cuDSS** uniform-batch baseline — 가장 직접적 경쟁자.
4. **공통 하니스**: 동일 Jacobian .mtx 입력, 동일 정확도, B sweep, factor/solve 분리 계측, ncu 프로파일.
5. **분석**: baseline이 *어디서·왜* 느린지(또는 빠른지) → 한계 기전 제시 또는 기여 철회.

---

## 6. 폐기된 주장 (코드 삭제됨)

partitioned-inverse GEMV solve(`deprecated/selinv/`), TC-routable front coarsening
(`deprecated/amalgamation/`), best-of-k ND 선택(`deprecated/best_of_k/`). 기여 목록에서 제외.

---

## 7. 결론 (head-to-head + 문헌조사 후)

게이트(공정 head-to-head + 문헌조사) 통과. 정직하게 정리하면:

- **prior art로 항복하는 것** (차별점 아님): shared 단일 커널 factor+update 융합(STRUMPACK native, Ghysels &
  Synk §3.2), 작은 행렬 packing/vbatched(MAGMA), 크기별 tier, 단일-symbolic 배치(cuDSS UBATCH),
  power-flow GPU-resident *개념*(Swirydowicz). "자체 커널"·"단일 symbolic"은 신규성 아님.
- **신규성 = 입도 배타성의 해소**: packing과 full-front fusion은 별개 축인데, 두 baseline은 입도가 배타적이라
  각각 하나만 한다 — MAGMA는 packing-하되-융합-안-함(op 입도), STRUMPACK native는 융합-하되-packing-안-함
  (front당 블록). 둘 다 *다른 이유로* occupancy 2%. 우리는 **세 번째 입도(sub-group, front보다 작음)**로 둘을
  한 커널에서 동시에 달성. **단순 조합 아님** — baseline 입도는 합쳐질 수 없고, sub-group 분해가 그걸 깬 것.
- **20× 기전**: 그 동시 구조 → occupancy 2%→30–59%, lane 7%→45–70%(§5b ncu). 이 회복(~12–20×)이 곧 측정된 20×.
- **실증 우위**: 동일 FP64·동일 GPU, **트리 깊이까지 매칭**한 뒤에도 STRUMPACK+MAGMA(두 경로 best) 대비 B=1
  factor 16–22×, solve 33–66×. cuPF 통합 ~3–4×.
- **정직한 한계**: cuDSS closed-source(내부 입도 미검증), 특허·미공개 미조사 → "전 세계 최초"가 아니라
  "**survey 범위 내 처음**". cuDSS보다 빠른 건 별도 증명됨(정황은 우리 편, 기전은 미확인).

⇒ **방어 가능한 기여**: *"전력조류 tiny-front 멀티프론탈에서, packing과 full-front fusion의 입도 배타성을
sub-group 분해로 해소해 둘을 한 커널에서 동시에 달성"* — 이 구조가 occupancy를 12–20× 회복해 측정된 16–66×
우위를 만들고, survey 범위 내 미보고 지점이다.

---

**참고문헌** (전체는 [`01-orientation/02-related-work-and-novelty.md`](01-orientation/02-related-work-and-novelty.md)):
MAGMA tiny-batched (Dong/Abdelfattah/Haidar/Dongarra), STRUMPACK GPU multifrontal (Ghysels et al.),
cuDSS uniform-batch (NVIDIA), Wang/Fraunhofer FP64 batched power-flow(arXiv:2101.02270),
Swirydowicz GPU-resident ACOPF(IJEPES'23), Spatula(MICRO'23), SABLE(arXiv:2606.07099).
