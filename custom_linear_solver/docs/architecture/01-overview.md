# 01 — 개요 & 정신 모델

> **층위**: 추상. 이 문서는 "무엇을, 왜, 어떤 순서로" 하는지의 큰 그림이다. 구현 세부는 04~09에서 다룬다.

---

## 1. 한 문장

`custom_linear_solver`는 **고정 sparsity의 비대칭 희소행렬 `A`를 multifrontal LU 분해로 GPU에서 풀고, 같은 패턴의
여러 우변/값을 배치로 처리**하는 라이브러리다. 1차 타깃은 Newton–Raphson(NR) 전력조류의 Jacobian이다.

## 2. 핵심 가정 넷 (설계 전체가 여기서 나온다)

1. **sparsity 고정, 값만 변함** — NR 루프는 매 반복 같은 비영 패턴에 값만 바꾼다. 그래서 비싼 **기호 분석
   `Analyze()`는 한 번**, 수치 `Factorize()`/`Solve()`만 반복한다.
2. **B-시스템 배치** — 같은 패턴·다른 값의 시스템 `B`개를 **한 symbolic plan**으로 동시에 분해/solve.
3. **device-resident 실행** — factor/solve를 CUDA graph로 캡처해 replay. host↔device 왕복은 수렴 스칼라 정도뿐.
4. **no-pivot** — 행렬이 대각우세(power-grid NR Jacobian)라고 가정. 수치 안정성을 위한 부분 pivoting을 하지 않고,
   특이 pivot만 정적 대각 shift로 막는다. → no-pivot의 정당성·한계는 [`../_legacy/02-design-analysis/02-no-pivoting-proof.md`](../_legacy/02-design-analysis/02-no-pivoting-proof.md).

이 도메인의 결정적 성질 하나: **front가 극단적으로 작다**(대부분 `fsz ≤ 16`). 일반 GPU 희소 직접 솔버는 큰 dense
front(거기에 FLOP이 몰림)를 가정하지만, 전력망 Jacobian엔 큰 front가 거의 없어 그런 라이브러리는 peak의 작은
일부만 쓴다. 이 솔버의 존재 이유는 **그 작은-front 분포에 맞춰 GPU 커널을 짠 것**이다.

## 3. multifrontal LU — 표준 추상화

multifrontal 직접 분해는 다음을 한다.

1. **fill-reducing 재정렬**(METIS nested dissection)로 분해 중 생기는 채움(fill)을 줄인다.
2. **elimination tree**를 만든다. 각 노드(supernode)는 인접 열 몇 개를 묶은 단위다.
3. 각 supernode를 작은 **dense 부분행렬 = front**로 보고, 그 front의 LU를 푼다.
4. front의 잔차(**Schur complement** = contribution block)를 **부모 front로 누적(extend-add)**한다.
5. tree를 leaf→root로 올라가며 반복하면 전체 LU가 완성된다.

표준 4단계(한 front 기준): **① panel LU → ② U-panel 삼각 solve → ③ trailing update(Schur) → ④ extend-add**.
상세는 [`06-factorization.md`](06-factorization.md).

## 4. end-to-end 파이프라인 (한눈에)

```
   사용자                     라이브러리                         GPU
   ─────                     ─────────                         ───
 SetData(A) ───────────────▶ 행렬 등록(CSR, device)
 SetRhs(b)/SetSolution(x) ─▶ 우변/해 등록

 Analyze()  ───────────────▶ [기호] CSR→CSC → A+Aᵀ 그래프 ──▶ METIS ND ──▶ etree
            (한 번)                → fill 패턴 → supernode 묶기 → multifrontal plan
                                   → tier 버킷·subtree 분할 → device arena 업로드

 Setup(B)   ───────────────▶ [런타임] B개분 front/벡터 arena 할당, 스트림/이벤트 생성
            (B 바뀔 때)          (내부 graph 모드면) factor graph 캡처

 Factorize()───────────────▶ [수치] 값 scatter → 분해(graph replay 또는 직접 발행) ─▶ L,U
            (매 NR 반복)
 Solve()    ───────────────▶ [수치] gather(b) → 전진/후진 대입 → scatter ─▶ x
```

- **Analyze**: 패턴만 본다. 한 번. [08 §1](08-runtime-and-batching.md).
- **Setup(B)**: B 시스템용 메모리/그래프를 잡는다. B가 바뀌면 다시. [04](04-memory-layout.md), [08 §2](08-runtime-and-batching.md).
- **Factorize/Solve**: 값으로 동작. 매 NR 반복. [06](06-factorization.md), [07](07-solve.md).

## 5. 용어집 (이 문서들 전반에서 쓰는 약어)

| 약어 | 뜻 |
|---|---|
| **front** | 한 supernode의 dense 부분행렬(`fsz × fsz`). LU의 작업 단위. |
| **fsz** | front 크기(front size). tier 라우팅의 입력. |
| **nc** | pivot 열 수(이 supernode가 소거하는 변수 수). front의 좌상단 `nc×nc` 블록. |
| **uc** | contribution 행/열 수 = `fsz − nc`. 부모로 넘길 잔차 블록(CB)의 차원. |
| **panel** | nc개 pivot 열로 이뤄진 supernode 패널(= front의 pivot 부분). |
| **CB** | contribution block = `uc×uc` Schur 잔차. extend-add로 부모에 더해진다. |
| **extend-add** | 자식 CB를 부모 front의 해당 행/열 위치에 더하는 조립 연산. |
| **supernode amalgamation** | 인접 열을 한 panel로 묶어 front/커널 수를 줄이는 것(`max_panel_width`). |
| **tier** | front 크기로 정해지는 커널 부류 small/mid/big. [05](05-front-tiers.md). |
| **B** | 배치 크기(동시에 푸는 시스템 수). `B==1`은 단일 시스템 특수 경로. |
| **selinv** | partitioned inverse — pivot 블록을 미리 역행렬화해 solve를 GEMV로 바꾸는 것(B=1). [07 §4](07-solve.md). |

## 6. 작은 예제로 따라가기 (개념)

변수 6개 그래프가 nested dissection으로 자식 `A={1,2}`, `B={3,4}`와 separator `{5,6}`로 갈린다고 하자.
elimination tree는 `(leaf A, leaf B) → root {5,6}`이다.

1. **leaf A의 front**(4×4): nc=2(자기변수 1,2), uc=2(separator 행 5,6). A의 LU를 풀고, 잔차 CB(2×2)를 root에 더한다.
2. **leaf B의 front**도 같은 모양. 잔차 CB를 root에 더한다(A와 B는 독립 → 동시 가능, 부모는 아직 분해 전이라 race-free).
3. **root front {5,6}**: A·B가 더한 잔차가 합쳐진 상태에서 LU를 푼다. 부모가 없으니 끝.

이 "front마다 LU 풀고 잔차를 부모에 더한다"가 multifrontal의 전부다. 나머지는 — 작은 front를 GPU에 **어떻게
매핑하느냐**(tier), 값을 **어떻게 배치하느냐**(메모리), B개를 **어떻게 겹치느냐**(런타임) — 의 엔지니어링이다.
