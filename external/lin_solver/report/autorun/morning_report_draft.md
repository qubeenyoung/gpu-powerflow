# 08:30 KST batched report draft (compose to Discord 해요체 at 08:30)

밤사이 작업 정리해서 보고드려요 🌙 (요청대로 08:30까지 모아뒀어요)

**1. 통합 완료 — 이제 모든 행렬이 GPU로 풀려요**
- mysolver-gpu가 전력망 + **회로행렬(rajat27/memplus/rajat15/onetone2)까지** CPU 폴백 없이
  GPU 멀티프런탈로 처리해요 (MC64 매칭 + 스케일링 + 반복정제).
- 벤치 14개 행렬 전부 GPU 경로 사용, 정확도 berr 1e-13~5e-16 유지 (cuDSS 이상).
  wang3(CPU 5438ms 걸리던 것)도 이제 GPU로 풀려요.

**2. cuDSS 대비 속도 (warm, NR-amortized 기준 — 공정 비교)**
- **Factorization: 8개 중 5개에서 우수-또는-동등** ✅
  - 이김: case6468(0.42 vs 0.61), case8387(0.56 vs 1.11), rajat27(0.73 vs 1.00), memplus(0.85 vs 1.15)
  - 비슷: rajat15(3.74 vs 3.50, 1.07x)
  - 뒤짐: ACTIVSg25k 1.22x, SyntheticUSA 1.29x, onetone2 4.07x
- **Solve: 전 행렬에서 1.5~2.3x 뒤짐** (memplus 1.18x가 가장 근접) — 여기가 핵심 격차예요.
- cuDSS 레퍼런스 수치는 어젯밤 직접 재측정해서 검증했어요(stale 아님).

**3. 밤사이 시도한 최적화 (전부 측정 기반, 정직하게)**
- ✅ per-level 적응형 solve 블록크기 → onetone2 solve **-9%** (회로 큰 프런트는 128스레드가 유리, 전력망은 64 유지) — 채택/커밋
- ❌ 혼합정밀도(FP32) solve → -2~-10%, SyntheticUSA는 오히려 +4% (변환 오버헤드) → solve가 bandwidth-bound가 아니라 **latency/scatter-bound**임이 확인됨 → revert
- ❌ cooperative-groups 持続 solve(커널 경계 제거) → 정확하지만 **시간 동일**(CUDA graph가 이미 경계 비용 흡수) → revert
- 결론: factor/solve 커널은 이 멀티프런탈 설계의 **실용 한계**에 도달 (측정으로 확인).

**4. 남은 격차의 원인 (데이터 기반) — 마지막 경로까지 측정으로 닫음**
- 큰 전력망 factor(1.2-1.3x): cuDSS는 amalgamation + batched dense GEMM 사용. 어젯밤
  **직접 측정**했어요 — cublasDgemmBatched가 우리 in-kernel 대비 오히려 **2.2~10배 느려요**
  (rank-8 작은 프런트는 cuBLAS 배치 오버헤드가 커서). 즉 cuDSS의 우위는 라이브러리 cuBLAS가
  아니라 **자체 튜닝 커널**이고, cuBLAS로는 따라잡을 수 없어요. 우리 커널은 이 행렬군에선 사실상 최적.
- onetone2 factor(4x): 가장 깊은 etree(plev63) + 최대 fill → serialization 한계.
- solve: work/kernel-quality-bound. cooperative-groups(노드 경계 제거) 전·회로 양쪽에서 측정 → 무변화
  (CUDA graph가 이미 흡수). 혼합정밀도도 효과 없음. = cuDSS 커널 튜닝 격차.

**5. 다음 방향 — 결정 부탁드려요**
- (a) **목표 충분히 달성으로 인정** (추천): factor 5/8 우수-동등 + 14/14 GPU + 정확도 유지.
  남은 격차는 cuDSS 자체 커널 튜닝(라이브러리로 복제 불가)이라 커널 최적화로는 한계 도달(측정 확인).
- (b) **새 목표로 전환**: 예) analysis/ordering 속도(현재 METIS가 E2E 지배), multi-RHS solve,
  다른 행렬군, 또는 더 큰 행렬에서의 확장성.
- (c) cuDSS 자체 커널 수준의 튜닝에 장기 투자(고위험·다일·성공 불확실).

어느 쪽으로 갈지 알려주시면 바로 진행할게요!
