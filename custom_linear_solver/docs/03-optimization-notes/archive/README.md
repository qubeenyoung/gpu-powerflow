# `03-optimization-notes/archive/` — 미적용 / 폐기된 최적화 기록

여기에 있는 문서는 **현재 코드에 반영되지 않은** 최적화 시도다. 회귀를 일으켰거나, 측정
ROI 가 부족했거나, 연구 단계에서 멈춘 가설들. 본 솔버가 지금 무엇을 하는지 이해하려면
`../` 의 active 문서들 (01–05, 09, 10, 13, 15, 16, 17) 을 보면 된다. 본 폴더는 다음 두
용도로만 보존된다:

1. **같은 lever 를 재시도할 때 동일한 실수를 피하기 위한 근거.** 어떤 가설이 왜 실패했는지,
   어떤 측정으로 기각됐는지가 기록되어 있다.
2. **변환률 / noise floor 같은 메타-수치의 출처.** 일부 active 문서가 이 기록의 측정 결과를
   참조한다 (예: docs/14 의 "sync −64% → wall −1~−4%" 변환률).

## 분류

### Research log (가설 단계, 미실행)

- [`06-tc-dedicated-path-study.md`](06-tc-dedicated-path-study.md) — `src/tc/` 전용 경로
  연구. WMMA tile grid 가 power-grid front 크기에서 setup amortize 불가 → 폐기. 현재 적용된
  FP16 path 는 batched 라우팅 의 `factor_*_tc` 로 진화 (docs/04).
- [`07-symbolic-gemm-research.md`](07-symbolic-gemm-research.md) — symbolic 재구성으로 work
  를 GEMM 형태로 변환하는 연구. ND ordering 제약 + amalgamate cap≥16 에서 +72% 회귀로 보류.
- [`08-tree-restructuring-research-plan.md`](08-tree-restructuring-research-plan.md) — elimination
  tree 자체 재구성 plan. spine fusion / multi-stream race / sibling amalgamation 모두 회귀.

### Failed experiment (구현 후 측정으로 회귀 확인)

- [`11-small-packed-experiment-2026-06-06.md`](11-small-packed-experiment-2026-06-06.md) —
  small tier multi-front-per-warp packing. case8387 +65~+133% 회귀. 유일 win 은 USA B=64
  −5.7% 한정. 명확한 negative result.
- [`14-factor-mid-opt-experiment-2026-06-06.md`](14-factor-mid-opt-experiment-2026-06-06.md) —
  `factor_mid_opt` 별도 커널 (P1+P2+P4 결합). P4 stage-in padding 이 integer division
  overhead 로 +85% 회귀, P1+P2 만 USA B≥16 −4%. 별도 path 유지 비용 정당화 못함 → rollback.
  P1 (reciprocal multiply) 만 default kernel 에 흡수.
- [`18-small-mid-memory-bound-2026-06-07.md`](18-small-mid-memory-bound-2026-06-07.md) —
  small/mid tier 의 memory-bound 분석. EXP-D (small warps/block 8→16) 시도 → +4~10% 회귀
  ("more warps in flight to hide latency" 가설 reject). lever 없음.

### Deprecated path (코드에서 이미 제거됨)

- [`12-tier-split-experiment-2026-06-06.deprecated.md`](12-tier-split-experiment-2026-06-06.deprecated.md) —
  T-split (`CLS_USE_TIER_SPLIT`). B≥16 −4~−6% marginal win + B<32 회귀 + dispatch 복잡도
  → 코드 제거.
- [`19-m2-persistent-subtree-rd-2026-06-07.deprecated.md`](19-m2-persistent-subtree-rd-2026-06-07.deprecated.md),
  [`20-m2-prototype-status-2026-06-07.deprecated.md`](20-m2-prototype-status-2026-06-07.deprecated.md) —
  M2 persistent subtree-walking 프로토타입. wall +240~820% 회귀로 전체 삭제.

## 파일 번호 보존 정책

본 폴더의 파일 번호 (06, 07, 08, 11, 12, 14, 18, 19, 20) 는 `../` 의 active 시리즈와의
시간순 cross-reference 를 유지하기 위해 그대로 둔다. active 폴더의 09, 10, 13, 15, 16, 17
중 일부가 본 폴더 문서를 참조하는 경우 `archive/NN-...` 로 링크된다.
