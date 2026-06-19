# `03-optimization-notes/archive/` — 미적용 / 폐기된 최적화 기록

여기에 있는 문서는 **현재 코드에 반영되지 않은** 최적화 시도다. 회귀를 일으켰거나, ROI 가
부족했거나, 연구 단계에서 멈춘 가설들. 본 솔버가 지금 무엇을 하는지 이해하려면 `../` 의
active 문서들을 보면 된다.

본 폴더의 유일한 목적: **같은 lever 를 재시도할 때 동일한 실수를 피하는 것.** 어떤 가설이
왜 실패했는지, 어떤 측정으로 기각됐는지가 기록돼 있다. (일부 active 문서가 참조하는
변환률 / noise floor 같은 메타-수치의 출처이기도 하다 — 예: "sync −64% → wall −1~−4%" 변환률.)

## 인덱스

| 파일 | 내용 (한 줄) |
|------|--------------|
| [`01-tc-dedicated-path.md`](01-tc-dedicated-path.md) | `src/tc/` 전용 TC 경로 — WMMA tile grid 가 power-grid front 크기에서 setup amortize 불가. negative result. |
| [`02-symbolic-gemm-research.md`](02-symbolic-gemm-research.md) | symbolic 재구성으로 work 를 GEMM 형태로 reshape — ND ordering 제약 + amalgamate cap 에서 +72% 회귀. dead-end. |
| [`03-tree-restructuring.md`](03-tree-restructuring.md) | elimination tree 자체 재구성 (대안 ordering / amalgamation) — spine fusion / multi-stream race / sibling amalgamation 모두 회귀. dead-end. |
| [`04-deprecated-experiments.md`](04-deprecated-experiments.md) | 6개 폐기 미세실험 묶음 — small-packed (case8387 +65~133%), tier-split (USA −4~6% 한정), factor_mid_opt (P4 +85%), small/mid memory-bound (EXP-D +4~10%), M2 persistent subtree ×2 (wall +240~820%). |
