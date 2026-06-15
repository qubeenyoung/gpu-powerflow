# big_split_2d — DEPRECATED (2026-06-11)

Big-tier TF32 trailing GEMM split across thread blocks, to fill idle SMs when an elimination-tree
level is under-utilized (few big fronts × batch < SM count, e.g. single-system B=1 near the root).

## 왜 만들었나
ncu 진단: big front 은 1 block/front → 1 block/SM(89 reg + shared staging) → warp 26%만 활성,
`__syncthreads()` barrier 가 지배 → 텐서코어 pipe 2.4%만 가동. "TC 가 느린 게 아니라 못 먹임".
→ trailing 을 여러 블록으로 fan-out 해 occupancy 를 올리면 TC 를 더 먹일 수 있다는 가설.

## 결과 (검증됨)
- **lever 는 작동**: v2(2D-tile) 에서 occupancy 제한 1→9-10 block/SM, barrier stall 5.34→0.04 (ncu).
- **usa B=1 factorize 1.14× / B=4 1.10×**.
- **그러나 조건부·취약**:
  - **70K 는 회귀(~0.95×)** — split_fill threshold 4~64 전부.
  - **B≥16 은 손해** (이미 GPU 가 참).
  - 분포가 거의 같은 usa/70K 가 정반대 → 분포만으론 예측 불가, multistream 이 이미 유휴 SM 일부를 채움.

## 왜 deprecate
2-커널 구조(panel → tiled trailing)가 **레벨마다 global L/U 왕복 + 2번째 launch** 를 치른다. 이
오버헤드가 occupancy 이득을 잡아먹어 **usa 단일시스템(B=1)에서만** 이득. 프로덕션은 보통 배치
(B≥16)로 돌고 factorize 전체를 CUDA graph 로 캡처하므로 split 은 도움 안 되거나 방해됨.

깔끔히 이기려면 **cooperative single kernel**(`grid.sync()` 로 panel→trailing 을 한 커널에서) 이
필요 — 2번째 launch 를 없앰(주 오버헤드). 단 cooperative launch 가 whole-iteration CUDA graph
캡처와 충돌할 가능성이 커서 미해결. (global 왕복 자체는 ~16 KB/front 라 싸다 — 진짜 비용은 launch.)

## 되살리려면
`big_split_2d.cuh` 의 **v2 (`factor_big_trail_tile_tf32` + `trailing_tile_tf32`)** 가 1.14× 를 낸
버전. dispatch 스니펫(`#if 0` 블록)을 `dispatch_factor_big` 의 TF32 분기에 복원하고, 게이트를
`under-filled && max_uc ≤ TC_UC_CAP && front 충분히 큼` 으로. cooperative 로 가려면 두 커널을
`grid.sync()` 로 합치고 graph 호환을 먼저 검증할 것.

## 관련
- 설계/한계: `docs/03-optimization-notes/05-tc-eligibility-relaxation-2026-06-11.md` (TC cap 완화는
  shipped — split 과 독립). split 정책 cost-model 은 세션 로그 참조.
- ncu 병목 진단: `docs/20260612_lab_meeting/small-tier-no-tensorcore.md` §5(d).
