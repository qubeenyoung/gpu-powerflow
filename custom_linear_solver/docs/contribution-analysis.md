# Contribution analysis — custom linear solver

> **이 문서는 [`main-report.md`](main-report.md) 로 통합되었습니다 (2026-06-16).**

기여·신규성 논증은 통합 리포트의 다음 절로 이동했다:

- **§3 핵심 기여 — 논리 구조** ([main-report.md](main-report.md#3-핵심-기여--논리-구조)):
  개별 기법은 prior art(§3.1) → head-to-head 격차(§3.2) → packing/fusion 입도 배타성(§3.3) →
  sub-group 분해로 해소(§3.4) → occupancy 회복 = 기전(§3.5) → 문헌 판정(§3.6).
- **§6 부가 요소(정직한 분해)**, **§7 정직한 한계**.

실험 상세는 [`05-reports/06-head-to-head-2026-06-16.md`](05-reports/06-head-to-head-2026-06-16.md),
선행연구는 [`01-orientation/02-related-work-and-novelty.md`](01-orientation/02-related-work-and-novelty.md).
