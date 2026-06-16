# 연구 서사 — 전력망 조류계산을 위한 배치 GPU 직접해법

> **이 문서는 [`main-report.md`](main-report.md) 로 통합되었습니다 (2026-06-16).**

서사·문제정의·기여·정직한 분해가 모두 통합 리포트에 있다:

- **§1 문제와 난점 — 근원은 작은 front** ([main-report.md](main-report.md#1-문제와-난점--근원은-작은-front))
- **§3 핵심 기여 — 논리 구조** (개별 기법은 prior art → head-to-head → 입도 배타성 → sub-group 해소 →
  occupancy 회복 → 문헌 판정)
- **§4 4-tier 구현**, **§6 부가 요소(정직한 분해)**, **§7 정직한 한계**

상세 실험: [`05-reports/06-head-to-head-2026-06-16.md`](05-reports/06-head-to-head-2026-06-16.md).
