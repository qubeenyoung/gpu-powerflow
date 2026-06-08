# Level-contiguous front arena — 음성 결과 (postorder 유지)

**작성일**: 2026-06-08
**선행**: [`docs/04/14`](14-fp32-singlestream-baseline-2026-06-08.md) (fp32 single-stream baseline)
**대상**: `src/plan/analyze.cu` `front_off` 할당 순서
**결론**: **기각.** front arena를 level별 연속으로 깔면 regression. postorder(panel-id 순) 유지.

## 0. TL;DR

| case | B | factor ID→LEVEL | solve ID→LEVEL |
|------|--:|:---------------:|:--------------:|
| case8387 | 1  | 0.336→0.342 (+2%) | 0.254→0.247 (−3%) |
| case8387 | 64 | **0.032→0.040 (+26%)** | **0.021→0.024 (+16%)** |
| ACTIVSg25k | 1  | 0.854→0.920 (+8%) | 0.524→0.548 (+5%) |
| ACTIVSg25k | 64 | 0.120→0.124 (+3%) | 0.054→0.054 (~0) |

interleaved ID vs LEVEL, 4 trial median. 어떤 config도 level-order가 이기지 않음.

## 1. 시도한 것

`front_off[p]`(front arena 내 panel p의 base offset)를 panel-id 순 prefix-sum이 아니라
**최종 `plcols`(level-major, level 내 subtree-contiguous) 순서**로 prefix-sum 하도록 변경.
같은 level의 front들이 `d_front`에서 하나의 연속 블록을 차지하게 됨.

- 안전성: `front_off`가 arena 배치의 **단일 진실원천**이고(모든 커널이 `front + front_off[p]`,
  front 크기는 `front_ptr`에서 파생), `a_pos`는 device에서 `front_off`로 계산되므로 소비자 코드
  변경 0. `total`(=Σfsz²) 불변.
- 정확성: relres 동일(8387 ~2e-5, 25k ~1.3e-4). 구현 자체는 정상.

## 2. 왜 손해인가 (메커니즘)

Multifrontal의 지배적 메모리 연산은 child→parent **extend-add**다. panel id가 postorder라서
**id-order 저장은 부모 front를 자식들 바로 위(인접 메모리)에 배치**한다 → extend-add의 공간/시간
지역성이 이미 최적에 가깝다. level-order로 깔면 부모는 한 레벨 위의 **먼 region**으로 분리되어
child→parent scatter의 stride가 커지고 L2 재사용이 깨진다. fan-in이 큰 **B=64에서 +26%**로
가장 크게 드러남(여러 자식이 동시에 멀리 있는 부모로 흩뿌림).

즉 dispatch *순서*(어떤 front가 같은 level에서 병렬)는 `plcols` 간접 인덱스로 충분하고,
*저장 순서*는 데이터 흐름(child→parent)을 따르는 postorder가 맞다.

## 3. 측정 방법론 교훈 (중요)

- 첫 **sequential** 측정(한 빌드 다 돌고 다른 빌드)에서 8387 B=1이 "factor −13%"로 나와
  win처럼 보였음. → **boost jitter였다.**
- 두 빌드를 **interleaved**(번갈아)로 측정하니 win이 사라지고 regression이 드러남.
- 클럭 고정이 불가한 환경(컨테이너)에선 **A/B는 반드시 interleaved + median**. 이 규약을
  doc 14 §4에 반영.

## 4. 재현 (참고)

`analyze.cu`의 `front_off` 할당 루프를 plcols 순서로 바꾸고 별도 빌드 → interleaved 비교:
```
for (int q = 0; q < P; ++q) { int p = plcols[q]; front_off[p] = acc; acc += fsz*fsz; }
```
현재 트리는 postorder(id-order)로 되돌려져 있음.

## 5. 후속 가능성 (열어둠)
- level-contiguous를 **단독 변경**으로 보면 기각이지만, 같은 level의 front들을 하나의
  **fused/batched 커널**로 처리하는 최적화의 전제로 쓰면 extend-add 손해를 상쇄+초과할 여지는 남음.
  그 커널 없이 레이아웃만 바꾸는 건 손해라는 게 이 문서의 범위.
