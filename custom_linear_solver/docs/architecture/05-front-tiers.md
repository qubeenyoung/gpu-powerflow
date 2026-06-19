# 05 — front tier 경계

> **층위**: 상세. front 크기로 커널을 고르는 규칙과 그 경계가 왜 32/64인지의 하드웨어 근거.
> 단일 진실원: `src/internal/types.hpp`.

---

## 1. 추상 — 왜 tier가 있는가

front가 작을수록 GPU에 다르게 매핑해야 효율적이다.

- **아주 작은 front 수만 개**(leaf): 하나에 한 block을 주면 대부분 thread가 idle이고, 매 step마다 block barrier를
  문다 → latency/점유 바운드. **여러 front를 한 워프 안에 packing**해야 GPU가 찬다.
- **중간 front**: 한 front를 block 하나가 통째로 shared에 올려 제자리 분해하는 게 가장 효율적.
- **크고 적은 front**(separator, 큰 도메인): 한 front를 여러 block에 **분산**해야 GPU를 채운다. 너무 크면 shared에
  통째로 못 올린다.

그래서 front 크기를 셋으로 나눠 각자 전용 커널로 보낸다. 라우팅은 **front 크기만의 결정적 함수**다(점유 휴리스틱
게이트가 어느 커널을 고를지 정하지 않는다 — 단, small tier에는 좁은 레벨용 점유 게이트가 하나 붙는다. [06 §5](06-factorization.md)).

## 2. 분류기 (`ClassifyFrontTier`, `types.hpp`)

```cpp
inline FrontTier ClassifyFrontTier(int front_size, bool /*fp64*/) {
  if (front_size <= kSmallFrontMax) return FrontTier::kSmall;  // <= 32
  if (front_size <= kMidFrontMax)   return FrontTier::kMid;    // <= 64
  return FrontTier::kBig;                                      //  > 64
}
```

| tier | front 크기 | 커널(`src/factorize/`) | 매핑 |
|---|---|---|---|
| **small** | `fsz ≤ 32` (`kSmallFrontMax = kWarpSize`) | `FactorSmall` (`small.cuh`) | warp당 1 front(sub-group 8/16/32 lane), 8 warp/block, `__syncwarp`만 |
| **mid** | `33 ≤ fsz ≤ 64` (`kMidFrontMax`) | `FactorMid` (`mid.cuh`) | front 전체 shared, 1 block/front |
| **big** | `fsz > 64` | `FactorBigPivot`→`Panels`→`Trail`[`Tf32`] (`big.cuh`) | global 상주, 한 front를 여러 block에 분산하는 3-launch 트리플(TF32 trailing은 텐서코어) |

## 3. 경계 근거 (둘 다 물리적)

### small | mid = 32 (워프 폭)
sub-group packing(8/16/32 lane을 한 front에)은 **워프 하나 안에서만** 성립한다. fsz ≤ 32면 한 front가 워프에 들어가
`__syncwarp`(~1 cyc)로 동기화되고 block barrier가 없다. 32를 넘으면 워프 경계를 넘어 packing이 깨지고, block-per-front로
가야 하는데 — 그건 수만 개 leaf front에서 ¾ idle thread + 매 pass block barrier라 비효율. 그래서 워프 폭이 자연 경계다.

### mid | big = 64 (whole-front shared 점유 교차점)
mid 커널은 **front 전체(`fsz²` 원소)를 shared에 staging**한다. shared 사용량이 `fsz²·elem`이라 fsz가 커지면 SM당
점유(동시 block 수)가 급감한다. 측정상 **64를 넘으면 점유가 ~2 block/SM 아래로** 떨어져, 차라리 front를 global에 두고
작은 타일만 staging하며 여러 block에 분산하는 big 커널이 빠르다. 그래서 64가 교차점이다.

## 4. 159/111은 tier 경계가 아니다 (자주 헷갈리는 점)

`WholeFrontSharedMax(fp64)` = 99 KiB opt-in shared(`kDynamicSharedMemoryOptInBytes`)에 `fsz²` dense staging이 들어가는
최대 fsz = **159(float) / 111(double)**. 이건 **big 커널 내부의 bounded-shared staging 상한**일 뿐 **tier 경계가
아니다**. tier 경계는 어디까지나 32 / 64다.

- mid|big을 64로 끊는 건 **점유**(2 block/SM) 기준이지 shared **용량**(159까지 들어감) 기준이 아니다.
- big 커널은 65~159 fronts를 global multi-block으로 처리하고, 159(float)/111(double)을 넘는 거대 separator는
  tiled bounded-shared로 떨어진다([06 §4](06-factorization.md)).

## 5. 이력 (현재는 3-tier)

옛날엔 4-tier였다: big을 panel-resident(65–111)와 large=global(>111)로 더 쪼갰다. 2026-06-18에 측정 결과
panel-resident 티어가 배치에서 ~1.2%뿐이고 65–111을 global multi-block으로 보내는 게 B=1에서 16% 빨라
**3-tier로 통합**했다("large"→"big" 개명). 경위: [`../_legacy/05-reports/10-tier-consolidation-2026-06-18.md`](../_legacy/05-reports/10-tier-consolidation-2026-06-18.md).

> `kNumFrontBuckets = 3`. analyze-time tier 버킷팅(`FrontBucket`)은 `ClassifyFrontTier`와 **정확히 일치해야** 한 레벨의
> 동질 tier 구간이 단일 커널로 라우팅된다(`types.hpp` 주석의 "must match" 결합).
