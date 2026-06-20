# 05 — front tier 경계

> **층위**: 상세. front 크기로 커널을 고르는 규칙과 그 경계(small|mid=32, mid|big=64 fp64 / 128 float)의 근거.
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
inline FrontTier ClassifyFrontTier(int front_size, bool fp64) {
  if (front_size <= kSmallFrontMax) return FrontTier::kSmall;  // <= 32
  if (front_size <= (fp64 ? kMidFrontMax          // <= 64  (double)
                          : kMidFrontMaxFloat))    // <= 128 (float)
    return FrontTier::kMid;
  return FrontTier::kBig;
}
```

mid|big 경계가 **정밀도 의존**이다: double front는 64에서, float(FP32/TF32) front는 128에서 끊는다(아래 §3).

| tier | front 크기 | 커널(`src/factorize/`) | 매핑 |
|---|---|---|---|
| **small** | `fsz ≤ 32` (`kSmallFrontMax = kWarpSize`) | `FactorSmall` (`small.cuh`) | warp당 1 front(sub-group 8/16/32 lane), 8 warp/block, `__syncwarp`만 |
| **mid** | `fsz ≤ 64`(fp64) / `≤ 128`(float) (`kMidFrontMax`/`kMidFrontMaxFloat`) | `FactorMid` (`mid.cuh`) | front 전체 shared, 1 block/front |
| **big** | 그 외 | `FactorBigPivot`→`Panels`→`Trail`[`Tf32`] (`big.cuh`) | global 상주, 한 front를 여러 block에 분산하는 3-launch 트리플(TF32 trailing은 텐서코어) |

## 3. 경계 근거 (둘 다 물리적)

### small | mid = 32 (워프 폭)
sub-group packing(8/16/32 lane을 한 front에)은 **워프 하나 안에서만** 성립한다. fsz ≤ 32면 한 front가 워프에 들어가
`__syncwarp`(~1 cyc)로 동기화되고 block barrier가 없다. 32를 넘으면 워프 경계를 넘어 packing이 깨지고, block-per-front로
가야 하는데 — 그건 수만 개 leaf front에서 ¾ idle thread + 매 pass block barrier라 비효율. 그래서 워프 폭이 자연 경계다.

### mid | big = 64 (fp64) / 128 (float) — shared-residency가 이기는 한계
mid 커널은 **front 전체(`fsz²` 원소)를 shared에 staging**해 한 launch로 분해한다. 경계는 두 가지가 정한다:

1. **shared 용량**: `fsz²·elem`이 99 KiB opt-in shared에 들어가야 한다 → float은 fsz≤159, double은 ≤111까지 가능.
2. **shared-resident mid vs 3-launch big triple 중 어느 게 빠른가**(측정). big triple은 한 front를 pivot/panels/trailing
   3개 커널로 쪼개 레벨당 ~3 launch를 내는데, 한 레벨에 big front가 몇 개뿐이면 각 launch가 저점유·latency 바운드다.
   **배치(B>1)**에선 batch가 occupancy를 채우므로, moderate front(65~128)를 shared-resident mid 단일 launch로 합치는
   게 더 빠르다(2026-06-20 측정 **~-1~11% factor 시간**; DRAM 바이트는 거의 불변 → launch 통합 + shared 재사용 +
   occupancy 이득, 대역폭 절감 아님). 정밀도 무관(FP32·TF32 동일), 텐서코어와 무관.

그래서 **float은 128**(shared 64 KiB, 빠름), **double은 64**(128² double = 128 KiB > 예산이라 더 못 올림)에서 끊는다.
128 초과 float / 64 초과 double 은 big tier(global multi-block)로 간다.

## 4. 159/111 = 절대 shared-residency 상한 (mid 경계와 구분)

`WholeFrontSharedMax` = 99 KiB opt-in shared(`kDynamicSharedMemoryOptInBytes`)에 `fsz²` dense staging이 들어가는
최대 fsz = **159(float) / 111(double)**. 이건 **shared에 통째로 올릴 수 있는 절대 상한**이다.

- mid 경계(float 128)는 이 상한(159)보다 **낮게** 잡았다 — 128 초과에선 occupancy(64 KiB→1 block/SM 부근)가 더 떨어져
  big multi-block이 균형상 낫기 때문. 즉 "shared에 들어가는 최대(159)"와 "mid로 보내는 게 빠른 한계(128)"는 다르다.
- float 128 초과 / double 64 초과는 big tier(global multi-block). 159(float)/111(double)을 넘는 거대 separator는 big
  커널 내부에서 tiled bounded-shared로 떨어진다([06 §4](06-factorization.md)).

## 5. 이력 (현재는 3-tier, mid 경계는 precision-aware)

- 옛 4-tier: big을 panel-resident(65–111)와 large=global(>111)로 쪼갰다. 2026-06-18에 panel-resident가 배치에서 ~1.2%뿐,
  65–111을 global multi-block으로 보내는 게 **B=1**에서 16% 빨라 **3-tier로 통합**했다.
- 2026-06-20: 그 통합은 B=1 기준이었다. **배치(B>1)**에선 65–128 float front를 shared-resident mid 단일 launch로 합치는
  게 ~1–11% 빨라(launch 통합+shared 재사용+occupancy; §3), float mid 경계를 64→128로 올렸다(fp64는 64 유지). 즉 mid|big은
  더 이상 고정 64가 아니라 정밀도 의존. 경위: [`../_legacy/05-reports/10-tier-consolidation-2026-06-18.md`](../_legacy/05-reports/10-tier-consolidation-2026-06-18.md).

> `kNumFrontBuckets = 3`. analyze-time tier 버킷팅(`FrontBucket`)은 `ClassifyFrontTier`와 **정확히 일치해야** 한 레벨의
> 동질 tier 구간이 단일 커널로 라우팅된다(`types.hpp` 주석의 "must match" 결합).
