# 07 — solve (삼각 대입)

> **층위**: 추상 + 상세. 분해된 `L`,`U`로 `x = A⁻¹b`를 푸는 경로. 표준 전진/후진 대입 → GPU 매핑 → B=1 특수화(selinv).
> 진입점 `src/solve/solve.cu`, 디스패치 `dispatch.cuh`, 프리미티브 `phases.cuh`.

---

## 1. 추상 — 두 번의 삼각 solve

`A = LU`면 `Ax=b`는 두 단계다.

```
1. 전진 대입(forward):  L y = b   (leaf→root, 아래에서 위로)
2. 후진 대입(backward): U x = y   (root→leaf, 위에서 아래로)
```

multifrontal에선 각 front가 자기 pivot 변수에 대한 작은 삼각 블록을 들고 있고, contribution 결합이 factor 때의
extend-add와 대칭(transpose)으로 일어난다.

- **전진**: front의 pivot 부분으로 `L_pp y_pivot = b_pivot`를 풀고, 그 결과를 contribution 행에 전파해 부모 RHS를 갱신.
- **후진**: 부모에서 내려온 해를 contribution 열로 빼낸 뒤 `U_pp x_pivot = rhs`를 푼다.

## 2. 입출력 치환 — gather / scatter (`permute.cuh`)

solve는 분해와 같은 **재정렬 좌표계**에서 돌아야 한다.

```
GatherRhs:        b[원래 순서]  ──(perm)──▶  d_y_batch[정렬 순서]
(레벨별 전진/후진 대입은 d_y_batch 위에서 in-place)
ScatterSolInverse: d_y_batch[정렬 순서] ──(iperm)──▶ x[원래 순서]
```

즉 한 번의 solve = `gather(b) → 레벨 전진 → 레벨 후진 → scatter(x)`. 이 전체가 하나의 graph로 캡처된다(§6).

## 3. per-front 단계 (`phases.cuh`)

| 방향 | 단계 | 연산 |
|---|---|---|
| 전진 | `FwdSubstitute` | warp-parallel로 `L_pp · y_pivot = rhs` (nc ≤ 32) |
| 전진 | `FwdCbUpdate` | contribution 행 갱신: `y[부모행] −= L21 · y_pivot` (형제 누적이라 `atomicAdd`) |
| 후진 | `BwdLoadRhsAndX` | pivot rhs와 contribution 해를 shared로 적재 |
| 후진 | `BwdCbSubtract` | `rhs −= U12 · x_cb` (contribution 열 기여 제거) |
| 후진 | `BwdSubstitute` | warp-parallel로 `U_pp · x = rhs` (back-substitution) |

전진의 `FwdCbUpdate`가 factor의 extend-add와 같은 구조(부모로 atomicAdd)인 게 핵심 대칭이다.

## 4. B=1 특수화 — selinv(partitioned inverse) → 병렬 GEMV (`solve/single.cuh`)

B=1에선 삼각 역대입의 직렬 의존이 그대로 latency가 된다. 그래서 factor 직후 **pivot 블록을 미리 역행렬화**해
두고(`FactorSingleInvertPivot`, [08 §4](08-runtime-and-batching.md)), solve를 역대입 대신 **GEMV**로 바꾼다.

```
일반(B>1):  L_pp y = rhs   (삼각 전진대입, 직렬 의존)
B=1(selinv): y = L_pp⁻¹ · rhs   (밀집 GEMV, 병렬)   ← SolveSingleFwd
            x = U_pp⁻¹ · rhs   (밀집 GEMV, 병렬)   ← SolveSingleBwd
```

`SolveSingleFwd`는 `Linv @ rhs` 후 CB 행을 부모로 atomicAdd, `SolveSingleBwd`는 contribution 열 기여를 병렬
reduction으로 뺀 뒤 `Uinv @ rhs`. 직렬 삼각 의존이 사라져 latency가 짧다. B=1이 다른 이유의 전체 그림은 [08 §4](08-runtime-and-batching.md).

## 5. 배치 디스패치 (`dispatch.cuh`)

`IssueSolveLevels`가 B>1 경로를 발행한다. factor와 같은 골격이다.

- **tier 라우팅**: small tier는 sub-group packing(`SolveFwdSmall`, sub-group 크기는 `max_nc`로 — 대입엔 nc lane만 필요),
  mid/big은 block-per-front. 좁은 레벨용 점유 게이트도 동일(`SolveSmallPacks`).
- **staged vs non-staged**: L 패널이 shared 예산에 맞으면 coalesced로 shared 적재, 크면 global 직독.
- **fixed-NC / exact-NC 특수화**: 한 구간의 front가 모두 같은 `nc`(흔히 8/14/16/20)면 컴파일타임 unroll 커널
  (`SolveFwdFixed<NC>`/`SolveBwdExactNc<NC>`)로 루프 오버헤드 제거.
- **멀티스트림/단일스트림 sweep**: 독립 서브트리를 스트림에 fork(전진), spine은 main 스트림, 후진을 위해 재-fork.

## 6. graph 캡처 / 캐시 (`solve.cu`)

내부 graph 모드(`CLS_INTERNAL_GRAPH`)면 전체 solve(`gather + 레벨 + scatter`)를 **하나의 graph로 lazy 캡처**하고
`(d_rhs, d_solution, d_perm, d_iperm, type)`를 키로 캐시한다. 같은 포인터·타입이면 재캡처 없이 `cudaGraphLaunch`만.
NR 루프는 보통 같은 버퍼를 재사용하므로 첫 solve에서 한 번 캡처하고 이후엔 replay다. 외부 모드면 캡처 없이 caller
스트림에 직접 발행([08 §2](08-runtime-and-batching.md)).

## 7. solve 코드 변종 (요약)

| 표준 연산 | 코드 변종 | 차이 |
|---|---|---|
| 전진 substitution | `SolveFwd` / `SolveFwdSmall<SG>` / `SolveFwdFixed<NC>` | tier packing · 고정-nc unroll |
| 후진 substitution | `SolveBwd` / `SolveBwdSmall<SG>` / `SolveBwdFixed<NC>` / `SolveBwdExactNc<NC>` | 동일 |
| B=1 GEMV | `SolveSingleFwd` / `SolveSingleBwd` | 역행렬화된 pivot |

[06 §6](06-factorization.md)의 factor 변종과 같은 동기(점유·barrier·정밀도)다.
