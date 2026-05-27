# Final Rationale: Hybrid NR with stale cuDSS-preconditioned GMRES

## One-line conclusion

최종적으로 남는 방향은 **hybrid Newton-Raphson solver + stale full-J cuDSS factorization을 전처리기로 쓰는 GMRES(1) refinement**다.

이 선택의 핵심은 다음이다.

- 중간 Newton iteration은 full cuDSS 수준으로 정확히 풀 필요가 없다.
- 하지만 반복기법 correction이 nonlinear mismatch를 충분히 줄이지 못하면 NR trajectory가 길어지거나 실패한다.
- 따라서 middle solver는 반드시 **accept/reject/fallback** 로직과 같이 써야 한다.
- cheap local preconditioner 계열은 kernel은 빠르지만 correction 품질이 부족했다.
- cuDSS factorization을 마지막 full-J step에서 재사용하면, middle step에서는 current Jacobian factorize를 하지 않고 **cuDSS solve만 전처리기로 호출**할 수 있다.
- 대규모 케이스에서는 full-J factorize 비용이 solve 비용보다 훨씬 크기 때문에, current factorize를 stale solve로 바꾸는 구조가 실제 속도 이득을 만든다.

모든 시간 비교에서 중요한 기준은 **warm cuDSS**다. 즉, full-J cuDSS analyze는 NR loop 밖에서 1회 수행하고, NR loop 안에서는 factorize+solve만 비교한다.

## Why hybrid is necessary

Newton-Raphson의 middle iteration에서 선형계 `J dx = F`를 반드시 `1e-3` relative residual까지 풀 필요는 없다. 실제 목표는 선형 residual 자체가 아니라 다음 NR 상태에서 mismatch가 줄어드는지다.

그래서 middle solver는 다음 형태여야 한다.

1. approximate `dx`를 계산한다.
2. 임시로 voltage update를 적용한다.
3. nonlinear mismatch가 충분히 줄면 accept한다.
4. 줄지 않거나 NaN/Inf/breakdown이면 full cuDSS로 fallback한다.
5. full cuDSS fallback이 발생하면 stale factor도 current Jacobian으로 refresh한다.

즉, 반복기법은 단독 solver라기보다 **middle correction generator**다. 불완전한 correction을 허용하되, NR 전체 수렴성은 fallback이 지킨다.

## Why preconditioner quality matters more than outer Krylov choice

처음에는 GMRES, MR1, BiCGSTAB 같은 outer solver를 바꾸면 해결될 수 있는지 확인했다. 결론은 아니었다. 같은 METIS block-Jacobi 전처리기에서는 outer solver를 바꿔도 correction 품질 한계가 먼저 드러났다.

### Standalone GMRES + block-Jacobi

J1/F1 standalone에서 restart 32, max 128까지 길게 돌려도 `1e-3`에 도달하지 못했다.

| case | iterations | final relative residual | converged |
|---|---:|---:|---:|
| case1197 | 128 | 2.98e-02 | no |
| case2736sp | 128 | 7.54e-02 | no |
| case3375wp | 128 | 1.90e-01 | no |
| case6468rte | 128 | 6.73e-02 | no |
| case_ACTIVSg10k | 128 | 5.95e-02 | no |

SpMV와 block-Jacobi apply 자체는 작았다. 긴 GMRES에서 눈에 띄는 비용은 dot/reduction이었고, 더 중요한 문제는 residual이 초반 이후 평탄화된다는 점이었다.

### MR1 / BiCGSTAB + block-Jacobi

BiCGSTAB(2)는 MR1보다 first middle correction 품질이 좋아졌다.

| solver | dx norm ratio vs cuDSS | dx cosine | theta ratio | middle trial ratio | linear rel res |
|---|---:|---:|---:|---:|---:|
| MR1 | 0.119 | 0.313 | 0.050 | 0.366 | 0.299 |
| BiCGSTAB(1) | 0.138 | 0.355 | 0.065 | 0.276 | 0.196 |
| BiCGSTAB(2) | 0.178 | 0.395 | 0.113 | 0.202 | 0.129 |
| BiCGSTAB(4) | 0.303 | 0.451 | 0.248 | 3.107 | 2.662 |

하지만 accepted middle trial만 보면 여전히 `dx`가 cuDSS step보다 작고, 특히 theta correction이 작다.

| solver | accepted dx norm ratio | accepted dx cosine | accepted theta ratio | accepted middle trial ratio |
|---|---:|---:|---:|---:|
| MR1 | 0.0479 | 0.277 | 0.0224 | 0.667 |
| BiCGSTAB(2) | 0.141 | 0.294 | 0.0726 | 0.225 |

BiCGSTAB(2)는 cheap middle solver 중 가장 나은 축이었지만 fallback을 충분히 줄이지 못했다. BiCGSTAB iteration을 4, 8, 16으로 늘려도 middle time만 증가했고, fallback/NR trajectory 개선은 충분하지 않았다.

이 결과는 outer Krylov method보다 **전처리기가 만드는 correction subspace가 더 중요하다**는 쪽으로 해석된다.

## Why Jacobi / block-Jacobi was not enough

METIS block-Jacobi는 diagonal dense block만 사용한다. 즉, reordered matrix에서 block 밖 coupling은 preconditioner apply에서 버린다.

METIS coupling diagnostic 결과:

| metric | mean value |
|---|---:|
| off-block NNZ ratio | 0.249 |
| off-block absolute-value ratio | 0.0787 |
| off-block Frobenius ratio | 0.205 |
| overall off-block cuDSS-dx effect ratio | 0.085 |

전체적으로 가장 큰 coupling 대부분은 block 안에 남아 있었지만, field별로 보면 cross coupling이 많이 block 밖에 있었다.

| field | off-block effect ratio |
|---|---:|
| J11, P-theta | 0.059 |
| J12, P-|V| | 0.358 |
| J21, Q-theta | 0.329 |
| J22, Q-|V| | 0.039 |

따라서 block-Jacobi는 싸지만, `J12/J21` cross term과 inter-block coupling을 충분히 반영하지 못한다. 이 손실이 theta correction 부족과 weak middle step으로 이어졌다.

Scalar Jacobi는 block-Jacobi보다 더 약한 subset이므로 이 방향에서 승산이 없었다. block-Jacobi를 최적화해서 apply를 빠르게 만들어도, 이미 병목은 apply time이 아니라 correction quality였다.

## Why ILU / block-ILU was not the final answer

기존 ILU/ILUT 계열은 품질은 올릴 수 있지만 GPU middle solver 목적에는 구조적 문제가 있었다.

- 기존 ILUT apply는 L/U SpSV를 사용한다.
- BiCGSTAB 한 iteration에서 preconditioner apply가 2회 발생한다.
- ILUT 기준으로는 한 iteration당 SpSV가 4회 발생한다.
- 이는 global triangular dependency를 만들고, 기존 병목을 다시 가져온다.

그래서 block graph ILU(0)를 별도로 진단했다. symbolic 분석에서는 block coloring ordering이 current block order보다 훨씬 얕은 level structure를 만들었다.

| block size | ordering | mean L/U levels | mean avg width | apply / block-Jacobi | factor / block-Jacobi setup |
|---:|---|---:|---:|---:|---:|
| 16 | block_coloring | 11.2 / 11.2 | 67.2 | 7.40 | 9.84 |
| 32 | block_coloring | 8 / 8 | 46.8 | 7.39 | 9.45 |

Numeric block ILU(0) pilot에서는 품질 개선은 확인됐다.

| candidate | standalone quality gate | hybrid fallback result |
|---|---:|---|
| block ILU(0), bs16 | pass 5/5 | fallback decreased 3/5 |
| block ILU(0), bs32 | pass 3/5 | fallback decreased 3/5 |

하지만 pilot 구현은 CPU dense FP32였고, 비용은 block-Jacobi보다 훨씬 컸다.

| preconditioner | avg setup ms | avg solve ms | avg middle total ms |
|---|---:|---:|---:|
| block-Jacobi bs16 | 1.55 | 0.71 | 0.31 |
| block ILU(0) bs16 CPU pilot | 26.84 | 4.05 | 30.56 |

이 결과는 두 가지를 동시에 말한다.

- off-diagonal/inter-block 정보를 넣으면 correction 품질은 좋아진다.
- 하지만 ILU류를 middle solver로 쓰려면 triangular dependency와 setup/apply cost를 매우 잘 제어해야 한다.

따라서 block ILU는 “전처리기 품질이 중요하다”는 근거는 되었지만, 현 시점의 최종 선택은 아니다.

## Why cuDSS as a preconditioner

최종 방향은 full-J cuDSS를 매 iteration factorize하는 대신, 마지막 full-J factorization을 stale preconditioner `M`으로 재사용하는 것이다.

Current system:

```text
J_k dx = F_k
```

Stale factor:

```text
M = J_s
```

여기서 `J_s`는 마지막으로 full cuDSS factorize가 성공한 Jacobian이다.

Middle step:

```text
dx0 = M^{-1} F_k
r0  = F_k - J_k dx0

GMRES(1) solves:
    J_k delta = r0
with right preconditioner:
    z = M^{-1} v

dx = dx0 + delta
```

중요한 점은 middle step에서 current `J_k`를 factorize하지 않는다는 것이다. Current `J_k`는 SpMV/residual 계산에만 쓰고, expensive factorization은 하지 않는다.

비용 구조는 다음처럼 바뀐다.

| path | middle linear work |
|---|---|
| pure cuDSS | current `J_k` factorize + solve |
| stale GMRES(1) | stale solve for predictor + current-J SpMV/dot + stale solve as preconditioner |

즉, 이 방법은 **factorize를 solve로 대체하는 효과**가 있다.

대규모 케이스에서는 full-J factorize가 solve보다 훨씬 비싸다. 그래서 stale solve를 2번 하더라도 current factorize를 피하면 이득이 난다. 반대로 작은 케이스나 trajectory가 길어지는 케이스에서는 stale solve 2회와 extra NR iteration이 이득을 지워버린다.

## Warm cuDSS baseline

아래 pure cuDSS 수치는 full-J analyze를 NR loop 밖에서 1회 수행한 warm 기준이다.

| case | NR iters | avg factor ms | avg solve ms | avg factor+solve ms |
|---|---:|---:|---:|---:|
| case2383wp | 6 | 2.324 | 0.673 | 2.997 |
| case3120sp | 6 | 0.411 | 0.226 | 0.637 |
| case9241pegase | 6 | 0.926 | 0.355 | 1.281 |
| case13659pegase | 5 | 1.152 | 0.444 | 1.596 |
| case6468rte | 3 | 0.557 | 0.295 | 0.852 |

따라서 replacement middle step과 비교해야 하는 것은 cold analyze 포함 시간이 아니라 `factor+solve` warm time이다.

## Final stale GMRES(1) result

Clean stale GMRES(1) run:

- 11/11 cases converged.
- NR iterations were within pure cuDSS + 1 on 9/11 cases.
- stale solve count matched the intended algorithm.
- Current-J SpMV was not the bottleneck.
- Bottleneck was stale cuDSS solve plus host-synchronized dot/nrm2.

Representative case-level results:

| case | pure warm NR-loop ms | stale GMRES(1) NR-loop ms | speedup | pure NR | stale NR | full cuDSS calls |
|---|---:|---:|---:|---:|---:|---:|
| case8387pegase | 4.963 | 4.862 | 1.021 | 3 | 3 | 2 / 3 |
| case_ACTIVSg25k | 12.609 | 11.326 | 1.113 | 4 | 4 | 2 / 4 |
| case_ACTIVSg70k | 41.648 | 32.334 | 1.288 | 6 | 6 | 2 / 6 |
| case_SyntheticUSA | 43.886 | 39.467 | 1.112 | 6 | 7 | 2 / 6 |
| case9241pegase | 9.940 | 24.452 | 0.407 | 6 | 16 | 2 / 6 |
| case13659pegase | 10.092 | 20.204 | 0.500 | 5 | 11 | 2 / 5 |

이 결과는 방향을 명확히 보여준다.

- 큰 ACTIVS 계열과 SyntheticUSA에서는 full factorize를 줄이는 효과가 실제 NR-loop speedup으로 이어졌다.
- PEGASE 일부 케이스에서는 stale factor trajectory가 길어져서 손해가 컸다.
- 따라서 stale GMRES는 unconditional replacement가 아니라 hybrid path여야 한다.

## Why not replace the second stale solve with block-Jacobi?

`stale_BJ1`도 테스트했다.

Idea:

```text
dx0 = M^{-1} F_k
r0  = F_k - J_k dx0
delta = block_jacobi_correction(r0)
dx = dx0 + delta
```

이 방식은 middle step에서 stale solve를 1회만 쓰므로 per-middle cost는 조금 줄었다.

| mode | accepted middle steps | median middle ms | median stale solve calls | correction cost | median mismatch ratio |
|---|---:|---:|---:|---|---:|
| stale GMRES(1) | 41 | 0.843 | 2 | 0.374 ms stale solve | 0.241 |
| stale BJ1 | 50 | 0.808 | 1 | 0.013 ms BJ apply | 0.276 |

하지만 total NR-loop에서는 `stale_BJ1`이 거의 전부 졌다. 이유는 correction이 약해서 hard case에서 NR trajectory가 길어지고 fallback이 생겼기 때문이다.

이 실험은 최종 논리를 더 강하게 만든다.

> middle step의 kernel time을 조금 줄이는 것보다, nonlinear mismatch를 충분히 줄이는 correction 품질이 더 중요하다.

## Final interpretation

지금까지의 실험 흐름은 다음으로 정리된다.

1. ILU/ILUT는 global triangular solve 때문에 GPU middle solver로 너무 무겁다.
2. Jacobi/block-Jacobi는 빠르지만 off-diagonal/inter-block coupling을 잃어 correction이 너무 약하다.
3. GMRES나 BiCGSTAB iteration을 늘리는 것만으로는 이 품질 한계를 넘지 못했다.
4. Block ILU(0)는 품질 개선 가능성을 보였지만, factor/apply work와 dependency가 커서 아직 최종 경로가 아니다.
5. Field correction, B'/B'', RAS, scaling, coarse correction은 일부 진단적 성과는 있었지만 pure cuDSS를 안정적으로 이기지 못했다.
6. stale full-J cuDSS factorization은 강한 전처리기 역할을 하며, middle step에서 current factorize를 stale solve로 대체한다.
7. 이 구조는 factorize가 solve보다 훨씬 비싼 대규모 케이스에서 실제 speedup을 만든다.
8. 단, stale factor가 오래되면 trajectory가 길어질 수 있으므로 fallback/refresh가 필수다.

따라서 최종 선택은:

```text
Hybrid NR
  first / fallback / polish:
      full-J cuDSS factorize + solve, then refresh stale factor

  middle:
      stale full-J cuDSS factor as preconditioner
      GMRES(1) refinement with current-J residual/SpMV
      accept only if nonlinear mismatch decreases enough
      otherwise fallback to full-J cuDSS
```

## What should be claimed

Claim:

- Hybrid middle solve can be approximate, but must be guarded by mismatch-based accept/reject/fallback.
- Cheap local preconditioners are not enough for robust NR progress.
- The limiting factor is preconditioner quality, not raw SpMV speed.
- cuDSS-as-preconditioner is attractive because it reuses a strong factorization and avoids current-J factorization in middle iterations.
- Large cases benefit when factorize cost dominates solve cost.

Do not overclaim:

- This is not a universal speedup on every case.
- It does not mean GMRES alone is a good standalone linear solver.
- It does not mean block-Jacobi is useless; it is useful as a cheap baseline and diagnostic, but not strong enough as final middle correction.
- End-to-end speedup is not “factorize/solve ratio”; measured NR-loop speedups on the best large cases were roughly 1.1x to 1.3x in the current runs.

## Source reports used

- `results/gmres_representative_j1_long_best.md`
- `results/bicgstab_block_jacobi_report.md`
- `results/bicgstab_bj_4k_reuse_report.md`
- `results/metis_coupling_drift_report.md`
- `results/block_ilu_symbolic_report.md`
- `results/block_ilu0_report.md`
- `results/pure_cudss_preanalyze_report.md`
- `results/stale_prec_gmres_report.md`
- `results/stale_gmres1_clean_report.md`
- `results/stale_bj1_compare_report.md`
