# Cleanup 중단 + front-distribution 스케줄링 정책 레이어 계획 (2026-06-10)

**상태**: 진행 중 작업의 status/decision 로그. 소스 cleanup 은 **중단**, 다음 방향은 정책 레이어 설계.

---

## 1. 지금까지 커밋된 것 (branch `integrate/tc-speedup-ozaki`, master 기준 +6 커밋)

| 커밋 | 내용 |
|---|---|
| `bde6774` | Codex TC-speedup + Ozaki 연구 노트(31–52) 전부 import + 색인 |
| `b7f71eb` | accepted 기능만 통합(TF32 TC dispatch + Ozaki), 실패 29옵션 제거, x3 제거, FP16-mid-TC 제외 |
| `4225c98` | 실패 실험 dead code 물리 삭제(unifdef ~1060줄) |
| `a6bac6a`/`4ef2a25`/`cc5f92b` | storyline.md + optimal-configuration.md (메소드 중심 재작성, tiny-front regime 척추) |

이 상태가 현재 baseline. accepted 기능은 CMake 옵션(default OFF) 토글로 보존돼 있음.

## 2. 시도한 cleanup 과 왜 중단했나

**목표였던 것**: 최적 경로 bake-in, 정밀도 옵션을 `{fp64, fp32, TC}` 3개로 축소, ablation/실패/dead
code 삭제(→ `deprecated/`).

**중단 이유**: **케이스별로 최적 알고리즘이 갈린다.** 단순히 옵션을 bake 하면 한 케이스에 최적인 설정이
다른 케이스를 망친다. n-threshold 로 가르려 했으나(저-fill auto-gate) 아래 §3 의 발견으로 그 접근이
틀렸음이 드러남 — **입력 크기 n 이 아니라 front 분포로 결정해야 하고, 그건 별도 정책 레이어가 필요**.
지금 bake 하는 것은 시기상조.

(시도한 partial cleanup — universal TC bake, drop strip, amalgamation `tc_path && n<16k` runtime gate —
은 working tree 에서 폐기. 재개 시 git 이력 + 본 노트 참조.)

## 3. 이번 세션의 핵심 실측 발견 (정책 설계의 근거)

### 3.1 TC win 의 분해 (experiment #3)
large case 의 tf32-vs-fp32 이득을 "커널 형태"와 "텐서코어 mma"로 분리(같은 staging/thread, mma 만
scalar 치환):

| case (B64) | 총(fp32/tf32) | = kernel-shape | × **TC mma** |
|---|---|---|---|
| 25K | 1.254× | 1.149× | **1.092×** |
| USA | 1.186× | 1.114× | **1.064×** |
| 70K | 1.207× | 1.124× | **1.074×** |

→ **TC mma 자체는 이득의 ~38%(+6~9%), 나머지 ~62%는 커널 형태(엔지니어링)**. note 28(thin-K
memory-bound)과 정합.

### 3.2 cap confound — note 50 의 "8387 1.24× TC pass" 는 비율 artifact
- note 50 은 8387 을 **cap30** 에서 측정 → fp32 0.0332 / tf32 0.0267 = **1.24×**.
- 그러나 cap30 은 amalgamation 의 fill 을 키워 **절대 성능을 악화**시킴. **자연 cap** 에서:
  - 8387: low-fill tf32 **0.0220** vs large tf32 0.0236 (절대 −7%), 그러나 fp32 0.0246 → 진짜 ratio
    **1.12×** (목표 1.2× 미달).
  - 13K: low-fill tf32 **0.0341** vs large 0.0377 (−10%), ratio **1.09×**.
- **결론**: low-fill 은 자연 cap 에서 절대 성능은 빠르지만, "1.2× TC ratio" 헤드라인은 cap 선택으로
  부풀린 것. 절대 가속(−7~10%)은 real.

### 3.3 low-fill 은 깨지기 쉬운 4-플래그 번들
amalgamation + small16(`kSmallFrontMax 16`) + mid128 + low_tc_force_all 을 **전부 함께** 켜야 win.
부분집합은 large 정책보다 **느림**(실측):

| 8387 tf32 (자연 cap) | 13K tf32 |
|---|---|
| full bundle **0.0220** ✓ | **0.0341** ✓ |
| large only 0.0236 | 0.0377 |
| amalgamation only 0.0251 ✗ | 0.0420 ✗ |
| amalgamation+small16 0.0249 ✗ | 0.0437 ✗ |

또한 small16 무조건 적용 시 **25K 악화**(tf32 0.103 vs 0.092; amalgamation 적용 시 relres 0.075→cap31
0.118). → low-fill 은 **작은 케이스 한정**이며 large 에 적용하면 해롭다.

### 3.4 column-U-solve 는 무효
note 29 의 61% 병목(panel-LU+U-solve barrier)을 겨눴으나 **효과 ≈0%**(fp32 에 동일 적용해도 −0.4~+0.9%
노이즈). negative result.

## 4. 다음 방향 — front-distribution 스케줄링 정책 레이어

n-threshold 가 아니라 **symbolic 직후 실제 front 분포**로 경로를 정한다:

```
reorder → symbolic → [front 분포 feature 추출] → policy(features, B, precision) → config
                                                                                   ↓
                                          factorize/solve 커널 스케줄링 (선택된 config 실행)
```

### 4.1 정책 입력 feature (post-symbolic)
- tier별 front 수 / **work(FLOP) 비중**(small/mid/big)
- **nc(=K) 분포** + TC-routable work%(nc≥4, uc≥16)
- trailing vs panel-LU/extend work 비중(61/19.5/19.2 가 케이스마다 다름)
- etree depth / spine 길이 / subtree 균형(multistream 예측)
- max_fsz, mean_fsz, fill ratio

### 4.2 배치 크기 B 는 1차 축 (포함 결정)
병목이 B 로 이동(B=1 launch/occupancy bound, B≥64 throughput bound; saturation point 가 tier 마다 다름,
doc 16). TC fill 도 B 의존. → policy 시그니처 = **`policy(front_features, B, precision) → config`**.

### 4.3 정책 도출 sweep (설계만, 미실행)
- **데이터**: 78 MATPOWER 케이스의 NR 선형시스템 생성 필요(현재 ~9개만 존재; 파이프라인
  `prepare_datasets/python/prepare_nr_linear_system.py`). 버킷: ~1k / ~3k / ~6k / ~10k / large(25k/70k/USA).
- **축**: case(버킷) × config{fp32/TC × large/low-fill × multistream} × B{1,4,16,64,256}.
- **측정**: factor_ms/sys, solve_ms/sys, relres.
- **도출**: 각 (case, B) winning config → feature 로 회귀/결정트리 → 임계값 규칙
  (예: "TC-routable work% > τ AND B ≥ b → low-fill, else large").

## 5. 재개 시 체크리스트
- cleanup(3-정밀도 bake)은 **정책 레이어가 정해진 뒤**에 — 그래야 무엇을 bake/auto-decide 할지 명확.
- low-fill 은 삭제하지 말 것(작은 케이스에서 −7~10% real). 단 "언제 켤지"는 정책이 결정.
- storyline 의 B2(low-fill) 서술은 "절대 −7~10% real, 단 ratio 1.2× 는 cap-inflated, 4-플래그 fragile,
  작은 케이스 한정"으로 정정 필요(§3.2/3.3).
