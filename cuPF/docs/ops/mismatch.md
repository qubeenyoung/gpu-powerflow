# ops/mismatch — 잔차 F와 수렴 norm

NR의 잔차(mismatch)와 수렴 판정 norm을 계산한다. Ibus([ops/ibus.md](ibus.md)) 다음
단계.

**소스**: `cpp/src/newton_solver/ops/mismatch/` (`compute_mismatch_from_ibus.cu`,
`reduce_mismatch_norm.cu`, `cuda_mismatch.cu`(디스패치), `cpu_mismatch.{cpp,hpp}`).

---

## 1. 잔차 F (`compute_mismatch_from_ibus.cu`)

주입 전력 `S(V) = V·conj(Ibus)`, 잔차 `F = S(V) − Sbus`. 실부=유효전력 P, 허부=무효전력 Q.

```
Re(S) = vr·ir + vi·ii   (P),   Im(S) = vi·ir − vr·ii   (Q)
```

잔차 벡터는 `[dP@pv | dP@pq | dQ@pq]` (길이 `dimF = n_pv + 2·n_pq`) — PV 버스는 P(각도)
행만, PQ 버스는 P·Q 모두. 한 스레드가 한 잔차 엔트리를 맡고, 인덱스에서 (case,bus)와
P/Q 여부를 푼다. 세 프로파일은 템플릿 헬퍼 `launch_mismatch_impl<Scalar,Storage>`를
공유한다(스칼라만 다름).

## 2. 수렴 norm (`reduce_mismatch_norm.cu`)

케이스별 L∞ norm `max|F|`를 device에서 구한다. 2단계 리덕션(글로벌 스크래치 없음):
블록이 자기 케이스 잔차를 grid-stride로 부분 max → shared memory 트리 리덕션 →
블록 partial을 `normF[case]`에 **atomicMax**. `gridDim.y = batch_size`, `gridDim.x`는
적응형(배치가 작을 때만 케이스당 블록 여럿 → atomicMax 경합 최소). 비음수 float/double의
IEEE 비트패턴 단조성을 이용해 정수 atomicMax로 부동소수 max를 구한다.

## 3. 디스패치 (`cuda_mismatch.cu`)

`CudaMismatchOp::run`(잔차)과 `CudaMismatchNormOp::run`(norm)을 프로파일별로 노출하되,
각각 템플릿 헬퍼 `run_compute_mismatch<Storage>` / `reduce_norm_into_ctx<NormScalar,Storage>`
로 본문을 공유한다. norm 단계는 케이스별 norm을 호스트로 가져와 **worst-case를 배치
norm**으로 삼고(전 케이스가 수렴해야 배치 수렴), non-finite면 던진다(FP32 발산 조기
표면화). 덤프가 켜지면 잔차를 기록.

## CPU (`cpu_mismatch.cpp`)

호스트에서 동일한 F/Norm 계산(`CpuMismatchOp`).
