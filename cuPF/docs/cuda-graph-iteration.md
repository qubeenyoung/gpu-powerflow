# CUDA Graph capture of the Newton iteration

전체 Newton 반복(ibus → mismatch → jacobian → 선형계 풀이 → voltage update)을 하나의
CUDA 그래프로 캡처해 매 스텝 `cudaGraphLaunch` 한 번으로 재생한다. 반복당 ~10개의 호스트
커널 런치가 1회 그래프 런치로 합쳐져 런치 지연(latency-bound 구간)을 제거한다.

## 활성화

- 빌드: `-DCUPF_ENABLE_CUSTOM_SOLVER=ON -DCUPF_ENABLE_CUDA_GRAPH=ON`
  - `CUPF_ENABLE_CUDA_GRAPH`는 `CUPF_ENABLE_CUSTOM_SOLVER`를 요구한다(아래 "custom 전용" 참고).
  - 이 옵션이 켜지면 번들된 `custom_linear_solver`가 **`CLS_INTERNAL_GRAPH=OFF`**(external/
    capturable 모드)로 빌드된다 — 솔버가 자체 내부 그래프 대신 호출자 스트림에 raw 커널을
    발행하므로 cuPF의 바깥 캡처에 그대로 기록된다.
- 런타임: `NewtonOptions.use_cuda_graph = True` (Python: `options.use_cuda_graph = True`).

## custom 솔버 전용인 이유

cuDSS의 `cudssExecute`는 host가 오케스트레이션하는 불투명 호출이라 stream capture가
불가능하다. custom 선형 솔버는 순수 커널 런치라서, 내부 그래프/host-sync를 끈 external 모드
에서는 cuPF의 iteration 그래프에 포함시킬 수 있다. 따라서 `use_cuda_graph=true`는
`backend=CUDA` + `cuda_linear_solver=Custom` 조합에서만 허용되며, 그 외에는 생성자에서
명확한 예외를 던진다.

## 루프 구조 (residual-at-bottom)

데이터 의존 host 제어흐름은 수렴 검사(잔차 norm의 D2H 복사 + 비교)뿐이다. 이를 그래프
**밖**에 두기 위해 루프를 "잔차를 끝에서 계산하는" 형태로 회전한다:

```
pre-loop (eager, 1회):  ibus → mismatch → norm   (초기 V의 잔차)  → 수렴이면 종료(스텝 없음)
graph body (캡처):       jacobian → prepare_rhs → factorize → solve → voltage_update
                         → ibus → mismatch → norm(device)         (1 Newton 스텝 + 다음 잔차)
매 replay:               graphLaunch → 새 V의 norm readback(host) → 수렴이면 break
```

본문 맨 위의 jacobian/solve는 직전 replay(첫 스텝은 pre-loop)의 끝에서 계산된 `Ibus`/`V`와
잔차 `d_F`를 (영속 디바이스 버퍼로 이어받아) 사용하므로 매 스텝 `J(V)·dx = F(V)`가 성립한다.
수렴을 일으킨 norm은 **우리가 보관하는 최종 V의 잔차**라서, eager 경로와 V 시퀀스·보고 norm·
반복수가 정확히 일치하고 낭비 스텝이 없다(검증: graph on/off의 `|ΔV|`가 atomic-add 비결정성
수준 — Mixed/batched는 비트 동일).

구현: `cpp/src/newton_solver/core/graph_iteration.hpp` (`run_iterations_graph`),
`run_iteration_stages`(`newton_solver.cpp`)가 graph-enabled custom 파이프라인이면 위임.

## 제약 (caveats)

- **forward 전용**: custom 솔버의 adjoint(역전파)는 미구현이라 그래프 모드도 forward
  solve에만 적용된다. `solve_adjoint*`는 기존대로 동작(그래프 미적용).
- **`CUPF_ENABLE_TIMING`과 양립 불가**: 타이밍 빌드는 stage마다 `cudaDeviceSynchronize`
  (`sync_cuda_for_timing`)를 넣는데 이는 stream capture 중 금지된 연산이다. 그래프 모드와
  타이밍 빌드를 함께 켜지 말 것.
- **`ENABLE_DUMP`와 양립 불가**: dump 경로(jacobian/residual)는 캡처 구간 안에서 D2H 복사를
  하므로 capture를 깬다. 디버그 dump는 그래프 모드에서 끌 것.
- **B==1도 batched 경로**: 그래프 모드에선 단일 케이스도 라이브러리의 uniform-batch 경로
  (B=1)로 통일해 하나의 capturable 코드 경로를 쓴다.
- **재캡처 조건**: batch 크기가 바뀌면 그래프를 재캡처한다. 희소 패턴(토폴로지)이 바뀌면
  `analyze()`(symbolic)를 다시 호출해야 하며, 이는 그래프 이전 단계라 동일하게 처리된다.
- **그래프 모드에서 per-op 커널 타이밍 없음**: external 모드의 factor/solve는 capture를 깨지
  않으려 host sync/이벤트 타이밍을 생략한다(`kernel_ms` 미보고).
