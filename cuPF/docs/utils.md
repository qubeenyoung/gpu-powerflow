# utils — 공용 유틸리티

CUDA 백엔드와 드라이버가 공유하는 작은 빌딩 블록. 모두 헤더 위주이며
`cpp/inc/utils/`에 있다.

---

## 1. `cuda_utils.hpp` — CUDA 공용

- **`DeviceBuffer<T>`**: device 메모리 RAII 래퍼. `resize`(cudaMalloc),
  `assign`(H2D), `copyTo`(D2H), `memsetZero`, `data/size/empty`. 복사 생성자·대입
  삭제(이동만) — device 메모리 우발 복사 방지. 모든 storage 버퍼의 기반
  ([storage.md](storage.md)).
- **현재 스트림**: `cupf_current_cuda_stream()`이 cuPF 작업을 띄울 스트림을 돌려준다.
  `ScopedCudaStream`은 (torch bridge 등에서) 호출 동안 현재 스트림을 외부 스트림으로
  바꾼다 — torch의 current CUDA stream과 통합하는 데 쓰인다([core.md](core.md)).
- **에러 체크**: `CUDA_CHECK(...)` / `CUDSS_CHECK(...)` 매크로 — 실패 시 예외.
- **`sync_cuda_for_timing()`**: 타이밍이 켜졌을 때 단계 경계에서 동기화(끄면 비용 거의
  없음). 커널 런처들이 launch 후 호출한다.

## 2. `timer.hpp` — 계층형 타이밍

`ScopedTimer`가 RAII로 한 단계의 wall-clock을 라벨별로 누적한다. `ENABLE_TIMING=ON`
빌드에서만 실제 수집되고(끄면 no-op → 보고값 0), 라벨 규약은 `NR.<phase>.<stage>`.
드라이버의 `StageScope`가 NVTX와 함께 묶어 호출한다.

## 3. `nvtx_trace.hpp` — 프로파일 범위

`ScopedNvtxRange`가 NVTX range를 열어 Nsight Systems 타임라인에 단계를 시각화한다.
NVTX 미지원/비활성 빌드에서는 no-op.

## 4. `dump.hpp` — 디버그 덤프

`isDumpEnabled()`가 true일 때(환경/플래그) NR 반복마다 벡터/배열/CSR 행렬을 파일로
기록(`dumpVector`, `dumpCSR`). 솔버 단계들이 잔차·dx·Jacobian을 선택적으로 덤프해
CPU/GPU·정밀도 간 결과를 대조하는 데 쓴다. 비활성 시 전부 no-op.

## 5. `logger.hpp` — 로깅

spdlog를 감싸는 얇은 래퍼. 빌드 설정(`CUPF_LOGGING_ENABLED`, spdlog 존재 여부)에 따라
실로깅/무동작으로 컴파일된다. spdlog가 없으면 로깅은 비활성.
