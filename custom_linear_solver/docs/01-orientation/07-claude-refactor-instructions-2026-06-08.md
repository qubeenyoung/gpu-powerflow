# 클로드 리팩토링 규칙 — 2026-06-08

`gpu-powerflow/custom_linear_solver/src` 정리 규칙이다. 긴 계획을 쓰지 말고, 아래 규칙을 한 모듈씩 적용한다.

기준: [CUTLASS/NVIDIA](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/programming_guidelines.html), [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html), [Doxygen](https://www.doxygen.nl/manual/docblocks.html). 충돌 시 `기존 public API 호환성 -> CUTLASS/NVIDIA CUDA 관례 -> Google C++ -> Doxygen` 순서로 따른다.

## 1. 실행

- 먼저 `src/` 전체 구조를 읽는다.
- 한 번에 하나만 고친다: 이름, 상수, 함수 분리, 파일 이동, 문서 갱신.
- 순서: `solver -> multifrontal -> plan -> factorize -> solve -> matrix/symbolic/reordering/profile -> tests/docs`.
- 사용자 승인 없이 진행한다.
- 성능 정책, 수치 결과은 보존한다.
- public api 수정 허용.
- 모듈 경계 변경 시 include, CMake, docs를 같이 갱신한다.
- 리팩토링 조건을 만족할 때까지 반복하여 점검한다.


```bash
cmake -S gpu-powerflow/custom_linear_solver -B gpu-powerflow/custom_linear_solver/build-refactor \
  -DCMAKE_BUILD_TYPE=Release -DCLS_BUILD_CUDA_OPS=ON -DCLS_BUILD_SCRIPTS=ON \
  -DCLS_BUILD_CUDSS_SCRIPT=OFF -DCLS_CUDA_ARCHITECTURES=86
cmake --build gpu-powerflow/custom_linear_solver/build-refactor -j
```

## 2. 이름

- 파일명, namespace, 변수, 함수 인자, struct field: `snake_case`.
- Type, struct, class, enum: `PascalCase`.
- namespace-scope constant: `inline constexpr kPascalCase`.
- scoped enum만 사용하고 enumerator는 `kPascalCase`.
- macro는 preprocessor가 필요할 때만 쓰고 `CLS_ALL_CAPS`.
- 기존 함수명은 이 repo 관례에 맞춰 `snake_case`를 유지한다. Google 함수명 `PascalCase`는 이 프로젝트의 local deviation이다.
- 새 `camelCase`와 의미 없는 약어를 만들지 않는다.
- `d_` / `h_` prefix는 device/host pointer 또는 mirror에만 쓴다.

짧은 이름 허용 범위:

- 좁은 loop scope의 `i`, `j`, `k`.
- CUDA 문맥의 `lane`, `warp`, `thread_idx`, `block_idx`.
- precision 약어 `FP64`, `FP32`, `FP16`, `TF32`.
- device hot loop의 매우 좁은 local 약어. host dispatch와 plan code에서는 풀어쓴다.

## 3. 상수와 정책

다음 값은 하나의 canonical header로 모은다. 파일마다 재정의하지 않는다.

```cpp
inline constexpr int kWarpSize = 32;
inline constexpr int kSmallFrontMax = 32;
inline constexpr int kMidFrontMax = 128;
inline constexpr int kNumFrontTiers = 3;
inline constexpr int kSmallTierCount = 1;
inline constexpr int kSmallTierWarpsPerBlock = 8;
inline constexpr int kMaxPivotColumns = 64;
inline constexpr int kTensorCorePivotColumnCap = 32;
inline constexpr std::size_t kMidSharedMemoryBudgetBytes = 96 * 1024;
inline constexpr std::size_t kDynamicSharedMemoryOptInBytes = 99 * 1024;
inline constexpr int kMaxSubtreeStreams = 8;
inline constexpr int kArenaAlignmentBytes = 256;
```

공통 API는 repo 함수명 관례에 맞춰 `snake_case`로 둔다.

```cpp
enum class FrontTier { kSmall, kMid, kBig };

constexpr FrontTier classify_front_tier(int front_size);
constexpr int front_tier_index(FrontTier tier);
constexpr int round_up_to_multiple(int value, int alignment);
```

`plan/analyze.cu`, `factorize/dispatch.cuh`, `solve/dispatch.cuh`는 같은 classifier를 사용한다. “must match” 주석은 제거한다.

## 5. 함수 분리

함수 하나는 한 책임만 가진다.

- scan: host metadata를 읽고 stats를 만든다.
- policy: tier, thread count, shared memory size를 결정한다.
- launch: kernel을 호출한다.
- orchestration: level, subtree, pass 순서를 결정한다.

## 7. 주석

- 주석은 이유와 invariant를 설명한다.
- 코드가 이미 말하는 내용을 반복하지 않는다.
- magic number 설명은 canonical constant 이름으로 대체한다.
- historical experiment 이름은 새 코드 주석에 넣지 않는다.
- ASCII diagram은 layout, dependency, launch hierarchy처럼 그림이 필요한 경우만 쓴다.

## 8. 금지

- `git reset --hard`, `git checkout --`, 대량 삭제.
- 사용자 변경 되돌리기.
- 리팩토링 중 새 heuristic 추가.
- 새 build flag/env var 추가.
- include cycle 생성.
- 성능 근거 없는 kernel signature 대개편.
