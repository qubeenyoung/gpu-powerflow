# gpu-powerflow Workspace

이 저장소는 GPU 기반 전력 조류 계산 실험을 위한 작업 공간이다. 현재 기준으로 빌드 가능한 솔버 스냅샷은 [`v2`](/workspace/v2) 아래에 있고, 루트에는 다음 리팩터링을 위한 빈 [`cuPF`](/workspace/cuPF) 디렉터리와 보조 자산이 함께 정리되어 있다.

## 현재 버전 요약

- 주 구현 대상 스냅샷은 `v2` 디렉터리에 보존되어 있다.
- CPU 경로는 FP64 single-case 기준으로 빌드와 smoke test가 가능하다.
- CUDA 경로도 single-case 기준으로 빌드와 기본 smoke test가 가능하다.
- precision-selection 리팩터링은 진행 중이며, 현재는 구조 정리와 실험 가능 상태 확보를 우선하고 있다.
- multi-batch는 이번 단계의 구현 범위에서 제외되어 있다.
- 루트 `cuPF/`는 다음 단계 구현을 위한 새 빈 작업 디렉터리다.

## 워크스페이스 구성

```text
.
├── cuPF/          # 다음 리팩터링 작업을 위한 빈 디렉터리
├── v2/            # 현재 빌드 가능한 cuPF 스냅샷
├── bash/          # 빌드/실행 보조 스크립트
├── python/        # Python 유틸리티와 실험 코드
├── exp/           # 실험 결과와 분석 산출물
└── refactor_docs/ # 로컬 설계 메모와 리팩터링 초안
```

`refactor_docs/`는 작업 중인 로컬 문서 보관용이며, 원격 저장소의 공식 추적 대상에서는 제외한다.

## v2 상태

[`v2`](/workspace/v2)는 현재 보존 중인 `cuPF` C++/CUDA 전력 조류 솔버 스냅샷이다. 희소 `Ybus`를 입력으로 받아 Newton-Raphson 방식으로 전력 조류를 계산한다.

현재 스냅샷의 특징은 다음과 같다.

- `analyze()`와 `solve()`를 분리한 구조를 유지한다.
- CPU backend는 Eigen 기반 FP64 경로를 사용한다.
- CUDA backend는 single-case 경로를 우선 지원한다.
- `solve_batch()`는 현재 비활성화되어 있으며 호출 시 예외를 던진다.
- Python 바인딩은 아직 legacy FP64 중심 인터페이스에 가깝다.

더 자세한 구현 설명과 제약 사항은 [`v2/README.md`](/workspace/v2/README.md)를 보면 된다.

## 리팩터링 요약

최근 작업의 중심은 정밀도 선택과 CUDA 연산 단계 분리 방향을 다시 잡는 것이었다.

- `NewtonOptions::precision`을 정밀도 선택의 source of truth로 두는 방향으로 정리했다.
- `FP32`, `Mixed`, `FP64`를 실험 가능한 구조로 가져가기 위해 기존 backend 중심 분기에서 operator 단계 분리 필요성을 확인했다.
- 앞으로는 `MismatchOp`, `JacobianOp`, `LinearSolveOp`, `VoltageUpdateOp` 같은 단계별 연산 단위를 조합하는 구조로 재편하는 것이 목표다.
- CUDA multi-batch kernel은 single-case kernel과 물리적으로 분리하고, 당장은 TODO로 남긴다.

즉, 현재 버전은 "새 구조로 완전히 옮겨진 상태"가 아니라, 빌드 가능한 기준점을 확보하고 다음 리팩터링을 준비하는 중간 스냅샷이다.

## 빌드와 실행

루트 [`bash/`](/workspace/bash) 아래에 간단한 래퍼 스크립트를 두었다.

- [`build_cupf.bash`](/workspace/bash/build_cupf.bash): 현재 `v2` 스냅샷용 CMake preset 기반 빌드
- [`run_cupf.bash`](/workspace/bash/run_cupf.bash): 현재 `v2` smoke 실행 래퍼

예시는 다음과 같다.

```bash
/workspace/bash/build_cupf.bash --cpu
/workspace/bash/run_cupf.bash --cpu

/workspace/bash/build_cupf.bash --cuda
/workspace/bash/run_cupf.bash --cuda --jacobian vertex_based
```

정식 빌드 옵션과 의존성은 [`v2/README.md`](/workspace/v2/README.md)에 정리되어 있다.

## 주의 사항

- 이 저장소에는 실험 코드와 로컬 분석 산출물이 함께 있다.
- 리팩터링 중인 영역은 문서, 바인딩, 벤치마크 하니스가 완전히 동기화되지 않았을 수 있다.
- 재현 가능한 기준은 우선 `v2`의 single-case CPU/CUDA smoke 경로로 보는 것이 안전하다.
