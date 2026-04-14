# Iterative Linear Solver Experiment

목적: cuPF Newton 반복의 `J dx = -F` 선형계를 cuDSS direct solve 대신 iterative solver로 풀 수 있는지 보기 위한 격리 실험.

이 폴더는 본체 solver 경로를 바꾸지 않고 두 단계로 나눕니다.

- `dump_linear_systems`: benchmark dump case를 로드해서 Newton 반복을 돌리고, cuDSS/KLU solve 직후 같은 iteration의 `J`, `rhs=-F`, direct `dx`를 저장합니다.
- `iterative_probe`: 저장된 스냅샷을 읽어서 Eigen `BiCGSTAB`로 풀고 residual과 direct solution 차이를 출력합니다.

기본 대상 case는 `case118_ieee`, `case2746wop_k`, `case8387_pegase`입니다. 여기서 `case2746wop_k`를 2XXX 대표 benchmark case로 잡았습니다.

## Build

```bash
cmake -S /workspace/exp/20260413/iterative -B /workspace/exp/20260413/iterative/build -GNinja
cmake --build /workspace/exp/20260413/iterative/build --target dump_linear_systems iterative_probe
```

CUDA/cuDSS 없이 CPU dump만 보고 싶으면:

```bash
cmake -S /workspace/exp/20260413/iterative -B /workspace/exp/20260413/iterative/build-cpu -GNinja -DITERATIVE_WITH_CUDA=OFF
cmake --build /workspace/exp/20260413/iterative/build-cpu --target dump_linear_systems iterative_probe
```

## Dump

benchmark의 mixed CUDA edge profile에 해당하는 FP32 `J`, FP32 `rhs=-F` 스냅샷:

```bash
/workspace/exp/20260413/iterative/build/dump_linear_systems \
  --profile cuda_mixed_edge \
  --dataset-root /workspace/datasets/cuPF_benchmark_dumps \
  --cases case118_ieee case2746wop_k case8387_pegase \
  --output-root /workspace/exp/20260413/iterative/dumps
```

초기 몇 개 iteration만 빠르게 덤프:

```bash
/workspace/exp/20260413/iterative/build/dump_linear_systems \
  --profile cuda_mixed_edge \
  --max-dump-iters 1
```

스냅샷 layout:

```text
dumps/<case>/<profile>/iter_000/
  J.csr
  rhs.txt
  x_direct.txt
  meta.txt
```

## Iterative Probe

전체 dump를 한 번에 probe:

```bash
/workspace/exp/20260413/iterative/build/iterative_probe \
  --snapshot-root /workspace/exp/20260413/iterative/dumps \
  --solver bicgstab_ilut \
  --tolerance 1e-8 \
  --max-iter 1000 \
  --output-csv /workspace/exp/20260413/iterative/results/bicgstab_ilut.csv
```

단일 스냅샷만 probe:

```bash
/workspace/exp/20260413/iterative/build/iterative_probe \
  --snapshot-dir /workspace/exp/20260413/iterative/dumps/case118_ieee/cuda_mixed_edge/iter_000
```
