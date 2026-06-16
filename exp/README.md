# exp/ — head-to-head 실험 작업 디렉터리 (로컬)

custom_linear_solver vs STRUMPACK+MAGMA head-to-head 실험의 산출물.
분석/결론은 `custom_linear_solver/docs/05-reports/06-head-to-head-2026-06-16.md` 참조.

## 구성
- `harness/` — STRUMPACK 벤치 하니스
  - `strumpack_bench.cpp` — 소스 (DIRECT, GPU, reorder 1회 + B×(refactor+solve))
  - `strumpack_bench` — no-MAGMA 빌드 (STRUMPACK native 경로)
  - `strumpack_bench_magma` — MAGMA vbatched 경로
- `cases/` — 입력 행렬 (`{case}/J.mtx` 실수 대각우세 J, `F.mtx` RHS)
  - `case_ACTIVSg25k` (n=25k), `case_SyntheticUSA` (n=82k), `case3120sp`
- `ybus_to_real_case.py` — 복소 Ybus .mtx → 실수 J.mtx + ones F.mtx 변환

## 외부 의존 (repo 밖, 옮기지 않음 — 절대경로 baked)
- MAGMA 2.8.0: `/opt/magma` (런타임), `/root/baselines/magma-2.8.0` (빌드)
- STRUMPACK: `/root/baselines/STRUMPACK` (`USE_MPI=OFF USE_CUDA=ON TPL_ENABLE_MAGMA=ON`)

## 실행
```bash
cd exp/harness
export LD_LIBRARY_PATH=/opt/magma/lib:$LD_LIBRARY_PATH
# 인자: J.mtx  B  repeat  [STRUMPACK opts]
./strumpack_bench_magma ../cases/case_ACTIVSg25k/J.mtx 1 10 --sp_reordering_method metis
# 깊이 매칭: --sp_reordering_method and    /  통계: --sp_verbose

# 우리 솔버
../../custom_linear_solver/build/custom_linear_solver_run \
  --matrix ../cases/case_ACTIVSg25k/J.mtx --rhs ../cases/case_ACTIVSg25k/F.mtx \
  --repeat 10 --warmup 3 --single-precision fp64 --analyze-info
```
