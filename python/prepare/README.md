# python.prepare — dataset preparation

Build the **power-grid benchmark matrices** for the custom_linear_solver from
MATPOWER `.m` cases (pure pandapower + scipy; no MATLAB/Julia).

## 계통(power-grid) 변환

`convert_linear_system.py` writes a Newton–Raphson Jacobian linear system
(`J.mtx` = the Jacobian, `F.mtx` = the power-mismatch RHS) per case:

```sh
# from the repo root
python3 -m python.prepare.convert_linear_system \
    --dataset-root <MATPOWER .m root> \
    --output-root  custom_linear_solver/tests/datasets/power \
    --cases case118 case1354pegase case_ACTIVSg25k case_SyntheticUSA
```

Each `--cases` entry produces `<output-root>/<case>/{J.mtx, F.mtx, metadata.json}`,
ready for the runners under `custom_linear_solver/tests/`.

Helpers in this package:
- `convert_linear_system.py` — MATPOWER `.m` → `J.mtx`/`F.mtx` (the main entry).
- `convert_m_to_mat.py` — MATPOWER `.m` → SciPy/PYPOWER `.mat`.
- `prepare.py` — parse a case, build Ybus/Sbus/V0, solve a SciPy reference PF.

## SuiteSparse (out-of-domain) 행렬은 여기서 만들지 않는다

회로·2D/3D-FEM 같은 out-of-domain 행렬은 변환이 아니라 **다운로드**한다 — SuiteSparse
Matrix Collection에서 받는 스크립트를 쓴다:

```sh
custom_linear_solver/tests/datasets/fetch_suitesparse.sh          # 기본 셋
custom_linear_solver/tests/datasets/fetch_suitesparse.sh --all    # 전체
```
