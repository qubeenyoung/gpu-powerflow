# MATLAB MATPOWER Runner

This folder contains a small MATLAB smoke runner for the MATPOWER installation
provided by the project image.

## Online Licensing

For a Campus-Wide or Individual MathWorks account, authenticate once with:

```bash
matlab -licmode onlinelicensing
```

For non-interactive local use, `.env` can provide credentials:

```bash
MATLAB_LICMODE=onlinelicensing
MATLAB_USER_ID=your-account@example.edu
MATLAB_PASSWORD=your-password
```

Then authenticate without printing the secret values:

```bash
./matlab/login_online.bash
```

After sign-in works, run the MATPOWER smoke case through the wrapper:

```bash
MATLAB_LICMODE=onlinelicensing matlab/run_matpower_case.bash case9
```

You can keep local settings in the repository root `.env` file:

```bash
MATLAB_LICMODE=onlinelicensing
MATPOWER_CASE=case9
MATPOWER_RESULT_JSON=/tmp/matpower-case9-summary.json
```

The root `.env` file is intentionally ignored by git.

## Network License Alternative

If the school provides a license server or a license file, use one of:

```bash
MLM_LICENSE_FILE=27000@license-server matlab/run_matpower_case.bash case9
MLM_LICENSE_FILE=/licenses/license.lic matlab/run_matpower_case.bash case9
```

`MATLAB_LICENSE_FILE` is also accepted by the wrapper and mapped to
`MLM_LICENSE_FILE` when the latter is unset.

## AC/NR Linear Solver Sweep

Run the default representative 100+ bus sweep:

```bash
./matlab/sweep_nr_lin_solvers.bash
```

The default case set is selected by bus-count bucket from
`matlab/matpower_case_selection_100plus.csv`. The default solver labels are:

```text
DEFAULT,BACKSLASH,LU,LU3,LU4,LU5
```

Useful `.env` overrides:

```bash
MATLAB_LICMODE=onlinelicensing
MATPOWER_SWEEP_CASES=case118,case300,case6468rte
MATPOWER_LIN_SOLVERS=DEFAULT,BACKSLASH,LU3,LU5
MATPOWER_SWEEP_CSV=results/matpower_ac_nr_lin_solver_sweep.csv
MATPOWER_SWEEP_REPEATS=1
MATPOWER_TOL=1e-8
MATPOWER_MAX_IT=10
```

The sweep fixes `model='AC'` and `pf.alg='NR'`, then varies only
`pf.nr.lin_solver`.
