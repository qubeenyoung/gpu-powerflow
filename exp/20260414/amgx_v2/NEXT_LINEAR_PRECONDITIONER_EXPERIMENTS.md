# AMGX v2 Linear Preconditioner Experiments

## Baseline

The current AMGX v2 solver uses a bus-local augmented scalar CSR system:

```text
x = [bus0.theta, bus0.Vm, bus1.theta, bus1.Vm, ...]
```

Slack rows and PV `Vm` rows are fixed identity rows. The matrix is assembled on
device and passed to scalar AMGX AMG as the FGMRES right preconditioner.

Baseline run:

- Result file: `results/natural_strict_all_inner500_restart200.csv`
- Ordering: `natural`
- Nonlinear tolerance: `1e-8`
- Linear tolerance: `1e-2`
- Inner max iterations: `500`
- GMRES restart: `200`
- Linear failure policy: strict
- Converged: `5 / 12`
- Failure mode: all 7 failures hit `linear_max_inner_iterations`

This means the main blocker is the inner linear solve, not the nonlinear
convergence test itself.

## Common Measurement Protocol

Run every experiment with the same 12 dump cases:

```text
case_ACTIVSg200
case_ACTIVSg500
MemphisCase2026_Mar7
case_ACTIVSg2000
Base_Florida_42GW
Texas7k_20220923
Base_Texas_66GW
Base_MIOHIN_76GW
Base_West_Interconnect_121GW
case_ACTIVSg25k
case_ACTIVSg70k
Base_Eastern_Interconnect_515GW
```

Primary metrics:

- Nonlinear convergence count at `||F||_inf <= 1e-8`.
- Final nonlinear residual.
- Total inner iterations.
- Linear failure count and failure reason.
- Per-case wall time.
- For failures, whether the first linear system makes useful progress.

Start with strict mode. After that, run `continue_on_linear_failure` only as a
diagnostic to see whether an inexact Newton update still decreases the nonlinear
residual.

## Experiment 1: AMGX 2x2 Block CSR Preconditioner

### Hypothesis

The current v2 ordering places `[theta_i, Vm_i]` next to each other, but scalar
AMGX still sees only scalar graph edges. A fixed 2x2 block system should expose
the bus-local physical coupling directly to AMGX aggregation and smoothing.

This is the most direct realization of the v2 idea.

### Implementation Plan

Add a block preconditioner path under `src/linear`:

- `block_amgx_preconditioner.hpp/.cu`
- `block_csr_matrix_view`

Build a 2x2 block CSR graph from the existing bus graph:

- One block row per bus-local position.
- One block column per neighboring bus in the Ybus graph.
- Each block stores:
  - `dP/dtheta`, `dP/dVm`
  - `dQ/dtheta`, `dQ/dVm`
- Slack and PV `Vm` fixed rows become identity rows inside the 2x2 block:
  - Slack block rows: identity on both local slots.
  - PV block row: active P row plus fixed `Vm` identity row.
  - PQ block row: full 2x2 physical Jacobian rows.

FGMRES operator should remain the assembled scalar SpMV path at first. Only the
preconditioner matrix format changes. This isolates whether AMGX block metadata
helps.

### Validation

First run:

- `case_ACTIVSg200`
- `case_ACTIVSg500`
- `case_ACTIVSg2000`
- `Base_Texas_66GW`

Then run all 12 cases if no correctness issue appears.

Success criteria:

- Keep all 5 currently converged cases converged.
- Improve `Base_Texas_66GW` or at least reduce its first linear residual enough
  to avoid `linear_max_inner_iterations` at 500.
- Reduce total linear failures compared with scalar AMGX.

Risk:

- AMGX block upload API details may require exact block layout and mode
  selection. If the API friction is high, keep this as a separate implementation
  path instead of mutating the scalar wrapper.
- If AMGX block CSR support is blocked by API or mode constraints, stop this
  branch early and prioritize experiment 4 rather than forcing a half-block
  scalar workaround.

## Experiment 3: Active-Only Reduced System

### Hypothesis

The augmented `2*n_bus` system is clean for bus-local indexing, but the fixed
identity rows for slack and PV `Vm` may be unhelpful to AMG coarsening. Solving
only the physical active unknowns may give AMGX a cleaner graph.

This is a comparison experiment, not the preferred final design unless it
clearly beats the augmented block path.

### Implementation Plan

Keep the bus-local 2-slot index as the canonical assembly index.

Add an active-slot map:

```text
active_slot -> reduced_index
reduced_index -> active_slot
```

Create a reduced CSR matrix by gathering active rows and active columns from the
augmented CSR:

- PV bus: keep P/theta row and theta column.
- PQ bus: keep P/theta and Q/Vm rows/columns.
- Slack bus: remove both slots.
- PV `Vm`: remove fixed slot.

Add gather/scatter kernels:

- `reduced_rhs = gather(active augmented residual)`
- `augmented_dx = scatter(reduced_dx)`, with inactive slots set to zero.

FGMRES operator can either:

- Use a reduced SpMV over the reduced CSR.
- Or scatter `z` to augmented, apply augmented SpMV, and gather output. The
  first version is cleaner for timing, the second is faster to implement.

### Validation

First run natural scalar AMGX with reduced CSR:

- `case_ACTIVSg200`
- `case_ACTIVSg500`
- `case_ACTIVSg2000`
- `Base_Texas_66GW`

Then run all 12 cases.

Success criteria:

- No regression on currently converged small cases.
- At least one of `Base_Florida_42GW`, `Base_MIOHIN_76GW`, or
  `Base_Texas_66GW` avoids the first linear failure.
- Reduced dimension and timing are recorded so any improvement is not just from
  solving a smaller system with worse residual behavior.

Risk:

- Removing fixed identity rows may help AMG, but it also loses the uniform 2x2
  structure that experiment 1 needs. Treat this as a control for the augmented
  formulation rather than a replacement for the block path.

## Experiment 4: Bus-Local Block Smoother Plus AMGX

### Hypothesis

Previous experiments showed that a bus-local block Jacobi path had a strong
signal on `Base_Texas_66GW`. That suggests the local 2x2 physics matters more
than generic scalar AMG can infer from the graph. A multiplicative combination
can use the bus-local block smoother for local theta/Vm coupling and AMGX for
global correction.

### Implementation Plan

Add a physical block smoother under `src/linear`:

- `bus_local_block_smoother.hpp/.cu`

Use the assembled bus-local Jacobian values to extract local diagonal blocks:

- Slack: identity block.
- PV: active P/theta scalar plus fixed Vm identity.
- PQ: 2x2 block inverse.

Implement preconditioner modes:

```text
block_only:
  z = B^{-1} r

additive:
  z = B^{-1} r + M_amgx^{-1} r

block_then_amgx:
  z0 = B^{-1} r
  e  = M_amgx^{-1} (r - A z0)
  z  = z0 + e

amgx_then_block:
  z0 = M_amgx^{-1} r
  e  = B^{-1} (r - A z0)
  z  = z0 + e
```

Prioritize `block_then_amgx`, because it first removes the local bus error that
AMGX is currently struggling to model.

### Validation

First run:

- `Base_Texas_66GW`
- `Base_Florida_42GW`
- `Base_MIOHIN_76GW`
- `case_ACTIVSg25k`

Then run all 12 cases if the first batch improves.

Success criteria:

- `Base_Texas_66GW` reaches nonlinear `1e-8` in strict mode or needs fewer
  linear failures under continue mode.
- Failure cases show lower final nonlinear residual than scalar AMGX.
- Runtime increase is not worse than the residual improvement justifies.

Risk:

- Additive mode can overcorrect because both terms approximate the same inverse.
  Multiplicative residual correction is the safer first comparison.

## Execution Order

1. Implement experiment 1: AMGX 2x2 block CSR preconditioner.
2. Run the 4-case smoke set and then all 12 cases if correct.
3. Implement experiment 4: bus-local block smoother plus AMGX, starting with
   `block_then_amgx`.
4. Run the focused failure set and then all 12 cases if promising.
5. Implement experiment 3: active-only reduced system as a control experiment.

This order favors the v2 design intent first: preserve bus-local 2-slot
structure and make the preconditioner understand it.

## Result Files

Use consistent output names:

```text
results/block2x2_amgx_strict_all_inner500_restart200.csv
results/block2x2_amgx_continue_all_inner500_restart200.csv
results/block_then_amgx_strict_all_inner500_restart200.csv
results/block_then_amgx_continue_all_inner500_restart200.csv
results/active_reduced_scalar_amgx_strict_all_inner500_restart200.csv
```

Each result should get a companion `.md` summary with:

- configuration,
- convergence count,
- bus-size-sorted table,
- comparison against `natural_strict_all_inner500_restart200.csv`,
- interpretation of whether the inner linear solve improved.
