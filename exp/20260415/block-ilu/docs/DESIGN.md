# Block-ILU Design

## Goal

Test whether the off-diagonal Jacobian coupling is weak enough that
`blockdiag(J11, J22)` is a useful preconditioner when each diagonal block is
approximately inverted by cuSPARSE ILU0.

The solver equation at each Newton step is:

```text
J dx = -F
```

with:

```text
J = [ J11  J12 ]
    [ J21  J22 ]

M ~= [ J11  0   ]
     [ 0    J22 ]
```

BiCGSTAB uses the full `J` for matrix-vector products and applies `M^-1`
inside the Krylov iteration.

## cuPF Jacobian Layout

Use the original cuPF reduced layout:

```text
pvpq = pv || pq
n_pvpq = len(pvpq)
n_pq = len(pq)
dimF = n_pvpq + n_pq

rows:
  [0, n_pvpq)        -> P rows for pvpq
  [n_pvpq, dimF)     -> Q rows for pq

columns:
  [0, n_pvpq)        -> theta columns for pvpq
  [n_pvpq, dimF)     -> Vm columns for pq
```

The four block shapes are:

```text
J11: n_pvpq x n_pvpq
J12: n_pvpq x n_pq
J21: n_pq   x n_pvpq
J22: n_pq   x n_pq
```

## What Gets Analyzed Once

The following are fixed for a case and should be built once before the Newton
loop:

- full `J` CSR pattern and Ybus-to-J scatter maps
- `J11` CSR pattern and Ybus-to-J11 scatter maps
- `J22` CSR pattern and Ybus-to-J22 scatter maps
- cuSPARSE descriptors for full `J`, `J11`, and `J22`
- cuSPARSE ILU analysis objects for `J11` and `J22`
- cuSPARSE triangular solve descriptors for the `L/U` factors of both blocks
- BiCGSTAB work vectors

The key constraint is that `row_ptr` and `col_idx` do not change after analysis.

## What Gets Updated In The Newton Loop

At every Newton iteration:

1. Compute mismatch `F`.
2. Stop if `||F||_inf <= nonlinear_tol`.
3. Zero full `J.values`, `J11.values`, and `J22.values`.
4. Launch one edge-based Jacobian value kernel.
5. The kernel writes:
   - all four blocks into full `J.values`
   - only `J11` contributions into `J11.values`
   - only `J22` contributions into `J22.values`
6. Run numeric ILU0 on `J11.values`.
7. Run numeric ILU0 on `J22.values`.
8. Solve `J dx = -F` with right-preconditioned BiCGSTAB.
9. Apply `dx` with the same cuPF voltage update layout.

## Preconditioner Apply

For a vector:

```text
r = [ r_p ]
    [ r_q ]
```

apply:

```text
z_p = U11^-1 L11^-1 r_p
z_q = U22^-1 L22^-1 r_q
z   = [ z_p ]
      [ z_q ]
```

No coupling solve is done through `J12` or `J21`.

## First-Pass Policy

Keep the first implementation conservative:

- rebuild ILU numeric factors every Newton iteration
- do not reorder `J11` or `J22`
- use natural cuPF ordering
- use FP64 only
- fail fast on structural or numerical zero pivot
- add `--continue-on-linear-failure` only after the first smoke path works

Reordering, pivot boosting, mixed precision, or reused ILU factors are separate
follow-up experiments.
