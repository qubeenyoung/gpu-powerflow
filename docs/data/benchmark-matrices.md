# Benchmark Matrices

Benchmark matrices are downloaded during the Docker build and stored inside the
image under:

```text
/datasets/benchmark_matrices
```

The download script keeps raw SuiteSparse Matrix Collection archives under
`downloads` and extracts Matrix Market files under `matrices`. The Docker build
then generates common companion files next to each selected matrix:
`rhs.mtx = A * x_true.mtx`, where `x_true.mtx` is generated from a fixed random
seed with entries sampled from `U(-1, 1)`.

```text
/datasets/benchmark_matrices/
  downloads/
  matrices/
  MANIFEST.txt
```

## Default Benchmark Set

The default Docker build downloads the selected 5-matrix benchmark set below.
This is the project default set, not only a smoke-test set.

| Set | Matrix | URL | Role |
|---|---|---|---|
| Circuit | `Hamm/memplus` | `https://sparse.tamu.edu/MM/Hamm/memplus.tar.gz` | Memory-circuit case with explicit-zero entries in Matrix Market |
| Circuit | `Rajat/rajat27` | `https://sparse.tamu.edu/MM/Rajat/rajat27.tar.gz` | Circuit case with many dmperm blocks and rank deficiency |
| Semiconductor | `Wang/wang3` | `https://sparse.tamu.edu/MM/Wang/wang3.tar.gz` | 3D semiconductor-device Jacobian-like matrix |
| Frequency-domain circuit | `ATandT/onetone2` | `https://sparse.tamu.edu/MM/ATandT/onetone2.tar.gz` | Harmonic-balance circuit matrix for direct factorization |
| Circuit | `Rajat/rajat15` | `https://sparse.tamu.edu/MM/Rajat/rajat15.tar.gz` | Larger Rajat circuit matrix with high numeric symmetry |

## Selected Matrix Properties

The dimensions below are from each matrix page in the SuiteSparse Matrix
Collection.

| Matrix | Rows x cols | Nonzeros | Pattern entries | Kind | Symmetric |
|---|---:|---:|---:|---|---|
| [`Hamm/memplus`](https://sparse.tamu.edu/Hamm/memplus) | 17,758 x 17,758 | 99,147 | 126,150 | Circuit simulation | No |
| [`Rajat/rajat27`](https://sparse.tamu.edu/Rajat/rajat27) | 20,640 x 20,640 | 97,353 | 99,777 | Circuit simulation | No |
| [`Wang/wang3`](https://sparse.tamu.edu/Wang/wang3) | 26,064 x 26,064 | 177,168 | 177,168 | Semiconductor device | No |
| [`ATandT/onetone2`](https://sparse.tamu.edu/ATandT/onetone2) | 36,057 x 36,057 | 222,596 | 227,628 | Frequency-domain circuit simulation | No |
| [`Rajat/rajat15`](https://sparse.tamu.edu/Rajat/rajat15) | 37,261 x 37,261 | 443,573 | 443,573 | Circuit simulation | No |

The Dockerfile default is the comma-separated form of those URLs:

```text
SUITESPARSE_MATRIX_URLS=https://sparse.tamu.edu/MM/Hamm/memplus.tar.gz,https://sparse.tamu.edu/MM/Rajat/rajat27.tar.gz,https://sparse.tamu.edu/MM/Wang/wang3.tar.gz,https://sparse.tamu.edu/MM/ATandT/onetone2.tar.gz,https://sparse.tamu.edu/MM/Rajat/rajat15.tar.gz
```

## Extended and Stress Candidates

These matrices are intentionally excluded from the default build to keep image
size and build time under control. Add them with `SUITESPARSE_MATRIX_URLS` when
you need a larger stress profile.

| Set | Matrix | URL | Why excluded from default |
|---|---|---|---|
| Circuit-mid | `ATandT/pre2` | `https://sparse.tamu.edu/MM/ATandT/pre2.tar.gz` | Circuit-mid behavior is already covered by the selected circuit matrices |
| Circuit-large | `Freescale/Freescale1` | `https://sparse.tamu.edu/MM/Freescale/Freescale1.tar.gz` | Larger industrial circuit stress case |
| Circuit-large | `Rajat/rajat31` | `https://sparse.tamu.edu/MM/Rajat/rajat31.tar.gz` | Larger sparse circuit-like stress case |
| Circuit-huge | `Freescale/circuit5M` | `https://sparse.tamu.edu/MM/Freescale/circuit5M.tar.gz` | Can dominate download size, extraction time, and memory |
| CFD | `Goodwin/Goodwin_127` | `https://sparse.tamu.edu/MM/Goodwin/Goodwin_127.tar.gz` | CFD stress case outside the circuit/device-focused default |
| Transport | `Janna/Transport` | `https://sparse.tamu.edu/MM/Janna/Transport.tar.gz` | Heavy transport-style stress case |
| Graph-large | `vanHeukelum/cage14` | `https://sparse.tamu.edu/MM/vanHeukelum/cage14.tar.gz` | Large graph case outside the circuit/device-focused default |

## Override Behavior

To download only a subset or a different matrix set, override
`SUITESPARSE_MATRIX_URLS` with a comma-separated URL list:

```bash
docker build \
  --build-arg SUITESPARSE_MATRIX_URLS='https://sparse.tamu.edu/MM/Wang/wang3.tar.gz' \
  -t sparse-direct-solver:3090 .
```

When a single URL is used, the script uses the URL basename as the archive
filename. With multiple URLs, the script uses numeric prefixes such as
`0001_memplus.tar.gz` to avoid archive-name collisions.

If you want a quick smoke-test image instead of the default benchmark set,
override the build argument explicitly:

```bash
docker build \
  --build-arg SUITESPARSE_MATRIX_URLS='https://sparse.tamu.edu/MM/Hamm/memplus.tar.gz' \
  -t sparse-direct-solver:3090 .
```

Example extended stress profile:

```bash
docker build \
  --build-arg SUITESPARSE_MATRIX_URLS='https://sparse.tamu.edu/MM/Freescale/Freescale1.tar.gz,https://sparse.tamu.edu/MM/Rajat/rajat31.tar.gz,https://sparse.tamu.edu/MM/Freescale/circuit5M.tar.gz,https://sparse.tamu.edu/MM/Goodwin/Goodwin_127.tar.gz,https://sparse.tamu.edu/MM/Janna/Transport.tar.gz,https://sparse.tamu.edu/MM/vanHeukelum/cage14.tar.gz' \
  -t sparse-direct-solver:3090 .
```

## Selection Criteria

The default set is designed to focus on circuit-like and device-simulation
direct-solver behavior without making every Docker build a large stress test:

| Criterion | What it checks |
|---|---|
| Matrix Market coverage | Coordinate matrices extracted from SuiteSparse Matrix Collection archives |
| Companion files | Generated `rhs.mtx` and `x_true.mtx` make SuiteSparse and MATPOWER systems use the same benchmark input shape |
| Circuit matrices | Very sparse unsymmetric matrices with hard pivoting behavior |
| Semiconductor device matrix | Non-circuit device-simulation sparsity and numerical behavior |
| Size sweep | Roughly 18K to 37K rows, suitable for frequent solver comparisons |
| Ordering stress | Multiple dmperm/SCC profiles across the selected circuit matrices |

## Experiment Metadata to Record

Each benchmark run should record:

| Field | Why it matters |
|---|---|
| Matrix name | Allows direct lookup in SuiteSparse Matrix Collection |
| Matrix dimensions and nonzeros | Normalizes factorization and solve timings |
| Symmetry flag | Determines whether symmetric or unsymmetric solver path is valid |
| Ordering method | Often dominates fill-in and memory behavior |
| Factorization time | Primary setup cost |
| Solve time | Repeated RHS cost |
| Peak memory | Key constraint for GPU and direct solvers |
| `berr` | Componentwise backward error, measured with `compute_error` |
| Absolute solution error | `||x - x_true||_2`, measured with `compute_error` |
| Solver version | Needed when comparing cuDSS-like behavior over time |
| CUDA architecture | Important for GPU solver reproducibility |

## Runtime Check

Inside the container, a quick listing should show downloaded archives and
extracted Matrix Market files:

```bash
find "$BENCHMARK_MATRIX_ROOT" -maxdepth 4 -type f | sort
```
