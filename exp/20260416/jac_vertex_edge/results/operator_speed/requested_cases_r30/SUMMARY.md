# Jacobian Operator Speed Probe

This run used `jacobian_operator_probe` with `warmup=5` and `repeats=30`.
Timing is CUDA event elapsed time for one Jacobian operator run, including the
Jacobian value-buffer zeroing done by each operator.

`edge_noatomic` is not a valid solver path. It is the same edge work with
plain load/add/store replacing `atomicAdd`, so the numbers are only an
upper-bound style probe for atomic time share.

`edge/vertex speedup` is `vertex mean us / edge_atomic mean us`.
`atomic time share` estimates the fraction of edge_atomic time attributable
to atomic updates: `(edge_atomic mean us - edge_noatomic mean us) / edge_atomic mean us * 100`.
Negative estimates from timing noise are reported as 0.0%.

| Case | Edge atomic (us) | Vertex (us) | Edge/Vertex speedup | Edge no-atomic (us) | Atomic time share | Vertex lane util | Edge lane util |
|---|---:|---:|---:|---:|---:|---:|---:|
| ACTIVSg | 17.765 | 35.775 | 2.014x | 12.757 | 28.2% | 49.27% | 95.80% |
| Base Eastern | 51.903 | 116.289 | 2.241x | 44.030 | 15.2% | 49.27% | 96.66% |
| Base West | 17.156 | 31.284 | 1.823x | 12.297 | 28.3% | 49.27% | 96.59% |
| Memphis | 7.422 | 7.939 | 1.070x | 7.798 | 0.0% | 49.27% | 96.61% |
| Texas7K | 9.712 | 15.469 | 1.593x | 8.562 | 11.8% | 49.27% | 96.30% |

Across these five cases, the geometric-mean edge_atomic speedup over vertex is
1.696x. The arithmetic-mean atomic time share estimate is 16.7%. Small cases can
be dominated by timing granularity noise.

Raw repeat data:

- `operator_speed_raw.csv`
- `operator_speed_summary.csv`
- `operator_speed_comparison.csv`
- `raw/*.csv`
