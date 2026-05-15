# METIS Block P/Q Theta/Vm Composition

- Region rule: `diagonal` means row and column unknowns are in the same METIS block; `offdiagonal` means they are in different blocks.
- Row types: P rows are the first `n_pvpq` rows, Q rows are the final `n_pq` rows.
- Column types: theta columns are the first `n_pvpq` columns, |V| columns are the final `n_pq` columns.

| case | block | offblock nnz | diag row P | diag col theta | diag J11 | diag J22 | off row P | off col theta | off J11 | off J22 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| case2383wp | 32 | 0.164 | 0.530 | 0.530 | 0.289 | 0.228 | 0.572 | 0.572 | 0.307 | 0.163 |
| case2383wp | 64 | 0.215 | 0.534 | 0.534 | 0.330 | 0.262 | 0.550 | 0.550 | 0.155 | 0.056 |
| case3120sp | 32 | 0.143 | 0.516 | 0.516 | 0.271 | 0.238 | 0.548 | 0.548 | 0.287 | 0.191 |
| case3120sp | 64 | 0.206 | 0.518 | 0.518 | 0.308 | 0.273 | 0.533 | 0.533 | 0.140 | 0.073 |
| case9241pegase | 32 | 0.251 | 0.533 | 0.533 | 0.296 | 0.230 | 0.540 | 0.540 | 0.276 | 0.196 |
| case9241pegase | 64 | 0.297 | 0.535 | 0.535 | 0.358 | 0.288 | 0.534 | 0.534 | 0.132 | 0.064 |
| case13659pegase | 32 | 0.280 | 0.545 | 0.545 | 0.311 | 0.221 | 0.511 | 0.511 | 0.241 | 0.219 |
| case13659pegase | 64 | 0.289 | 0.533 | 0.533 | 0.339 | 0.273 | 0.541 | 0.541 | 0.173 | 0.090 |
| case6468rte | 32 | 0.206 | 0.507 | 0.507 | 0.263 | 0.248 | 0.501 | 0.501 | 0.235 | 0.232 |
| case6468rte | 64 | 0.238 | 0.507 | 0.507 | 0.304 | 0.291 | 0.504 | 0.504 | 0.107 | 0.098 |
