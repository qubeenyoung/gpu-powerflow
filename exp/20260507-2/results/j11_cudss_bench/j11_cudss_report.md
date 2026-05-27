# J11 cuDSS Cost Bench

## Answers

1. J11 square for selected cases: `15/15` measured rows succeeded. Failed rows, if any, are marked in the CSV.
2. J11 size versus full J: mean n ratio=`0.539860978557`, mean nnz ratio=`0.280911644236`.
3. J11 factorize+solve cheaper than full J: J1 mean ratio=`0.603186386638`, all J0/J1/J2 mean=`0.575628626949`, all median=`0.602679468431`.
4. J11 solve-only cheaper than full J solve: J1 mean ratio=`0.668386536519`, all mean=`0.644277524043`. Including analysis, J1 total ratio=`0.888833758002`, all total ratio=`0.887967596739`.
5. J0/J1/J2 timing variation: worst within-case factor+solve ratio spread=`0.639032443246`.
6. Decision by requested rule: `maybe worth testing only if it strongly reduces fallback`.

## J1 Snapshot

| case | ratio n | ratio nnz | ratio factor+solve | ratio solve | full f+s ms | J11 f+s ms |
|---|---:|---:|---:|---:|---:|---:|
|case2383wp|0.536728255971|0.291956662122|0.687147866994|0.69853779933|0.5663698|0.3891798|
|case3120sp|0.520614254715|0.273343943228|0.703359571147|0.763372538054|0.6094766|0.4286812|
|case9241pegase|0.542380840573|0.290884925664|0.5128896709|0.631860318349|1.2266954|0.6291594|
|case13659pegase|0.588073196986|0.291385952159|0.509855355717|0.578989896818|1.7658216|0.9003136|
|case6468rte|0.511508344538|0.256986738004|0.602679468431|0.669172130046|0.9804482|0.590896|

## Notes

- Timings exclude dump parsing and H2D buffer creation for both full J and J11.
- Each measurement creates a fresh cuDSS solver object and measures analysis, factorization, and solve phases separately.
- J11 is extracted as the existing field ordering top-left `P-theta` block. No NR update or hybrid integration is performed.
