# dx Difference Localization Diagnostic

## Answers

1. Block ILU closer to cuDSS than block-Jacobi: `10/10` case/iteration/block-size pairs by total dx error. Mean dx error ratio BJ=`0.979426941848`, BILU=`0.903265768196`.
2. Theta versus |V|: mean theta error ratio BJ=`0.989739655674`, BILU=`0.921127420092`; mean |V| error ratio BJ=`0.82031546779`, BILU=`0.679709515932`.
3. Largest remaining dx errors under BILU:
   - `case13659pegase` J1 bs16: dx_error_ratio=`0.999772366963`.
   - `case13659pegase` J1 bs32: dx_error_ratio=`0.999643630808`.
   - `case9241pegase` J1 bs16: dx_error_ratio=`0.962660199329`.
   - `case9241pegase` J1 bs32: dx_error_ratio=`0.955650399101`.
   - `case3120sp` J1 bs16: dx_error_ratio=`0.920884654464`.
4. Largest block-local BILU dx errors:
   - `case2383wp` J1 bs32 block 15: dx_error_norm=`6.44837705986`, residual_error_norm=`0.00051766511865`.
   - `case2383wp` J1 bs32 block 53: dx_error_norm=`6.40254423559`, residual_error_norm=`0.00908439200684`.
   - `case2383wp` J1 bs32 block 90: dx_error_norm=`6.27283612208`, residual_error_norm=`0.315017835618`.
   - `case2383wp` J1 bs32 block 118: dx_error_norm=`6.18738767201`, residual_error_norm=`4.1202568204`.
   - `case2383wp` J1 bs32 block 54: dx_error_norm=`5.87890531246`, residual_error_norm=`0.131710562525`.
5. Equation-space residual error reduction: `9/10` pairs improved. Mean residual-error norm BJ=`11.1622044498`, BILU=`5.17141880686`.

## Strongest equation-space improvements

- `case13659pegase` J1 bs16 bus 2791: residual improvement=`11.0991415214`, dx improvement=`1.95131104467e-05`.
- `case3120sp` J1 bs32 bus 3028: residual improvement=`8.92129386564`, dx improvement=`0.751034857901`.
- `case3120sp` J1 bs16 bus 107: residual improvement=`5.42848704032`, dx improvement=`0.00169549176055`.
- `case13659pegase` J1 bs32 bus 3169: residual improvement=`4.60828938337`, dx improvement=`-0.034002666705`.
- `case2383wp` J1 bs16 bus 100: residual improvement=`3.12183611794`, dx improvement=`0.0306625361424`.

## Notes

- BJ uses BiCGSTAB(2) + current unknown-level METIS block-Jacobi.
- BILU uses BiCGSTAB(2) + CPU pilot block ILU(0), block coloring order.
- `residual_error = J * (dx_iter - dx_cudss)`; this localizes the equation-space difference from the direct Newton correction.
- Bus IDs are internal dump bus indices.
