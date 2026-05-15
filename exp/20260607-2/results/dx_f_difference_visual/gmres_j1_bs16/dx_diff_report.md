# dx Difference Localization Diagnostic

## Answers

1. Block ILU closer to cuDSS than block-Jacobi: `5/5` case/iteration/block-size pairs by total dx error. Mean dx error ratio BJ=`0.967677853182`, BILU=`0.906267063955`.
2. Theta versus |V|: mean theta error ratio BJ=`0.978918603656`, BILU=`0.923818985749`; mean |V| error ratio BJ=`0.765429407185`, BILU=`0.679593546779`.
3. Largest remaining dx errors under BILU:
   - `case13659pegase` J1 bs16: dx_error_ratio=`0.999705900766`.
   - `case9241pegase` J1 bs16: dx_error_ratio=`0.963258110019`.
   - `case3120sp` J1 bs16: dx_error_ratio=`0.905855903677`.
   - `case2383wp` J1 bs16: dx_error_ratio=`0.875651713635`.
   - `case6468rte` J1 bs16: dx_error_ratio=`0.786863691679`.
4. Largest block-local BILU dx errors:
   - `case2383wp` J1 bs16 block 89: dx_error_norm=`5.0409355906`, residual_error_norm=`1.56998184426e-06`.
   - `case2383wp` J1 bs16 block 161: dx_error_norm=`4.8247081398`, residual_error_norm=`0.000172175861249`.
   - `case2383wp` J1 bs16 block 221: dx_error_norm=`4.81026916819`, residual_error_norm=`1.38140582382`.
   - `case2383wp` J1 bs16 block 88: dx_error_norm=`4.63131703488`, residual_error_norm=`3.16855313376e-05`.
   - `case2383wp` J1 bs16 block 219: dx_error_norm=`4.52515905495`, residual_error_norm=`1.07606945583`.
5. Equation-space residual error reduction: `5/5` pairs improved. Mean residual-error norm BJ=`10.0481208608`, BILU=`4.53538198602`.

## Strongest equation-space improvements

- `case13659pegase` J1 bs16 bus 2791: residual improvement=`6.40820661168`, dx improvement=`0.00494073624991`.
- `case3120sp` J1 bs16 bus 107: residual improvement=`5.22904717448`, dx improvement=`0.00637390589217`.
- `case2383wp` J1 bs16 bus 100: residual improvement=`2.98522950104`, dx improvement=`0.036491713596`.
- `case9241pegase` J1 bs16 bus 7917: residual improvement=`0.516920158545`, dx improvement=`-0.0015528030033`.
- `case6468rte` J1 bs16 bus 4736: residual improvement=`0.000207157238302`, dx improvement=`3.96398713853e-05`.

## Notes

- BJ uses the selected fixed-iteration Krylov solver with current unknown-level METIS block-Jacobi.
- BILU uses the selected fixed-iteration Krylov solver with CPU pilot block ILU(0), block coloring order.
- `residual_error = J * (dx_iter - dx_cudss)`; this localizes the equation-space difference from the direct Newton correction.
- Bus IDs are internal dump bus indices.
