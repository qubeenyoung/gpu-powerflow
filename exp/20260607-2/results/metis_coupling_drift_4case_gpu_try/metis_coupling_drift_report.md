# METIS Coupling Retention and Jacobian Drift Diagnostic

## Answers

1. Strong coupling cut by current METIS blocks: off-block abs ratio mean = `0.0740215543593`, off-block Frobenius ratio mean = `0.203493623049`.
2. Field concentration by cuDSS-dx effect: largest mean off-block effect is `J12` = `0.334207478864`.
3. cuDSS-dx weighted coupling outside blocks: mean offblock_effect_ratio = `0.0767447677956`.
4. Top bus-pair preservation: top-5% coupling kept ratio mean = `0.961795906127`, top-5% effect kept ratio mean = `0.952031197323`.
5. Jacobian numeric drift: mean rel_change_all = `0.0344332512352`, mean rel_change_offblock = `0.0300178922349`.
6. Bus-aware weighted METIS evidence: use the top-effect kept ratio and offblock_effect_ratio above. Low top-effect kept ratio supports the hypothesis; high top-effect kept ratio weakens it.

## Notes

- Partition mode: existing unknown-level METIS, block size 64 unless overridden.
- Effect metric uses diagnostic cuDSS solution `dx` for each dumped `Jk/Fk`.
- `J11=P-theta`, `J12=P-|V|`, `J21=Q-theta`, `J22=Q-|V|`.
