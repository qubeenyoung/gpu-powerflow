# Block ILU(0) Symbolic Feasibility

This is symbolic only: no numeric block factorization, no triangular solve, and no hybrid NR run was performed.

## Pattern Check

- case2383wp: checked J0/J1/J2; pattern_same_as_J1=true
- case3120sp: checked J0/J1/J2; pattern_same_as_J1=true
- case9241pegase: checked J0/J1/J2; pattern_same_as_J1=true
- case13659pegase: checked J0/J1/J2; pattern_same_as_J1=true
- case6468rte: checked J0/J1/J2; pattern_same_as_J1=true

## Block Counts

| block size | min blocks | max blocks | mean blocks |
|---:|---:|---:|---:|
|8|564|3026|1622.2|
|16|279|1467|798.6|
|32|139|730|397.4|

## Ordering Summary

| block size | ordering | mean L levels | mean U levels | mean avg width | mean apply / BJ | mean factor / BJ setup | rejects |
|---:|---|---:|---:|---:|---:|---:|---:|
|8|block_coloring|14|14|116.238744589|7.09271668104|9.85319117899|0|
|8|block_metis_nd|32.6|32.6|46.0287428684|7.09271668104|9.8537960149|0|
|32|block_coloring|8|8|46.806984127|7.39341953355|9.4510880256|0|
|16|block_coloring|11.2|11.2|67.1623015873|7.39522997889|9.84371107888|0|
|32|block_metis_nd|18.8|18.8|19.8618649565|7.39341953355|9.4512535764|0|
|8|block_rcm|78.2|78.2|21.1957070594|7.09271668104|9.85293059338|0|
|16|block_metis_nd|29.8|29.8|25.7881109558|7.39522997889|9.84311918823|0|
|8|current_metis_block_order|95.4|95.4|16.7764200289|7.09271668104|9.85435137842|0|
|16|block_rcm|62.6|62.6|13.1717113912|7.39522997889|9.84495224803|0|
|16|current_metis_block_order|74.2|74.2|10.4667084571|7.39522997889|9.8437843853|0|
|32|block_rcm|52.2|52.2|7.26598392009|7.39341953355|9.45153827848|0|
|32|current_metis_block_order|61.8|61.8|6.13860934664|7.39341953355|9.45081303878|1|

## Answers

1. Shallowest levels: `block_coloring` at block size `8` has the best aggregate symbolic score.
2. Widest levels: `block_coloring` at block size `8` has the largest mean level width.
3. Coloring width: block coloring mean width is `116.238744589`; compare with the table above before treating it as parallel enough.
4. Obviously infeasible rows: `1` of `60` candidates were classified as reject by level depth, width, and work ratio.
5. Numeric pilot recommendation: at most these candidates are worth considering:
   - block_size=`8`, ordering=`block_coloring`, mean L/U levels=`14/14`, mean apply/BJ=`7.09271668104`, mean factor/BJ setup=`9.85319117899`
   - block_size=`32`, ordering=`block_coloring`, mean L/U levels=`8/8`, mean apply/BJ=`7.39341953355`, mean factor/BJ setup=`9.4510880256`
   - block_size=`16`, ordering=`block_coloring`, mean L/U levels=`11.2/11.2`, mean apply/BJ=`7.39522997889`, mean factor/BJ setup=`9.84371107888`
