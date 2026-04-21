# TODO

## Phase 0: Skeleton

- [x] Add top-level `CMakeLists.txt`
- [x] Add `src/CMakeLists.txt`
- [x] Add CUDA/C++ utility headers for error checking and device buffers
- [x] Add `block_ilu_probe` executable shell

## Phase 1: cuPF Reduced Jacobian

- [x] Implement reduced index model
- [x] Implement full `J` CSR pattern builder
- [x] Implement `J11` and `J22` CSR pattern builders
- [x] Implement edge-based FP64 value assembly kernel
- [x] Add CPU download/debug norm check against `jacobian_analysis`

## Phase 2: cuSPARSE ILU Blocks

- [x] Implement `CusparseIlu0Block`
- [x] Add zero-pivot reporting
- [x] Validate `J11` factorization on `case_ACTIVSg200`
- [x] Validate `J22` factorization on `case_ACTIVSg200`

## Phase 3: Preconditioned BiCGSTAB

- [x] Implement full-J CSR SpMV
- [x] Implement right-preconditioned BiCGSTAB
- [x] Track residual and breakdown reasons
- [x] Verify finite solve on one Newton step

## Phase 4: Newton Loop

- [x] Implement CUDA mismatch kernel or reuse cuPF equivalent
- [x] Implement CUDA voltage update or reuse cuPF equivalent
- [x] Wire Newton loop: mismatch, Jacobian update, ILU update, BiCGSTAB, voltage update
- [ ] Add summary CSV and step trace CSV

## Phase 5: Smoke Runs

- [x] `case_ACTIVSg200`
- [ ] `case_ACTIVSg500`
- [ ] `case_ACTIVSg2000`

Record failures explicitly rather than hiding them.
