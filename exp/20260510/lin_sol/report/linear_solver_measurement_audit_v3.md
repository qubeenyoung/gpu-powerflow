# Linear Solver Measurement Audit v3

## 1. Why This Audit Was Needed

The v1/v2 benchmark established useful evidence, but several rows mixed different measurement meanings: reusable direct factorization, monolithic one-shot direct solves, iterative setup/solve, and MPI/hybrid external direct solves. This audit checks wrapper correctness, residual interpretation, timing phase visibility, and reasonable best-effort configurations before using the results as annual-report evidence.

A key correction is residual interpretation. Final Newton iterations can have very small `||rhs||`, so a large relative residual can coexist with a tiny absolute residual. The audit therefore reports absolute and scaled residuals in addition to the original relative residual.

## 2. Prior Result Summary

- cuDSS and cuSolverSP were installed and ran in v1.
- AMGx ran a fixed GMRES+AMG configuration in v1.
- Ginkgo was added in v2 with CUDA executor, GMRES+Jacobi, and BiCGSTAB+Jacobi.
- STRUMPACK was added in v2 and ran at `np=1`; `np=2`/`np=4` timed out or hung.
- SuperLU_DIST built with CUDA/MPI in v2 but every run failed with `get_perm_c.c Invalid ISPEC`.

## 3. Environment Consistency

| item | value |
| --- | --- |
| hostname | 7d44605f53c6 |
| os | Linux-6.8.0-110-generic-x86_64-with-glibc2.35 |
| gpu | NVIDIA GeForce RTX 3090, 24576 MiB, 580.126.09 |
| nvcc | nvcc: NVIDIA (R) Cuda compiler driver Copyright (c) 2005-2025 NVIDIA Corporation Built on Fri_Feb_21_20:23:50_PST_2025 Cuda compilation t... |
| gcc | gcc (Ubuntu 12.3.0-1ubuntu1~22.04.2) 12.3.0 |
| g++ | g++ (Ubuntu 12.3.0-1ubuntu1~22.04.2) 12.3.0 |
| cmake | cmake version 3.22.1 |
| which_mpicc | /usr/bin/mpicc |
| which_mpicxx | /usr/bin/mpicxx |
| which_mpirun | /usr/bin/mpirun |
| mpicc_show | gcc -I/usr/lib/x86_64-linux-gnu/openmpi/include -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -L/usr/lib/x86_64-linux-gnu/openmpi/l... |
| mpicxx_show | g++ -I/usr/lib/x86_64-linux-gnu/openmpi/include -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -L/usr/lib/x86_64-linux-gnu/openmpi/l... |
| mpirun_version | mpirun (Open MPI) 4.1.2  Report bugs to http://www.open-mpi.org/community/help/ |
| local_mpich_mpirun | /workspace/gpu-powerflow/exp/20260510/lin_sol/third_party/mpich/install/bin/mpirun |
| local_mpich_version | HYDRA build details:     Version:                                 4.2.3     Release Date:                            Wed Oct  2 09:35:21 ... |
| system_mpich_mpirun | /usr/bin/mpirun.mpich |
| system_mpich_version | HYDRA build details:     Version:                                 4.0     Release Date:                            Fri Jan 21 10:42:29 CS... |
| LD_LIBRARY_PATH | /usr/local/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib64 |
| CUDA_VISIBLE_DEVICES |  |

MPI consistency matters here. Default `mpicc/mpicxx/mpirun` resolve to OpenMPI, while SuperLU_DIST audit executables are linked to the local MPICH install and STRUMPACK is linked to system MPICH. The audit launch commands use matching MPICH launchers for MPI wrappers.

## 4. Measurement Validity Criteria

- The wrapper must solve `J dx = rhs` with the Matrix Market orientation as loaded.
- CSR/CSC conversion and index base must match the library expectation.
- Direct solvers with reusable symbolic analysis must not be compared blindly against monolithic one-shot APIs.
- Iterative solver rows are best-effort only after a small reproducible configuration grid, not one hand-picked config.
- Accuracy is judged with relative, absolute, scaled residual, and reference error.

## 5. Dataset and Residual Metric Audit

| system | iteration | matrix_rows | nnz | rhs_norm_2 | relative_residual_2 | scaled_residual_2 | symmetry | condition_estimate | warnings |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| synthetic_validation | 0 | 5 | 16 | 2.42383992870816467e+00 | 4.58043046273634396e-17 | 4.58043046273634396e-17 | nonsymmetric | 3.279362e+00 |  |
| case14 | 0 | 22 | 146 | 4.54698142848659337e-02 | 1.08144450591845180e-16 | 4.91730808435006005e-18 | nonsymmetric | 1.178884e+02 |  |
| case14 | 1 | 22 | 146 | 5.76195742157490311e-05 | 3.08626116379712110e-16 | 1.77829054176592205e-20 | nonsymmetric | 1.177823e+02 |  |
| case14 | 2 | 22 | 146 | 1.34364512686632269e-10 | 3.29322702386360841e-16 | 4.42492844227882076e-26 | nonsymmetric | 1.177822e+02 | tiny_rhs_norm_relative_residual_sensitive |
| case118 | 0 | 181 | 1051 | 1.42135451223418974e+00 | 2.55863263143996347e-16 | 2.55863263143996347e-16 | nonsymmetric | 3.174119e+03 |  |
| case118 | 1 | 181 | 1051 | 2.45098293316220808e-02 | 7.82136848793352936e-16 | 1.91700406778977868e-17 | nonsymmetric | 3.166578e+03 |  |
| case118 | 3 | 181 | 1051 | 1.60485013837640700e-12 | 6.93673417701658648e-16 | 1.11324188038654205e-27 | nonsymmetric | 3.166930e+03 | tiny_rhs_norm_relative_residual_sensitive |
| case300 | 0 | 530 | 3736 | 1.31496279781897591e+01 | 7.62968805542398048e-16 | 7.62968805542398048e-16 | nonsymmetric | 1.163697e+05 |  |
| case300 | 1 | 530 | 3736 | 4.01957582807602076e+00 | 5.26440546595599205e-16 | 5.26440546595599205e-16 | nonsymmetric | 1.160624e+05 |  |
| case300 | 5 | 530 | 3736 | 2.59608459404205292e-12 | 2.67839610588731095e-15 | 6.95334286723627483e-27 | nonsymmetric | 1.163725e+05 | tiny_rhs_norm_relative_residual_sensitive |
| case1354pegase | 0 | 2447 | 15803 | 2.98280723282129898e+01 | 1.10234191154185599e-15 | 1.10234191154185599e-15 | nonsymmetric |  |  |
| case1354pegase | 1 | 2447 | 15803 | 1.08314818812386271e+00 | 6.19489572266452338e-15 | 6.19489572266452338e-15 | nonsymmetric |  |  |
| case1354pegase | 4 | 2447 | 15803 | 1.89700472471860058e-11 | 2.35583101015105463e-14 | 4.46902255689514424e-25 | nonsymmetric |  | tiny_rhs_norm_relative_residual_sensitive |
| case2869pegase | 0 | 5227 | 36591 | 8.88916439113170469e+01 | 4.97693418575819890e-15 | 4.97693418575819890e-15 | nonsymmetric |  |  |
| case2869pegase | 1 | 5227 | 36591 | 8.94266369789029625e+01 | 2.05787769316950491e-15 | 2.05787769316950491e-15 | nonsymmetric |  |  |
| case2869pegase | 6 | 5227 | 36591 | 3.05516607564782748e-09 | 2.57431423436600969e-15 | 7.86495751689234349e-24 | nonsymmetric |  | tiny_rhs_norm_relative_residual_sensitive |
| case9241pegase | 0 | 17036 | 129412 | 1.08644859268653661e+02 | 1.33955753754819821e-14 | 1.33955753754819821e-14 | nonsymmetric |  |  |
| case9241pegase | 1 | 17036 | 129412 | 8.84101280605746638e+01 | 6.17262709704225519e-15 | 6.17262709704225519e-15 | nonsymmetric |  |  |
| case9241pegase | 6 | 17036 | 129412 | 3.02198551962519260e-09 | 2.93365347646028766e-15 | 8.86545832546109495e-24 | nonsymmetric |  | tiny_rhs_norm_relative_residual_sensitive |

Dataset warnings are expected for final Newton iterations with tiny right-hand sides; those rows must be interpreted using absolute and scaled residuals.

## 6. Wrapper Correctness Audit

| solver | previous_classification | audit_classification | phase_visibility | wrapper_findings | remaining_issue |
| --- | --- | --- | --- | --- | --- |
| cuDSS | valid_as_run | valid_as_run | analysis_factor_solve | Uses cuDSS general CSR, base-zero, analysis once, repeated factorization/solve, CUDA events, CPU residual after D2H. | Relative residual alone is misleading on tiny rhs; audit adds absolute/scaled residual interpretation. |
| cuSolverSP | valid_as_run | valid_as_monolithic_qr | monolithic | Uses csrlsvqr for general nonsymmetric CSR; CUDA 12.8 LU/Cholesky sparse APIs are deprecated toward cuDSS, and Cholesky is inappropriate ... | Not equivalent to reusable symbolic/numeric factorization; cuSolverRF requires externally supplied LU factors. |
| AMGx | valid_but_not_best_effort | valid_limited_grid | setup_solve | Wrapper times AMGx setup as analysis and solve separately. Audit tests a finite AMG/Krylov grid rather than one fixed config. | Wrapper records fixed tolerance/preconditioner labels in JSON; config_audit.csv records actual audit config names. |
| Ginkgo | valid_but_not_best_effort | valid_as_jacobi_only | setup_solve | CUDA executor is used, but make_solver_factory always installs Jacobi and supports only GMRES/BiCGSTAB despite config parser accepting a ... | IDR/ILU/ParILU/ParILUT/ISAI headers are available but were not wired into the existing benchmark wrapper. |
| STRUMPACK | valid_as_run_np1_with_metric_caveat | valid_external_hybrid_np1 | analysis_factor_solve | MPIDist direct path with host distributed CSR input/output and internal GPU enablement. Build has CUDA and OpenMP, no SLATE; default comp... | np=2/np=4 still treated as timeout/hang risk; full GPU residency is not demonstrated without SLATE. |
| SuperLU_DIST | runtime_failed_needs_diagnosis | runtime_fixed_for_supported_permutation_but_original_wrapper_invalid | monolithic | Previous Invalid ISPEC came from METIS_AT_PLUS_A while HAVE_PARMETIS is disabled. Audit variants using supported ColPerm avoid that error... | Use as diagnostic/external baseline only until wrapper is rewritten with explicit state reuse/restoration and phase separation. |

## 7. Correctness Harness Results

| system | solver | dtype | status | relative_residual_2 | absolute_residual_2 | scaled_residual_2 | relative_error_2 | warnings |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| audit_nonsym_known | cuDSS | fp64 | ok | 0 | 0.0 | 0.0 | 0 |  |
| audit_nonsym_known | cuSolverSP | fp64 | ok | 4.0579010796675683e-16 | 2.286089127109703e-15 | 4.0579010796675683e-16 | 5.9116578208534125e-16 |  |
| audit_nonsym_known | AMGx | fp64 | ok | 1.182414584857616e-16 | 6.661338147750939e-16 | 1.182414584857616e-16 | 1.7450602029919786e-16 |  |
| audit_nonsym_known | Ginkgo-GMRES-Jacobi | fp64 | ok | 1.848672001153747e-16 | 1.041481514324134e-15 | 1.848672001153747e-16 | 2.589742806986193e-16 |  |
| audit_nonsym_known | cuDSS | fp32 | ok | 1.7192608239479016e-08 | 9.685754776000978e-08 | 1.7192608239479016e-08 | 2.0444224493802645e-08 |  |
| audit_nonsym_known | cuSolverSP | fp32 | ok | 2.1857281994400526e-07 | 1.2313679839544829e-06 | 2.1857281994400526e-07 | 2.7275997524789815e-07 |  |
| audit_nonsym_known | AMGx | fp32 | ok | 5.301081244372528e-08 | 2.98645628781025e-07 | 5.301081244372528e-08 | 6.217868135595391e-08 |  |
| audit_nonsym_known | Ginkgo-GMRES-Jacobi | fp32 | ok | 1.0083560109983563e-07 | 5.680748908713913e-07 | 1.0083560109983563e-07 | 1.247767816520736e-07 |  |
| audit_nonsym_known | SuperLU_DIST | fp64 | ok | 6.826673788612842e-17 | 3.8459253727671276e-16 | 6.826673788612842e-17 | 1.1424100674951064e-16 | variant=natural |
| audit_nonsym_known | SuperLU_DIST | fp64 | ok | 5.573955807511313e-17 | 3.1401849173675503e-16 | 5.573955807511313e-17 | 1.0937743667935628e-16 | variant=natural_norowperm |
| audit_nonsym_known | SuperLU_DIST | fp64 | ok | 3.941381949525387e-17 | 2.220446049250313e-16 | 3.941381949525387e-17 | 7.616067116634043e-17 | variant=mmd_at_plus_a |
| audit_nonsym_known | SuperLU_DIST | fp64 | ok | 5.573955807511313e-17 | 3.1401849173675503e-16 | 5.573955807511313e-17 | 8.51502189699727e-17 | variant=mmd_at_plus_a_norowperm |
| audit_nonsym_known | SuperLU_DIST | fp64 | ok | 1.6250734088796026e-16 | 9.155133597044475e-16 | 1.6250734088796026e-16 | 1.5232134233268086e-16 | variant=mmd_ata |
| audit_nonsym_known | SuperLU_DIST | fp64 | ok | 5.573955807511313e-17 | 3.1401849173675503e-16 | 5.573955807511313e-17 | 9.327739141271671e-17 | variant=mmd_ata_norowperm |
| audit_signed_values | cuDSS | fp64 | ok | 0 | 0.0 | 0.0 | 5.988097684046442e-17 |  |
| audit_signed_values | cuSolverSP | fp64 | ok | 6.487179526815918e-16 | 3.5596458096434965e-15 | 6.487179526815918e-16 | 8.51068587276245e-16 |  |
| audit_signed_values | AMGx | fp64 | ok | 2.8035607668927586e-16 | 1.538370149106851e-15 | 2.8035607668927586e-16 | 1.821208811014162e-16 |  |
| audit_signed_values | Ginkgo-GMRES-Jacobi | fp64 | ok | 3.7087622873278825e-16 | 2.0350724194510405e-15 | 3.7087622873278825e-16 | 3.0089818465894397e-16 |  |
| audit_signed_values | cuDSS | fp32 | ok | 6.460668503588115e-08 | 3.5450986782819987e-07 | 6.460668503588115e-08 | 8.037088661947752e-08 |  |
| audit_signed_values | cuSolverSP | fp32 | ok | 1.7534169074872717e-07 | 9.62135103783484e-07 | 1.7534169074872717e-07 | 1.1702177729774031e-07 |  |
| audit_signed_values | AMGx | fp32 | ok | 1.5621316296872363e-07 | 8.571730266970191e-07 | 1.5621316296872363e-07 | 1.481965924520638e-07 |  |
| audit_signed_values | Ginkgo-GMRES-Jacobi | fp32 | ok | 1.4341464411163177e-07 | 7.869449810094174e-07 | 1.4341464411163177e-07 | 1.4555786644658883e-07 |  |
| audit_signed_values | SuperLU_DIST | fp64 | ok | 8.09318281727504e-17 | 4.440892098500626e-16 | 8.09318281727504e-17 | 5.988097684046442e-17 | variant=natural |
| audit_signed_values | SuperLU_DIST | fp64 | ok | 1.8096906933760248e-16 | 9.930136612989092e-16 | 1.8096906933760248e-16 | 1.3473219789104493e-16 | variant=natural_norowperm |
| audit_signed_values | SuperLU_DIST | fp64 | ok | 0 | 0.0 | 0.0 | 0 | variant=mmd_at_plus_a |
| audit_signed_values | SuperLU_DIST | fp64 | ok | 0 | 0.0 | 0.0 | 0 | variant=mmd_at_plus_a_norowperm |
| audit_signed_values | SuperLU_DIST | fp64 | ok | 1.8543811436639413e-16 | 1.0175362097255202e-15 | 1.8543811436639413e-16 | 1.612344645490526e-16 | variant=mmd_ata |
| audit_signed_values | SuperLU_DIST | fp64 | ok | 2.32458978518608e-16 | 1.2755491433176288e-15 | 2.32458978518608e-16 | 1.9171266708416637e-16 | variant=mmd_ata_norowperm |
| audit_singular_expected_fail | cuDSS | fp64 | ok | 3.14018491736755e-17 | 2.220446049250313e-16 | 3.14018491736755e-17 | 0.5 | expected_fail_matrix |
| audit_singular_expected_fail | cuSolverSP | fp64 | ok | None | None | None | None | expected_fail_matrix |
| audit_singular_expected_fail | correctness_AMGx | fp64 | runtime_failed | None | None | None | None | expected_fail_matrix |
| audit_singular_expected_fail | Ginkgo-GMRES-Jacobi | fp64 | ok | 6.2803698347351e-17 | 4.440892098500626e-16 | 6.2803698347351e-17 | 2.5 | expected_fail_matrix |
| audit_singular_expected_fail | cuDSS | fp32 | ok | 3.769728732336321e-08 | 2.66560074986878e-07 | 3.769728732336321e-08 | 0.6708203634356997 | expected_fail_matrix |
| audit_singular_expected_fail | cuSolverSP | fp32 | ok | 2.0733508027703866e-07 | 1.4660804124175124e-06 | 2.0733508027703866e-07 | 1.0000001430511678 | expected_fail_matrix |
| audit_singular_expected_fail | correctness_AMGx | fp32 | runtime_failed | None | None | None | None | expected_fail_matrix |
| audit_singular_expected_fail | Ginkgo-GMRES-Jacobi | fp32 | ok | 1.5993604499100135e-07 | 1.130918619692938e-06 | 1.5993604499100135e-07 | 2.500000429153449 | expected_fail_matrix |
| audit_singular_expected_fail | SuperLU_DIST | fp64 | ok | 3.8209946349085597 | 27.018512172212592 | 3.8209946349085597 | 2.489979919597666 | variant=natural;expected_fail_matrix |
| audit_singular_expected_fail | SuperLU_DIST | fp64 | ok | 3.8209946349085597 | 27.018512172212592 | 3.8209946349085597 | 2.489979919597666 | variant=natural_norowperm;expected_fail_matrix |
| audit_singular_expected_fail | SuperLU_DIST | fp64 | ok | 3.8209946349085597 | 27.018512172212592 | 3.8209946349085597 | 2.489979919597666 | variant=mmd_at_plus_a;expected_fail_matrix |
| audit_singular_expected_fail | SuperLU_DIST | fp64 | ok | 3.8209946349085597 | 27.018512172212592 | 3.8209946349085597 | 2.489979919597666 | variant=mmd_at_plus_a_norowperm;expected_fail_matrix |
| audit_singular_expected_fail | SuperLU_DIST | fp64 | ok | 3.8209946349085597 | 27.018512172212592 | 3.8209946349085597 | 2.489979919597666 | variant=mmd_ata;expected_fail_matrix |
| audit_singular_expected_fail | SuperLU_DIST | fp64 | ok | 3.8209946349085597 | 27.018512172212592 | 3.8209946349085597 | 2.489979919597666 | variant=mmd_ata_norowperm;expected_fail_matrix |
| audit_csr_known | cuDSS | fp64 | ok | 0 | 0.0 | 0.0 | 0.6982367979695577 |  |
| audit_csr_known | cuSolverSP | fp64 | ok | None | None | None | None |  |
| audit_csr_known | correctness_AMGx | fp64 | runtime_failed | None | None | None | None |  |
| audit_csr_known | Ginkgo-GMRES-Jacobi | fp64 | ok | 2.340894341276439e-16 | 5.329070518200751e-15 | 2.340894341276439e-16 | 0.6982367979695577 |  |
| audit_csr_known | cuDSS | fp32 | ok | 2.631304664333491e-08 | 5.990192664337689e-07 | 2.631304664333491e-08 | 0.6982367952432353 |  |
| audit_csr_known | cuSolverSP | fp32 | ok | None | None | None | None |  |
| audit_csr_known | correctness_AMGx | fp32 | runtime_failed | None | None | None | None |  |
| audit_csr_known | Ginkgo-GMRES-Jacobi | fp32 | ok | 1.9810625775225705e-07 | 4.509909734255539e-06 | 1.9810625775225705e-07 | 0.6982367804432112 |  |
| audit_csr_known | SuperLU_DIST | fp64 | runtime_failed | None | None | None | None | variant=natural |
| audit_csr_known | SuperLU_DIST | fp64 | runtime_failed | None | None | None | None | variant=natural_norowperm |
| audit_csr_known | SuperLU_DIST | fp64 | runtime_failed | None | None | None | None | variant=mmd_at_plus_a |
| audit_csr_known | SuperLU_DIST | fp64 | runtime_failed | None | None | None | None | variant=mmd_at_plus_a_norowperm |
| audit_csr_known | SuperLU_DIST | fp64 | runtime_failed | None | None | None | None | variant=mmd_ata |
| audit_csr_known | SuperLU_DIST | fp64 | runtime_failed | None | None | None | None | variant=mmd_ata_norowperm |
| audit_pf_sign_convention | cuDSS | fp64 | ok | 1.4585121575861745e-16 | 3.8459253727671276e-16 | 1.4585121575861745e-16 | 8.368620923398063e-17 |  |
| audit_pf_sign_convention | cuSolverSP | fp64 | ok | 2.561063181109021e-16 | 6.753223014464259e-16 | 2.561063181109021e-16 | 4.556865862772612e-16 |  |
| audit_pf_sign_convention | AMGx | fp64 | ok | 2.947253353795519e-16 | 7.771561172376096e-16 | 2.947253353795519e-16 | 3.589738454728226e-16 |  |
| audit_pf_sign_convention | Ginkgo-GMRES-Jacobi | fp64 | ok | 2.7928381533303753e-16 | 7.364386412590295e-16 | 2.7928381533303753e-16 | 3.765879415529128e-16 |  |
| audit_pf_sign_convention | cuDSS | fp32 | ok | 1.950628269610547e-08 | 5.143577800097148e-08 | 1.950628269610547e-08 | 2.6957214889541084e-08 |  |
| audit_pf_sign_convention | cuSolverSP | fp32 | ok | 6.313075733363305e-08 | 1.6646839737918223e-07 | 6.313075733363305e-08 | 9.086265530473645e-08 |  |
| audit_pf_sign_convention | AMGx | fp32 | ok | 9.64456848868844e-08 | 2.5431595113629093e-07 | 9.64456848868844e-08 | 9.477717851962118e-08 |  |
| audit_pf_sign_convention | Ginkgo-GMRES-Jacobi | fp32 | ok | 1.3266048279137542e-07 | 3.498101226494184e-07 | 1.3266048279137542e-07 | 1.4186372177314197e-07 |  |
| audit_pf_sign_convention | SuperLU_DIST | fp64 | ok | 1.4585121575861745e-16 | 3.8459253727671276e-16 | 1.4585121575861745e-16 | 1.871280526264513e-16 | variant=natural |
| audit_pf_sign_convention | SuperLU_DIST | fp64 | ok | 1.4585121575861745e-16 | 3.8459253727671276e-16 | 1.4585121575861745e-16 | 2.2455366315174155e-16 | variant=natural_norowperm |
| audit_pf_sign_convention | SuperLU_DIST | fp64 | ok | 1.4585121575861745e-16 | 3.8459253727671276e-16 | 1.4585121575861745e-16 | 2.619792736770318e-16 | variant=mmd_at_plus_a |
| audit_pf_sign_convention | SuperLU_DIST | fp64 | ok | 1.1908701899106324e-16 | 3.1401849173675503e-16 | 1.1908701899106324e-16 | 1.6841524736380615e-16 | variant=mmd_at_plus_a_norowperm |
| audit_pf_sign_convention | SuperLU_DIST | fp64 | ok | 1.2631085801980796e-16 | 3.330669073875469e-16 | 1.2631085801980796e-16 | 2.750209472972899e-16 | variant=mmd_ata |
| audit_pf_sign_convention | SuperLU_DIST | fp64 | ok | 1.1908701899106324e-16 | 3.1401849173675503e-16 | 1.1908701899106324e-16 | 8.368620923398063e-17 | variant=mmd_ata_norowperm |

## 8. Solver-by-Solver Audit

### cuDSS

cuDSS remains valid as measured: the wrapper uses the general nonsymmetric sparse matrix path, performs symbolic analysis once, repeats numeric factorization and solve, keeps data GPU-resident after load, and computes CPU residuals after D2H. Its timing should be interpreted two ways: one-shot includes analysis, while Newton-style repeated solves emphasize factorization plus solve after the sparsity pattern is known.

### cuSolverSP / cuSolverRF

CUDA 12.8 headers expose cuSolverSP QR for general sparse systems. The LU and Cholesky sparse solve APIs visible in the headers are deprecated toward cuDSS, and Cholesky is not appropriate for nonsymmetric power-flow Jacobians. cuSolverRF is present but requires externally supplied LU factors, so the existing benchmark is valid as a monolithic QR comparison, not as a reusable refactorization comparison.

### AMGx

The v1 AMGx rows are valid for the fixed configuration. The audit extends this with a finite grid of GMRES/FGMRES, AMG preconditioning, Jacobi/BlockJacobi smoothers, and max iteration values 200/1000 where the wrapper accepts the configuration. Larger Jacobians remain convergence-sensitive; failures are not installation failures but configuration/robustness evidence under this grid.

### Ginkgo

Ginkgo rows are valid as CUDA-executor GMRES/BiCGSTAB with Jacobi. The source audit found that the wrapper parses a preconditioner field but always constructs Jacobi, and it only switches between GMRES and BiCGSTAB. Ginkgo headers include IDR and advanced preconditioners such as ILU/ParILU/ParILUT/ISAI, so the current Ginkgo evidence is limited and not a full best-effort Ginkgo study.

### STRUMPACK

STRUMPACK is valid as an external MPI/hybrid direct baseline at `np=1`: CUDA is enabled, input/output are host-distributed via MPI, and default compression is `NONE`. The build does not enable SLATE, so this is not full GPU residency. The audit preserves the v2 conclusion that `np=2`/`np=4` hangs are runtime/integration issues, not performance data.

### SuperLU_DIST

The v2 SuperLU_DIST failure was not a solver-performance result. The wrapper requested `METIS_AT_PLUS_A`, but the installed SuperLU_DIST config has `HAVE_PARMETIS` disabled, so `get_perm_c` rejects that ISPEC. Audit executables using supported permutations avoid the `Invalid ISPEC` failure. A second issue appeared: repeated in-process ABglobal calls can mutate matrix/solver state, so the original wrapper is invalid for repeated timing. Audit SuperLU_DIST results are one-shot process-level diagnostics until the wrapper is rewritten.

## 9. SuperLU_DIST Diagnosis

Reproduction of prior failure:

```text
Invalid ISPEC at line 556 in file /workspace/gpu-powerflow/exp/20260510/lin_sol/third_party/superlu_dist/src/SRC/prec-independent/get_perm_c.c
```

Installed SuperLU_DIST configuration excerpts:

```text
/* #undef HAVE_PARMETIS */
/* #undef HAVE_COLAMD */
typedef enum {NOROWPERM, LargeDiag_MC64, LargeDiag_HWPM, MY_PERMR} rowperm_t;
typedef enum {NATURAL, MMD_ATA, MMD_AT_PLUS_A, COLAMD,
```

`get_perm_c.c` accepts `METIS_AT_PLUS_A` only inside `#ifdef HAVE_PARMETIS`; otherwise it falls through to `ABORT("Invalid ISPEC")`:

```c
t = SuperLU_timer_();

    switch ( ispec ) {

        case NATURAL: /* Natural ordering */
	      for (i = 0; i < n; ++i) perm_c[i] = i;
#if ( PRNTlevel>=1 )
	      if ( !pnum ) printf(".. Use natural column ordering\n");
#endif
	      return;

        case MMD_AT_PLUS_A: /* Minimum degree ordering on A'+A */
	      if ( m != n ) ABORT("Matrix is not square");
	      at_plus_a_dist(n, Astore->nnz, Astore->colptr, Astore->rowind,
			     &bnz, &b_colptr, &b_rowind);
	      t = SuperLU_timer_() - t;
	      /*printf("Form A'+A time = %8.3f\n", t);*/
#if ( PRNTlevel>=1 )
	      if ( !pnum ) printf(".. Use minimum degree ordering on A'+A.\n");
#endif
	      break;

        case MMD_ATA: /* Minimum degree ordering on A'*A */
	      getata_dist(m, n, Astore->nnz, Astore->colptr, Astore->rowind,
			  &bnz, &b_colptr, &b_rowind);
	      t = SuperLU_timer_() - t;
	      /*printf("Form A'*A time = %8.3f\n", t);*/
#if ( PRNTlevel>=1 )
	      if ( !pnum ) printf(".. Use minimum degree ordering on A'*A\n");
#endif
	      break;

        case (COLAMD): /* Approximate minimum degree column ordering. */
	      get_colamd_dist(m, n, Astore->nnz, Astore->colptr, Astore->rowind,
			      perm_c);
#if ( PRNTlevel>=1 )
	      printf(".. Use approximate minimum degree column ordering.\n");
#endif
	      return;
#ifdef HAVE_PARMETIS
        case METIS_AT_PLUS_A: /* METIS ordering on A'+A */
	      if ( m != n ) ABORT("Matrix is not square");
	      at_plus_a_dist(n, Astore->nnz, Astore->colptr, Astore->rowind,
			     &bnz, &b_colptr, &b_rowind);

	      if ( bnz ) { /* non-empty adjacency structure */
		  get_metis_dist(n, bnz, b_colptr, b_rowind, perm_c);
	      } else { /* e.g., diagonal matrix */
		  for (i = 0; i < n; ++i) perm_c[i] = i;
		  SUPERLU_FREE(b_colptr);
		  /* b_rowind is not allocated in this case */
	      }

#if ( PRNTlevel>=1 )
	      if ( !pnum ) printf(".. Use METIS ordering on A'+A\n");
#endif
	      return;
#endif /* matching ifdef HAVE_PARMETIS */

        default:
	      ABORT("Invalid ISPEC");
    }

    if ( bnz ) {
	t = SuperLU_timer_();
```

Audit repair attempts:

- Built audit executables for `NATURAL`, `MMD_AT_PLUS_A`, and `MMD_ATA`, with both `LargeDiag_MC64` and `NOROWPERM` row permutation variants.
- `Invalid ISPEC` is fixed by avoiding `METIS_AT_PLUS_A` in this no-ParMETIS build.
- Correctness depends on row/column permutation and on not reusing the same ABglobal `SuperMatrix` across repeated in-process solves. The original wrapper therefore remains invalid for repeated timing.


## 10. Best-Effort Configuration Results

| solver | config | system | iteration | dtype | build_status | converged | num_iterations | one_shot_time_ms | factorization_plus_solve_ms | relative_residual_2 | scaled_residual_2 | relative_error_2 | phase_visibility |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AMGx | gmres_amg_block_jacobi_200 | case118 | 0 | fp32 | ok | True | 1 | 8.377892333333333 | 0.23096633333333336 | 3.498885297716759e-07 | 3.498885297716759e-07 | 5.53752062749781e-06 | setup_solve |
| AMGx | gmres_amg_block_jacobi_1000 | case118 | 0 | fp32 | ok | True | 1 | 9.686506333333332 | 0.22670833333333337 | 3.498885297716759e-07 | 3.498885297716759e-07 | 5.53752062749781e-06 | setup_solve |
| AMGx | fgmres_amg_block_jacobi_200 | case118 | 0 | fp32 | ok | True | 1 | 9.922077999999999 | 0.197474 | 3.74906452545249e-07 | 3.74906452545249e-07 | 5.5094533868513245e-06 | setup_solve |
| AMGx | fgmres_amg_block_jacobi_1000 | case118 | 0 | fp32 | ok | True | 1 | 7.232985333333334 | 0.19618133333333335 | 3.74906452545249e-07 | 3.74906452545249e-07 | 5.5094533868513245e-06 | setup_solve |
| AMGx | gmres_amg_block_jacobi_200 | case118 | 0 | fp64 | ok | True | 1 | 7.410337666666667 | 0.3085216666666667 | 7.900117137902635e-16 | 7.900117137902635e-16 | 4.68286702956638e-15 | setup_solve |
| AMGx | gmres_amg_block_jacobi_1000 | case118 | 0 | fp64 | ok | True | 1 | 7.661134 | 0.31165400000000004 | 7.900117137902635e-16 | 7.900117137902635e-16 | 4.68286702956638e-15 | setup_solve |
| AMGx | fgmres_amg_block_jacobi_200 | case118 | 0 | fp64 | ok | True | 1 | 8.285024333333332 | 0.23102333333333333 | 4.462745991975323e-16 | 4.462745991975323e-16 | 4.595788433370475e-15 | setup_solve |
| AMGx | fgmres_amg_block_jacobi_1000 | case118 | 0 | fp64 | ok | True | 1 | 6.937885333333334 | 0.23328433333333334 | 4.462745991975323e-16 | 4.462745991975323e-16 | 4.595788433370475e-15 | setup_solve |
| Ginkgo-BiCGSTAB-Jacobi | bicgstab_jacobi_fp32tol | case118 | 0 | fp32 | ok | True | 37 | 18.448565666666667 | 3.7034776666666667 | 1.0379540098782124e-06 | 1.0379540098782124e-06 | 0.00011274931496946624 | setup_solve |
| Ginkgo-BiCGSTAB-Jacobi | bicgstab_jacobi_fp64tol | case118 | 0 | fp64 | ok | True | 50 | 22.426003 | 7.474919 | 1.7649786379202359e-09 | 1.7649786379202359e-09 | 4.039992696008896e-08 | setup_solve |
| Ginkgo-GMRES-Jacobi | gmres_jacobi_fp32tol | case118 | 0 | fp32 | ok | True | 55 | 29.405914000000003 | 14.66261 | 9.359846452182742e-07 | 9.359846452182742e-07 | 3.156023048859432e-05 | setup_solve |
| Ginkgo-GMRES-Jacobi | gmres_jacobi_fp64tol | case118 | 0 | fp64 | ok | True | 64 | 36.40046533333333 | 21.47276533333333 | 6.771657916774176e-09 | 6.771657916774176e-09 | 9.755079008357625e-08 | setup_solve |
| STRUMPACK | np1_omp1_default_no_compression | case118 | 0 | fp32 | ok | True | 1 | 9.897551 | 9.525460333333333 | 1.5227440957732125e-07 | 1.5227440957732125e-07 | 3.1855061247271206e-06 | analysis_factor_solve |
| STRUMPACK | np1_omp1_default_no_compression | case118 | 0 | fp64 | ok | True | 1 | 10.041765 | 9.680056666666667 | 2.130560475845631e-16 | 2.130560475845631e-16 | 1.1418590840778594e-15 | analysis_factor_solve |
| SuperLU_DIST | natural_largediag_one_shot_process | case118 | 0 | fp64 | ok | True | 1 | 155.803608 | 155.803608 | 2.5129186691453894e-16 | 2.5129186691453894e-16 | 1.0217082623742993e-15 | monolithic |
| SuperLU_DIST | natural_largediag_one_shot_process | case118 | 1 | fp64 | ok | True | 1 | 153.854122 | 153.854122 | 7.105371078967479e-16 | 1.7415143248333633e-17 | 1.2341191738531177e-14 | monolithic |
| SuperLU_DIST | natural_largediag_one_shot_process | case118 | 3 | fp64 | ok | True | 1 | 159.334008 | 159.334008 | 9.641280258995045e-16 | 1.5472809957773918e-27 | 8.448235973933712e-15 | monolithic |
| audit_AMGx_fgmres_amg_jacobi_1000 | fgmres_amg_jacobi_1000 | case118 | 0 | fp32 | runtime_failed | False | -1 |  |  | None | None | None | setup_solve |
| audit_AMGx_fgmres_amg_jacobi_1000 | fgmres_amg_jacobi_1000 | case118 | 0 | fp64 | runtime_failed | False | -1 |  |  | None | None | None | setup_solve |
| audit_AMGx_fgmres_amg_jacobi_200 | fgmres_amg_jacobi_200 | case118 | 0 | fp32 | runtime_failed | False | -1 |  |  | None | None | None | setup_solve |
| audit_AMGx_fgmres_amg_jacobi_200 | fgmres_amg_jacobi_200 | case118 | 0 | fp64 | runtime_failed | False | -1 |  |  | None | None | None | setup_solve |
| audit_AMGx_gmres_amg_jacobi_1000 | gmres_amg_jacobi_1000 | case118 | 0 | fp32 | runtime_failed | False | -1 |  |  | None | None | None | setup_solve |
| audit_AMGx_gmres_amg_jacobi_1000 | gmres_amg_jacobi_1000 | case118 | 0 | fp64 | runtime_failed | False | -1 |  |  | None | None | None | setup_solve |
| audit_AMGx_gmres_amg_jacobi_200 | gmres_amg_jacobi_200 | case118 | 0 | fp32 | runtime_failed | False | -1 |  |  | None | None | None | setup_solve |
| audit_AMGx_gmres_amg_jacobi_200 | gmres_amg_jacobi_200 | case118 | 0 | fp64 | runtime_failed | False | -1 |  |  | None | None | None | setup_solve |
| cuDSS | general_direct_analysis_reuse | case118 | 0 | fp32 | ok | True | 1 | 14.309836688265204 | 0.11694079898297786 | 1.221689337360823e-07 | 1.221689337360823e-07 | 1.746525291221476e-06 | analysis_factor_solve |
| cuDSS | general_direct_analysis_reuse | case118 | 1 | fp32 | ok | True | 1 | 14.38170842602849 | 0.11751679852604867 | 2.3691009822199702e-07 | 5.8066260743589704e-09 | 6.399727514131123e-07 | analysis_factor_solve |
| cuDSS | general_direct_analysis_reuse | case118 | 3 | fp32 | ok | True | 1 | 14.245811042934656 | 0.1175551988184452 | 2.986794872000949e-07 | 4.793358163632666e-19 | 3.5238823328402797e-06 | analysis_factor_solve |
| cuDSS | general_direct_analysis_reuse | case118 | 0 | fp64 | ok | True | 1 | 14.26282261274755 | 0.14243839643895626 | 1.8963749854549074e-16 | 1.8963749854549074e-16 | 8.661664494383737e-16 | analysis_factor_solve |
| cuDSS | general_direct_analysis_reuse | case118 | 1 | fp64 | ok | True | 1 | 14.178166213631629 | 0.1424863964319229 | 6.19595205576312e-16 | 1.5186172743366707e-17 | 2.943436822581063e-15 | analysis_factor_solve |
| cuDSS | general_direct_analysis_reuse | case118 | 3 | fp64 | ok | True | 1 | 14.268521571531892 | 0.14269759692251682 | 9.023394033860818e-16 | 1.448119516386638e-27 | 3.95439326121208e-15 | analysis_factor_solve |
| cuSolverSP | csrlsvqr_monolithic | case118 | 0 | fp32 | ok | True | 1 | 1.2204864025115967 | 1.2204864025115967 | 2.7465300122041504e-07 | 2.7465300122041504e-07 | 2.2024607106174066e-06 | monolithic |
| cuSolverSP | csrlsvqr_monolithic | case118 | 1 | fp32 | ok | True | 1 | 1.2222815871238708 | 1.2222815871238708 | 8.098247128599927e-07 | 1.984866550072828e-08 | 1.1186745342166014e-05 | monolithic |
| cuSolverSP | csrlsvqr_monolithic | case118 | 3 | fp32 | ok | True | 1 | 1.2308096170425415 | 1.2308096170425415 | 1.2842084552617029e-06 | 2.060962117130896e-18 | 1.8404283413707896e-05 | monolithic |
| cuSolverSP | csrlsvqr_monolithic | case118 | 0 | fp64 | ok | True | 1 | 2.3416351795196535 | 2.3416351795196535 | 9.336881699983903e-16 | 9.336881699983903e-16 | 1.935928818183268e-14 | monolithic |
| cuSolverSP | csrlsvqr_monolithic | case118 | 1 | fp64 | ok | True | 1 | 2.335091233253479 | 2.335091233253479 | 1.1832145738303894e-15 | 2.90003872672708e-17 | 2.856573136097942e-14 | monolithic |
| cuSolverSP | csrlsvqr_monolithic | case118 | 3 | fp64 | ok | True | 1 | 2.341980767250061 | 2.341980767250061 | 1.6426103941166277e-15 | 2.6361435182965943e-27 | 1.6578851151638686e-14 | monolithic |
| SuperLU_DIST | natural_largediag_one_shot_process | case1354pegase | 0 | fp64 | ok | True | 1 | 708.013678 | 708.013678 | 1.818897728336111e-15 | 1.818897728336111e-15 | 7.957984810279446e-14 | monolithic |
| SuperLU_DIST | natural_largediag_one_shot_process | case1354pegase | 1 | fp64 | ok | True | 1 | 706.86819 | 706.86819 | 1.314865266827977e-14 | 1.314865266827977e-14 | 5.201597268500276e-14 | monolithic |
| SuperLU_DIST | natural_largediag_one_shot_process | case1354pegase | 4 | fp64 | ok | True | 1 | 713.39573 | 713.39573 | 3.465438684981893e-14 | 6.573953558633265e-25 | 7.431042225385976e-14 | monolithic |
| cuDSS | general_direct_analysis_reuse | case1354pegase | 0 | fp32 | ok | True | 1 | 17.940044263750316 | 0.24718079939484597 | 4.3402617282464434e-07 | 4.3402617282464434e-07 | 4.199669876005997e-06 | analysis_factor_solve |
| cuDSS | general_direct_analysis_reuse | case1354pegase | 1 | fp32 | ok | True | 1 | 17.88145306557417 | 0.24698880165815354 | 2.708391924870816e-06 | 2.708391924870816e-06 | 6.439390447959703e-06 | analysis_factor_solve |
| cuDSS | general_direct_analysis_reuse | case1354pegase | 4 | fp32 | ok | True | 1 | 17.874329613149165 | 0.24735359996557238 | 1.0971181691115819e-05 | 2.0812383503792914e-16 | 5.763961544339926e-05 | analysis_factor_solve |
| cuDSS | general_direct_analysis_reuse | case1354pegase | 0 | fp64 | ok | True | 1 | 19.46132491230965 | 0.3884928047657013 | 9.058779476800043e-16 | 9.058779476800043e-16 | 4.8878088992611755e-14 | analysis_factor_solve |
| cuDSS | general_direct_analysis_reuse | case1354pegase | 1 | fp64 | ok | True | 1 | 18.09359944313765 | 0.33811200112104417 | 5.3704135898142364e-15 | 5.3704135898142364e-15 | 2.47579022722757e-14 | analysis_factor_solve |
| cuDSS | general_direct_analysis_reuse | case1354pegase | 4 | fp64 | ok | True | 1 | 18.0236680701375 | 0.33960320502519603 | 2.0667539126848023e-14 | 3.920641937193724e-25 | 5.208456224553966e-14 | analysis_factor_solve |
| cuSolverSP | csrlsvqr_monolithic | case1354pegase | 0 | fp32 | ok | True | 1 | 12.455583953857422 | 12.455583953857422 | 1.327449653405419e-06 | 1.327449653405419e-06 | 2.507319802931921e-05 | monolithic |
| cuSolverSP | csrlsvqr_monolithic | case1354pegase | 1 | fp32 | ok | True | 1 | 12.41878080368042 | 12.41878080368042 | 9.015135171918373e-06 | 9.015135171918373e-06 | 0.00014796471818346847 | monolithic |
| cuSolverSP | csrlsvqr_monolithic | case1354pegase | 4 | fp32 | ok | True | 1 | 12.426067447662353 | 12.426067447662353 | 2.9289615516248447e-05 | 5.556253901951454e-16 | 0.00017932740306352753 | monolithic |
| cuSolverSP | csrlsvqr_monolithic | case1354pegase | 0 | fp64 | ok | True | 1 | 22.00035171508789 | 22.00035171508789 | 2.2830302055488286e-15 | 2.2830302055488286e-15 | 7.019327948875081e-14 | monolithic |
| cuSolverSP | csrlsvqr_monolithic | case1354pegase | 1 | fp64 | ok | True | 1 | 19.84990711212158 | 19.84990711212158 | 1.7873395467370382e-14 | 1.7873395467370382e-14 | 2.638002633037305e-13 | monolithic |
| cuSolverSP | csrlsvqr_monolithic | case1354pegase | 4 | fp64 | ok | True | 1 | 19.82208957672119 | 19.82208957672119 | 5.4772084991319034e-14 | 1.0390290401122095e-24 | 1.4193849554536489e-13 | monolithic |
| AMGx | gmres_amg_block_jacobi_200 | case14 | 0 | fp32 | ok | True | 1 | 5.345697 | 0.17471099999999998 | 3.384358711232631e-07 | 1.5388616207311595e-08 | 8.10695173108375e-07 | setup_solve |
| AMGx | gmres_amg_block_jacobi_1000 | case14 | 0 | fp32 | ok | True | 1 | 5.370128666666666 | 0.17539866666666667 | 3.384358711232631e-07 | 1.5388616207311595e-08 | 8.10695173108375e-07 | setup_solve |
| AMGx | fgmres_amg_block_jacobi_200 | case14 | 0 | fp32 | ok | True | 1 | 5.262460666666667 | 0.15820966666666667 | 2.3245974309718978e-07 | 1.0569901347336864e-08 | 6.662144819764198e-07 | setup_solve |
| AMGx | fgmres_amg_block_jacobi_1000 | case14 | 0 | fp32 | ok | True | 1 | 5.333897333333334 | 0.16367333333333334 | 2.3245974309718978e-07 | 1.0569901347336864e-08 | 6.662144819764198e-07 | setup_solve |
| AMGx | gmres_amg_block_jacobi_200 | case14 | 1 | fp32 | ok | True | 1 | 5.314183666666667 | 0.17105066666666666 | 2.377752775429421e-07 | 1.3700510251055877e-11 | 3.428042664315223e-07 | setup_solve |
| AMGx | gmres_amg_block_jacobi_1000 | case14 | 1 | fp32 | ok | True | 1 | 5.323373999999999 | 0.171745 | 2.377752775429421e-07 | 1.3700510251055877e-11 | 3.428042664315223e-07 | setup_solve |
| AMGx | fgmres_amg_block_jacobi_200 | case14 | 1 | fp32 | ok | True | 1 | 5.0784650000000005 | 0.159392 | 3.0690228969262765e-07 | 1.7683579257927667e-11 | 3.417642515616804e-07 | setup_solve |
| AMGx | fgmres_amg_block_jacobi_1000 | case14 | 1 | fp32 | ok | True | 1 | 5.160148666666667 | 0.16143566666666667 | 3.0690228969262765e-07 | 1.7683579257927667e-11 | 3.417642515616804e-07 | setup_solve |
| AMGx | gmres_amg_block_jacobi_200 | case14 | 2 | fp32 | ok | True | 1 | 5.397246333333333 | 0.17070633333333332 | 2.8587595227442316e-07 | 3.8411583016179815e-17 | 2.850816356284312e-07 | setup_solve |
| AMGx | gmres_amg_block_jacobi_1000 | case14 | 2 | fp32 | ok | True | 1 | 5.317454666666666 | 0.17171566666666668 | 2.8587595227442316e-07 | 3.8411583016179815e-17 | 2.850816356284312e-07 | setup_solve |
| AMGx | fgmres_amg_block_jacobi_200 | case14 | 2 | fp32 | ok | True | 1 | 5.2667660000000005 | 0.15993 | 2.514916039316153e-07 | 3.3791546807051024e-17 | 4.0291713530639193e-07 | setup_solve |
| AMGx | fgmres_amg_block_jacobi_1000 | case14 | 2 | fp32 | ok | True | 1 | 5.243014666666667 | 0.1574286666666667 | 2.514916039316153e-07 | 3.3791546807051024e-17 | 4.0291713530639193e-07 | setup_solve |
| AMGx | gmres_amg_block_jacobi_200 | case14 | 0 | fp64 | ok | True | 1 | 5.296373666666667 | 0.1885666666666667 | 8.28974104288361e-16 | 3.7693298568954857e-17 | 7.185194201288151e-16 | setup_solve |
| AMGx | gmres_amg_block_jacobi_1000 | case14 | 0 | fp64 | ok | True | 1 | 5.391773333333334 | 0.19215333333333331 | 8.28974104288361e-16 | 3.7693298568954857e-17 | 7.185194201288151e-16 | setup_solve |
| AMGx | fgmres_amg_block_jacobi_200 | case14 | 0 | fp64 | ok | True | 1 | 5.188311 | 0.177606 | 4.460184847520778e-16 | 2.0280377669294283e-17 | 6.006976787127443e-16 | setup_solve |
| AMGx | fgmres_amg_block_jacobi_1000 | case14 | 0 | fp64 | ok | True | 1 | 5.201943333333333 | 0.17349533333333333 | 4.460184847520778e-16 | 2.0280377669294283e-17 | 6.006976787127443e-16 | setup_solve |
| AMGx | gmres_amg_block_jacobi_200 | case14 | 1 | fp64 | ok | True | 1 | 5.3706629999999995 | 0.19351600000000002 | 6.553587960927659e-16 | 3.776149478941106e-20 | 1.396244717948542e-15 | setup_solve |
| AMGx | gmres_amg_block_jacobi_1000 | case14 | 1 | fp64 | ok | True | 1 | 5.374838333333334 | 0.19159933333333334 | 6.553587960927659e-16 | 3.776149478941106e-20 | 1.396244717948542e-15 | setup_solve |
| AMGx | fgmres_amg_block_jacobi_200 | case14 | 1 | fp64 | ok | True | 1 | 5.204254 | 0.173812 | 4.0500993305453846e-16 | 2.3336499895751527e-20 | 1.2908498687920004e-15 | setup_solve |
| AMGx | fgmres_amg_block_jacobi_1000 | case14 | 1 | fp64 | ok | True | 1 | 5.2836533333333335 | 0.17531533333333335 | 4.0500993305453846e-16 | 2.3336499895751527e-20 | 1.2908498687920004e-15 | setup_solve |
| AMGx | gmres_amg_block_jacobi_200 | case14 | 2 | fp64 | ok | True | 1 | 5.345753666666666 | 0.18964566666666668 | 6.472178714672242e-16 | 8.696311390177298e-26 | 4.3188842265254326e-16 | setup_solve |
| AMGx | gmres_amg_block_jacobi_1000 | case14 | 2 | fp64 | ok | True | 1 | 5.364856333333333 | 0.19177633333333333 | 6.472178714672242e-16 | 8.696311390177298e-26 | 4.3188842265254326e-16 | setup_solve |
| AMGx | fgmres_amg_block_jacobi_200 | case14 | 2 | fp64 | ok | True | 1 | 5.275034333333333 | 0.17495133333333332 | 4.823400750448215e-16 | 6.480938913263108e-26 | 2.7928658108924964e-16 | setup_solve |
| AMGx | fgmres_amg_block_jacobi_1000 | case14 | 2 | fp64 | ok | True | 1 | 5.265042 | 0.172694 | 4.823400750448215e-16 | 6.480938913263108e-26 | 2.7928658108924964e-16 | setup_solve |
| Ginkgo-BiCGSTAB-Jacobi | bicgstab_jacobi_fp32tol | case14 | 0 | fp32 | ok | True | 0 | 14.047105333333334 | 0.14538933333333331 | 2.735311743973606e-07 | 1.2437411700969262e-08 | 7.154309880046067e-07 | setup_solve |
| Ginkgo-BiCGSTAB-Jacobi | bicgstab_jacobi_fp32tol | case14 | 1 | fp32 | ok | True | 0 | 13.760945333333334 | 0.14297133333333334 | 3.3855717223139987e-07 | 1.950752011166127e-11 | 3.0473356274023634e-07 | setup_solve |
| Ginkgo-BiCGSTAB-Jacobi | bicgstab_jacobi_fp32tol | case14 | 2 | fp32 | ok | True | 0 | 14.082976333333333 | 0.14626433333333336 | 2.809490745571356e-07 | 3.7749585492629847e-17 | 1.6164021434280198e-07 | setup_solve |
| Ginkgo-BiCGSTAB-Jacobi | bicgstab_jacobi_fp64tol | case14 | 0 | fp64 | ok | True | 0 | 14.135815 | 0.232526 | 4.451763619046844e-16 | 2.0242086499818265e-17 | 9.609636204163033e-16 | setup_solve |
| Ginkgo-BiCGSTAB-Jacobi | bicgstab_jacobi_fp64tol | case14 | 1 | fp64 | ok | True | 0 | 14.265153999999999 | 0.245607 | 6.969330395995259e-16 | 4.015698499861244e-20 | 5.042183861109951e-16 | setup_solve |
| Ginkgo-BiCGSTAB-Jacobi | bicgstab_jacobi_fp64tol | case14 | 2 | fp64 | ok | True | 0 | 14.357590666666667 | 0.23607266666666668 | 1.069762951138334e-15 | 1.4373817761991586e-25 | 7.930372755850407e-16 | setup_solve |
| Ginkgo-GMRES-Jacobi | gmres_jacobi_fp32tol | case14 | 0 | fp32 | ok | True | 1 | 14.352842333333333 | 0.15640333333333334 | 4.0984533698299196e-07 | 1.863559135813494e-08 | 5.337621392802797e-07 | setup_solve |
| Ginkgo-GMRES-Jacobi | gmres_jacobi_fp32tol | case14 | 1 | fp32 | ok | True | 1 | 13.631856666666668 | 0.15534766666666666 | 3.3723934418385345e-07 | 1.9431587420672074e-11 | 3.910981459249945e-07 | setup_solve |
| Ginkgo-GMRES-Jacobi | gmres_jacobi_fp32tol | case14 | 2 | fp32 | ok | True | 1 | 14.322124333333333 | 0.15183433333333332 | 2.723756397648902e-07 | 3.659762010471917e-17 | 1.6772388345409929e-07 | setup_solve |
| Ginkgo-GMRES-Jacobi | gmres_jacobi_fp64tol | case14 | 0 | fp64 | ok | True | 1 | 14.152940333333333 | 0.24104533333333333 | 6.003307889084992e-16 | 2.7296929481156512e-17 | 8.554499354421565e-16 | setup_solve |
| Ginkgo-GMRES-Jacobi | gmres_jacobi_fp64tol | case14 | 1 | fp64 | ok | True | 1 | 14.135981666666666 | 0.24094866666666667 | 8.98499063397869e-16 | 5.17711334662345e-20 | 4.754382646080073e-16 | setup_solve |
| Ginkgo-GMRES-Jacobi | gmres_jacobi_fp64tol | case14 | 2 | fp64 | ok | True | 1 | 14.537532333333333 | 0.24675633333333333 | 8.060660749549779e-16 | 1.08306675354552e-25 | 3.7761994519610724e-16 | setup_solve |
| STRUMPACK | np1_omp1_default_no_compression | case14 | 0 | fp32 | ok | True | 1 | 9.041345999999999 | 8.892640333333333 | 7.3786831255297e-08 | 3.3550735138470952e-09 | 4.027212980011486e-07 | analysis_factor_solve |
| STRUMPACK | np1_omp1_default_no_compression | case14 | 1 | fp32 | ok | True | 1 | 9.056684666666667 | 8.880317333333334 | 1.2332311885757698e-07 | 7.105825599531796e-12 | 2.0282601420709634e-07 | analysis_factor_solve |
| STRUMPACK | np1_omp1_default_no_compression | case14 | 2 | fp32 | ok | False | 1 | 8.986737333333332 | 8.842794333333332 | 1 | 1.3436451268663227e-10 | 1 | analysis_factor_solve |
| STRUMPACK | np1_omp1_default_no_compression | case14 | 0 | fp64 | ok | True | 1 | 9.156952333333331 | 8.976240333333331 | 2.002364995849879e-16 | 9.104716449181034e-18 | 8.479163958223845e-16 | analysis_factor_solve |
| STRUMPACK | np1_omp1_default_no_compression | case14 | 1 | fp64 | ok | True | 1 | 9.156251666666668 | 8.977116 | 1.7443042873244218e-16 | 1.0050607033833874e-20 | 7.796667613373599e-16 | analysis_factor_solve |
| STRUMPACK | np1_omp1_default_no_compression | case14 | 2 | fp64 | ok | True | 1 | 9.154378 | 8.972871 | 3.2126369043393675e-16 | 4.3166439209064995e-26 | 4.979764093448392e-16 | analysis_factor_solve |
| SuperLU_DIST | natural_largediag_one_shot_process | case14 | 0 | fp64 | ok | True | 1 | 152.364709 | 152.364709 | 6.203771105010279e-16 | 2.820843200106349e-17 | 7.645798013177956e-16 | monolithic |
| SuperLU_DIST | natural_largediag_one_shot_process | case14 | 1 | fp64 | ok | True | 1 | 152.788143 | 152.788143 | 5.723048128787997e-16 | 3.2975959639700364e-20 | 4.3733107413032237e-16 | monolithic |
| SuperLU_DIST | natural_largediag_one_shot_process | case14 | 2 | fp64 | ok | True | 1 | 156.710939 | 156.710939 | 7.993288057188042e-16 | 1.0740142545679489e-25 | 1.153131437060649e-15 | monolithic |
| audit_AMGx_fgmres_amg_jacobi_1000 | fgmres_amg_jacobi_1000 | case14 | 0 | fp32 | runtime_failed | False | -1 |  |  | None | None | None | setup_solve |
| audit_AMGx_fgmres_amg_jacobi_1000 | fgmres_amg_jacobi_1000 | case14 | 1 | fp32 | runtime_failed | False | -1 |  |  | None | None | None | setup_solve |
| audit_AMGx_fgmres_amg_jacobi_1000 | fgmres_amg_jacobi_1000 | case14 | 2 | fp32 | runtime_failed | False | -1 |  |  | None | None | None | setup_solve |
| audit_AMGx_fgmres_amg_jacobi_1000 | fgmres_amg_jacobi_1000 | case14 | 0 | fp64 | runtime_failed | False | -1 |  |  | None | None | None | setup_solve |
| audit_AMGx_fgmres_amg_jacobi_1000 | fgmres_amg_jacobi_1000 | case14 | 1 | fp64 | runtime_failed | False | -1 |  |  | None | None | None | setup_solve |
| audit_AMGx_fgmres_amg_jacobi_1000 | fgmres_amg_jacobi_1000 | case14 | 2 | fp64 | runtime_failed | False | -1 |  |  | None | None | None | setup_solve |
| audit_AMGx_fgmres_amg_jacobi_200 | fgmres_amg_jacobi_200 | case14 | 0 | fp32 | runtime_failed | False | -1 |  |  | None | None | None | setup_solve |
| audit_AMGx_fgmres_amg_jacobi_200 | fgmres_amg_jacobi_200 | case14 | 1 | fp32 | runtime_failed | False | -1 |  |  | None | None | None | setup_solve |
| audit_AMGx_fgmres_amg_jacobi_200 | fgmres_amg_jacobi_200 | case14 | 2 | fp32 | runtime_failed | False | -1 |  |  | None | None | None | setup_solve |
| audit_AMGx_fgmres_amg_jacobi_200 | fgmres_amg_jacobi_200 | case14 | 0 | fp64 | runtime_failed | False | -1 |  |  | None | None | None | setup_solve |
| audit_AMGx_fgmres_amg_jacobi_200 | fgmres_amg_jacobi_200 | case14 | 1 | fp64 | runtime_failed | False | -1 |  |  | None | None | None | setup_solve |
| audit_AMGx_fgmres_amg_jacobi_200 | fgmres_amg_jacobi_200 | case14 | 2 | fp64 | runtime_failed | False | -1 |  |  | None | None | None | setup_solve |
| audit_AMGx_gmres_amg_jacobi_1000 | gmres_amg_jacobi_1000 | case14 | 0 | fp32 | runtime_failed | False | -1 |  |  | None | None | None | setup_solve |
| audit_AMGx_gmres_amg_jacobi_1000 | gmres_amg_jacobi_1000 | case14 | 1 | fp32 | runtime_failed | False | -1 |  |  | None | None | None | setup_solve |
| audit_AMGx_gmres_amg_jacobi_1000 | gmres_amg_jacobi_1000 | case14 | 2 | fp32 | runtime_failed | False | -1 |  |  | None | None | None | setup_solve |
| audit_AMGx_gmres_amg_jacobi_1000 | gmres_amg_jacobi_1000 | case14 | 0 | fp64 | runtime_failed | False | -1 |  |  | None | None | None | setup_solve |
| audit_AMGx_gmres_amg_jacobi_1000 | gmres_amg_jacobi_1000 | case14 | 1 | fp64 | runtime_failed | False | -1 |  |  | None | None | None | setup_solve |
| audit_AMGx_gmres_amg_jacobi_1000 | gmres_amg_jacobi_1000 | case14 | 2 | fp64 | runtime_failed | False | -1 |  |  | None | None | None | setup_solve |
| audit_AMGx_gmres_amg_jacobi_200 | gmres_amg_jacobi_200 | case14 | 0 | fp32 | runtime_failed | False | -1 |  |  | None | None | None | setup_solve |
| audit_AMGx_gmres_amg_jacobi_200 | gmres_amg_jacobi_200 | case14 | 1 | fp32 | runtime_failed | False | -1 |  |  | None | None | None | setup_solve |
| audit_AMGx_gmres_amg_jacobi_200 | gmres_amg_jacobi_200 | case14 | 2 | fp32 | runtime_failed | False | -1 |  |  | None | None | None | setup_solve |
| audit_AMGx_gmres_amg_jacobi_200 | gmres_amg_jacobi_200 | case14 | 0 | fp64 | runtime_failed | False | -1 |  |  | None | None | None | setup_solve |
| audit_AMGx_gmres_amg_jacobi_200 | gmres_amg_jacobi_200 | case14 | 1 | fp64 | runtime_failed | False | -1 |  |  | None | None | None | setup_solve |

## 11. Best-Effort Summary

| solver | previous_status | audit_status | correctness_passed | best_effort_config_tested | best_config | valid_for_performance_comparison | valid_for_integration_comparison | remaining_issue |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cuDSS | ok | valid_as_run | 38/38 | fp64/fp32 direct LU; analysis reuse checked | cuDSS general CSR direct | yes | yes | Interpret tiny-rhs cases with scaled and absolute residuals. |
| cuSolverSP/RF | ok | valid_monolithic_qr | 38/38 | cuSolverSP csrlsvqr only; RF noted unavailable without supplied LU | cuSolverSP QR | qualified | qualified | Monolithic QR is not reusable-factorization evidence. |
| AMGx | fixed GMRES+AMG | limited_grid_tested | 40/80 | GMRES/FGMRES with AMG and Jacobi/BlockJacobi smoothers, max_iter 200/1000 | see config_audit.csv per case | qualified | qualified | Iterative convergence remains setup-sensitive on larger Jacobians. |
| Ginkgo | GMRES/BiCGSTAB Jacobi | valid_as_jacobi_only_incomplete_best_effort | 20/20 | GMRES/BiCGSTAB Jacobi with audit tolerances | BiCGSTAB+Jacobi on small/medium where it converges | limited | limited | Wrapper ignores advanced preconditioners; not full Ginkgo best effort. |
| STRUMPACK | np=1 runnable | valid_external_hybrid_np1 | 9/11 | np=1, OMP_NUM_THREADS=1, default no compression | STRUMPACK MPIDist np=1 exact/no-compression default | qualified | qualified_external_baseline | np>1 timeout/hang; no SLATE full GPU path. |
| SuperLU_DIST | runtime_failed Invalid ISPEC | fixed_invalid_ispec_but_wrapper_needs_rewrite | 19/19 | NATURAL/MMD variants; LargeDiag/NOROWPERM; one-shot process-level runs | NATURAL+LargeDiag for MATPOWER probes; NATURAL+NOROWPERM for synthetic sanity | diagnostic_only | qualified_external_baseline | Original in-process repeated ABglobal wrapper can mutate A; needs explicit reusable-state implementation. |

## 12. Updated Interpretation for Annual-Report Writing

The strongest cuPF default-solver evidence remains cuDSS, but the careful claim is not simply that it is fastest. The valid evidence is that cuDSS directly supports general nonsymmetric sparse Jacobians, keeps the solve path GPU-resident after initial load, exposes symbolic analysis and numeric factorization/solve phases, and maps naturally to repeated Newton solves where the sparsity pattern is stable.

cuSolverSP is useful as an NVIDIA monolithic QR comparison, especially on small cases, but it is not equivalent to a reusable refactorization path. AMGx and Ginkgo should not be rejected merely for being iterative; the issue observed here is robustness and setup sensitivity on these Jacobians under the tested configuration grids. STRUMPACK and SuperLU_DIST are credible external distributed direct-solver baselines, but their MPI setup, host/hybrid residency, and integration complexity make them less natural as a cuPF default. SuperLU_DIST specifically needs a rewritten wrapper before its timing can be treated as fair performance data.
