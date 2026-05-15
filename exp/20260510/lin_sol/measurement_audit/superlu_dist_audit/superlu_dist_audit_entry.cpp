#include <superlu_ddefs.h>

#ifndef SUPERLU_AUDIT_COLPERM
#define SUPERLU_AUDIT_COLPERM MMD_AT_PLUS_A
#endif

#ifndef SUPERLU_AUDIT_ROWPERM
#define SUPERLU_AUDIT_ROWPERM LargeDiag_MC64
#endif

// The original second-pass wrapper used METIS_AT_PLUS_A.  The installed
// SuperLU_DIST build has HAVE_PARMETIS disabled, so get_perm_c rejects that
// enum at runtime.  Include the original wrapper after redefining only that
// token so each audit executable tests one supported ColPerm variant.
#define METIS_AT_PLUS_A SUPERLU_AUDIT_COLPERM
#define LargeDiag_MC64 SUPERLU_AUDIT_ROWPERM
#include "../../solvers/superlu_dist/superlu_dist_benchmark.cpp"
