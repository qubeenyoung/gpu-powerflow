#ifndef EXP_20260519_SYMBOLIC_FACTORIZATION_H
#define EXP_20260519_SYMBOLIC_FACTORIZATION_H

#include "symbolic/etree.h"
#include "symbolic/septree.h"

#include <stddef.h>

/*
 * SymbolicFactorization: 기호 분석 단계의 최상위 자료구조 및 진입점.
 *
 * symbolic_factorization_analyze() 한 번 호출로 다음 파이프라인이 실행된다:
 *
 *   소거 트리 구성  →  분리자 트리 구성  →  프론트 트리 구성
 *   →  update set 계산  →  update-to-parent 맵  →  스케줄 계산
 *   →  저장 계획  →  엔트리 어셈블리 계획  →  기여 어셈블리 계획  →  검증
 *
 * 알고리즘 참고:
 *   STRUMPACK sparse/EliminationTree.hpp/.cpp and sparse/fronts/Front.hpp/.cpp:
 *   symbolic_factorization(), setup_tree(), front metadata, upd_to_parent().
 */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int front_id;
    int separator_id;
    int parent;
    int left_child;
    int right_child;
    int sep_begin;
    int sep_end;
    int num_pivots;
    int num_updates;
    int num_front_vars;
    int *pivot_vars;
    int *update_vars;
    int *front_vars;
    int *update_to_parent;
    int num_update_to_parent;
} FrontSymbolic;

typedef struct {
    int num_fronts;
    int num_levels;
    int *factor_order;
    int *forward_order;
    int *backward_order;
    int *level_ptr;
    int *level_fronts;
} FrontSchedule;

typedef struct {
    int num_entries;
    int *source_index;
    int *target_front;
    int *local_row;
    int *local_col;
    size_t *F_offset;
} EntryAssemblyPlan;

typedef struct {
    /*
     * STRUMPACK-style contribution assembly does not materialize nupd*nupd
     * entry maps.  Each child front already owns update_to_parent, which maps
     * child update variable index -> parent front local index. Numeric assembly
     * computes dense offsets from that map on the fly.
     */
    int num_child_fronts;
    int total_update_indices;
} ContributionAssemblyPlan;

typedef struct {
    int num_fronts;
    int *npiv;
    int *nupd;
    int *nfront;
    size_t *F_offset;
    size_t *L11_offset;
    size_t *U11_offset;
    size_t *L21_offset;
    size_t *U12_offset;
    size_t *C_offset;
    size_t total_dense_entries;
    size_t total_dense_bytes;
} FrontStoragePlan;

typedef struct {
    int n;
    SeparatorTree separator_tree;
    int num_fronts;
    FrontSymbolic *fronts;
    int *etree_parent;
    int *etree_postorder;
    /*
     * input_perm/input_inv_perm are the METIS permutation used by the caller
     * to construct A_perm = P A P^T.
     *
     * separator_perm/separator_inv_perm are the additional STRUMPACK-style
     * etree-postordered permutation inside the already permuted A_perm
     * coordinate system:
     *   separator_perm[final_position] = A_perm variable id
     *   separator_inv_perm[A_perm variable id] = final_position
     *
     * final_perm/final_inv_perm combine both levels and are expressed in the
     * original matrix coordinate system:
     *   final_perm[original variable id] = final separator position
     *   final_inv_perm[final separator position] = original variable id
     */
    int *input_perm;
    int *input_inv_perm;
    int *separator_perm;
    int *separator_inv_perm;
    int *final_perm;
    int *final_inv_perm;
    FrontSchedule schedule;
    EntryAssemblyPlan entry_assembly;
    ContributionAssemblyPlan contribution_assembly;
    FrontStoragePlan storage;
} SymbolicFactorization;

/*
 * Function: symbolic_build_front_schedule
 *
 * Purpose:
 *   Build traversal orders and level groups for symbolic/numeric front walks.
 *
 * Inputs:
 *   tree     - Separator tree whose nodes are symbolic fronts.
 *
 * Outputs:
 *   schedule - factor_order and forward_order are children-before-parent.
 *              backward_order is the reverse traversal. level_ptr/level_fronts
 *              group fronts by tree level, leaves first.
 *
 * Returns:
 *   SDS_OK on success, or a project error code.
 */
int symbolic_build_front_schedule(const SeparatorTree *tree,
                                  FrontSchedule *schedule);

/*
 * Function: symbolic_build_front_storage_plan
 *
 * Purpose:
 *   Compute dense block sizes and global dense-storage offsets for every front.
 *
 * Inputs:
 *   fronts     - Symbolic fronts.
 *   num_fronts - Number of fronts.
 *
 * Outputs:
 *   storage    - Per-front npiv/nupd/nfront and offsets for F, L11, U11, L21,
 *                U12, and C. Offsets are counted in double entries.
 *
 * Returns:
 *   SDS_OK on success, or a project error code.
 */
int symbolic_build_front_storage_plan(const FrontSymbolic *fronts,
                                      int num_fronts,
                                      FrontStoragePlan *storage);

/*
 * Function: symbolic_build_entry_assembly_plan
 *
 * Purpose:
 *   Assign each original A_perm nonzero to exactly one lowest symbolic front
 *   that contains both row and column variables.
 *
 * Inputs:
 *   a_perm_pattern - Permuted matrix in CSC format.
 *   symbolic       - Symbolic object with fronts and storage plan already built.
 *
 * Outputs:
 *   plan           - Source CSC value index and target dense F offset per entry.
 *
 * Returns:
 *   SDS_OK on success, or a project error code.
 */
int symbolic_build_entry_assembly_plan(const CSCMatrix *a_perm_pattern,
                                       const SymbolicFactorization *symbolic,
                                       EntryAssemblyPlan *plan);

/*
 * Function: symbolic_build_contribution_assembly_plan
 *
 * Purpose:
 *   Validate/count child contribution mappings without expanding dense
 *   contribution blocks entry-by-entry.
 *
 * Inputs:
 *   symbolic - Symbolic object with update_to_parent maps and storage plan.
 *
 * Outputs:
 *   plan     - Summary counters only. The actual mapping is each child
 *              FrontSymbolic::update_to_parent array.
 *
 * Returns:
 *   SDS_OK on success, or a project error code.
 */
int symbolic_build_contribution_assembly_plan(const SymbolicFactorization *symbolic,
                                              ContributionAssemblyPlan *plan);

void symbolic_front_schedule_destroy(FrontSchedule *schedule);
void symbolic_entry_assembly_plan_destroy(EntryAssemblyPlan *plan);
void symbolic_contribution_assembly_plan_destroy(ContributionAssemblyPlan *plan);
void symbolic_front_storage_plan_destroy(FrontStoragePlan *storage);

/*
 * Function: symbolic_build_update_sets
 *
 * Purpose:
 *   Build one sorted symbolic update set per separator/front.
 *
 * Inputs:
 *   a_perm_pattern - Permuted sparse matrix pattern in CSC format.
 *   tree           - Separator tree.
 *
 * Outputs:
 *   fronts         - Front array whose pivot_vars are already initialized.
 *                    update_vars and front_vars are allocated here.
 *
 * Returns:
 *   SDS_OK on success, or a project error code.
 *
 * Notes:
 *   Children are processed before parents because separator ids are created in
 *   postorder. This mirrors STRUMPACK's symbolic_factorization() data flow.
 */
int symbolic_build_update_sets(const CSCMatrix *a_perm_pattern,
                               const SeparatorTree *tree,
                               FrontSymbolic *fronts);

/*
 * Function: symbolic_build_front_tree
 *
 * Purpose:
 *   Create one FrontSymbolic object for each separator tree node.
 *
 * Inputs:
 *   tree   - Separator tree.
 *   perm   - Final separator-tree variable order.
 *   n      - Number of variables.
 *
 * Outputs:
 *   fronts - Caller-allocated array of tree->num_separators fronts.
 *
 * Returns:
 *   SDS_OK on success, or a project error code.
 */
int symbolic_build_front_tree(const SeparatorTree *tree,
                              const int *perm,
                              int n,
                              FrontSymbolic *fronts);

/*
 * Function: symbolic_build_update_to_parent_maps
 *
 * Purpose:
 *   Map each child update variable into its parent front_vars array.
 *
 * Inputs:
 *   tree   - Separator tree.
 *
 * Outputs:
 *   fronts - update_to_parent arrays are allocated for non-root fronts.
 *
 * Returns:
 *   SDS_OK on success.
 *   SDS_ERR_SYMBOLIC if a child update variable is not present in the parent.
 *
 * Notes:
 *   This is the C-style equivalent of STRUMPACK Front::upd_to_parent().
 */
int symbolic_build_update_to_parent_maps(const SeparatorTree *tree,
                                         FrontSymbolic *fronts);

/*
 * Function: symbolic_factorization_analyze
 *
 * Purpose:
 *   Run the complete symbolic path:
 *   etree -> postorder -> separator tree -> fronts -> update maps ->
 *   schedules -> assembly plans -> dense storage plan.
 *
 * Inputs:
 *   a_perm_pattern - METIS-permuted matrix pattern in CSC format.
 *   perm           - METIS permutation used to form A_perm.
 *   inv_perm       - Inverse METIS permutation.
 *
 * Outputs:
 *   symbolic       - Initialized symbolic object. Caller owns its arrays and
 *                    must call symbolic_factorization_destroy().
 *
 * Returns:
 *   SDS_OK on success, or a project error code.
 *
 * Notes:
 *   Since A_perm is already METIS ordered, separator_perm stores the final
 *   etree-postordered variable sequence in A_perm coordinates. final_perm and
 *   final_inv_perm additionally combine that ordering with the input METIS
 *   permutation and are expressed in original matrix coordinates.
 */
int symbolic_factorization_analyze(const CSCMatrix *a_perm_pattern,
                                   const int *perm,
                                   const int *inv_perm,
                                   SymbolicFactorization *symbolic);

/*
 * Function: symbolic_factorization_validate
 *
 * Purpose:
 *   Validate separator ownership, front metadata, update ordering, and
 *   update-to-parent maps, traversal schedules, assembly plans, and storage
 *   offsets.
 */
int symbolic_factorization_validate(const SymbolicFactorization *symbolic);

/*
 * Function: symbolic_factorization_print
 *
 * Purpose:
 *   Print etree, separator tree, fronts, update sets, update maps, schedules,
 *   assembly plans, and dense storage offsets.
 */
void symbolic_factorization_print(const SymbolicFactorization *symbolic);

/*
 * Function: symbolic_factorization_destroy
 *
 * Purpose:
 *   Free all arrays owned by a SymbolicFactorization.
 */
void symbolic_factorization_destroy(SymbolicFactorization *symbolic);

#ifdef __cplusplus
}
#endif

#endif
