#include "symbolic/assembly_plan.h"
#include "symbolic/front_schedule.h"
#include "symbolic/front_symbolic.h"
#include "symbolic/storage_plan.h"
#include "symbolic/symbolic_factorization.h"
#include "symbolic/symbolic_internal.h"
#include "symbolic/symbolic_validate.h"

#include <stdlib.h>
#include <string.h>

/*
 * Algorithmic reference:
 *   third_party/lin_sol/strumpack/src/sparse/EliminationTree.hpp
 *   third_party/lin_sol/strumpack/src/sparse/EliminationTree.cpp
 *   third_party/lin_sol/strumpack/src/sparse/fronts/Front.hpp
 *   third_party/lin_sol/strumpack/src/sparse/fronts/Front.cpp
 *
 * This file now owns only the high-level symbolic orchestration:
 *   etree -> separator tree -> fronts -> schedules/plans -> validation.
 * Detailed front, schedule, storage, assembly, print, and validation logic
 * lives in focused files under src/symbolic/.
 */

int symbolic_factorization_analyze(const CSCMatrix *a_perm_pattern,
                                   const int *perm,
                                   const int *inv_perm,
                                   SymbolicFactorization *symbolic)
{
    int rc;
    const int n = a_perm_pattern ? a_perm_pattern->ncols : 0;

    if (!symbolic || symbolic_validate_csc_pattern(a_perm_pattern) != SDS_OK ||
        symbolic_validate_permutation_array(perm, n) != SDS_OK ||
        symbolic_validate_permutation_array(inv_perm, n) != SDS_OK) {
        return SDS_ERR_BAD_INPUT;
    }
    memset(symbolic, 0, sizeof(*symbolic));
    symbolic->n = n;

    symbolic->etree_parent = (int *)malloc((size_t)n * sizeof(int));
    symbolic->etree_postorder = (int *)malloc((size_t)n * sizeof(int));
    symbolic->input_perm = (int *)malloc((size_t)n * sizeof(int));
    symbolic->input_inv_perm = (int *)malloc((size_t)n * sizeof(int));
    symbolic->separator_perm = (int *)malloc((size_t)n * sizeof(int));
    symbolic->separator_inv_perm = (int *)malloc((size_t)n * sizeof(int));
    symbolic->final_perm = (int *)malloc((size_t)n * sizeof(int));
    symbolic->final_inv_perm = (int *)malloc((size_t)n * sizeof(int));
    if (!symbolic->etree_parent || !symbolic->etree_postorder ||
        !symbolic->input_perm || !symbolic->input_inv_perm ||
        !symbolic->separator_perm || !symbolic->separator_inv_perm ||
        !symbolic->final_perm || !symbolic->final_inv_perm) {
        symbolic_factorization_destroy(symbolic);
        return SDS_ERR_ALLOC;
    }
    memcpy(symbolic->input_perm, perm, (size_t)n * sizeof(int));
    memcpy(symbolic->input_inv_perm, inv_perm, (size_t)n * sizeof(int));

    rc = symbolic_build_elimination_tree(a_perm_pattern, symbolic->etree_parent);
    if (rc == SDS_OK) {
        rc = symbolic_etree_postorder(n, symbolic->etree_parent,
                                      symbolic->etree_postorder);
    }
    if (rc == SDS_OK) {
        rc = separator_tree_build_from_etree_and_perm(
            n,
            symbolic->etree_parent,
            symbolic->etree_postorder,
            symbolic->input_perm,
            symbolic->input_inv_perm,
            symbolic->separator_perm,
            symbolic->separator_inv_perm,
            symbolic->final_perm,
            symbolic->final_inv_perm,
            &symbolic->separator_tree);
    }
    if (rc != SDS_OK) {
        symbolic_factorization_destroy(symbolic);
        return rc;
    }

    symbolic->num_fronts = symbolic->separator_tree.num_separators;
    symbolic->fronts =
        (FrontSymbolic *)calloc((size_t)symbolic->num_fronts, sizeof(FrontSymbolic));
    if (!symbolic->fronts) {
        symbolic_factorization_destroy(symbolic);
        return SDS_ERR_ALLOC;
    }

    rc = symbolic_build_front_tree(&symbolic->separator_tree, symbolic->separator_perm,
                                   n, symbolic->fronts);
    if (rc == SDS_OK) {
        rc = symbolic_build_update_sets(a_perm_pattern, &symbolic->separator_tree,
                                        symbolic->fronts);
    }
    if (rc == SDS_OK) {
        rc = symbolic_build_update_to_parent_maps(&symbolic->separator_tree,
                                                  symbolic->fronts);
    }
    if (rc == SDS_OK) {
        rc = symbolic_build_front_schedule(&symbolic->separator_tree,
                                           &symbolic->schedule);
    }
    if (rc == SDS_OK) {
        rc = symbolic_build_front_storage_plan(symbolic->fronts,
                                               symbolic->num_fronts,
                                               &symbolic->storage);
    }
    if (rc == SDS_OK) {
        rc = symbolic_build_entry_assembly_plan(a_perm_pattern, symbolic,
                                                &symbolic->entry_assembly);
    }
    if (rc == SDS_OK) {
        rc = symbolic_build_contribution_assembly_plan(symbolic,
                                                       &symbolic->contribution_assembly);
    }
    if (rc == SDS_OK) {
        rc = symbolic_factorization_validate(symbolic);
    }
    if (rc != SDS_OK) {
        symbolic_factorization_destroy(symbolic);
    }

    return rc;
}

void symbolic_factorization_destroy(SymbolicFactorization *symbolic)
{
    if (!symbolic) {
        return;
    }
    symbolic_fronts_destroy(symbolic->fronts, symbolic->num_fronts);
    free(symbolic->fronts);
    separator_tree_destroy(&symbolic->separator_tree);
    free(symbolic->etree_parent);
    free(symbolic->etree_postorder);
    free(symbolic->input_perm);
    free(symbolic->input_inv_perm);
    free(symbolic->separator_perm);
    free(symbolic->separator_inv_perm);
    free(symbolic->final_perm);
    free(symbolic->final_inv_perm);
    symbolic_front_schedule_destroy(&symbolic->schedule);
    symbolic_entry_assembly_plan_destroy(&symbolic->entry_assembly);
    symbolic_contribution_assembly_plan_destroy(&symbolic->contribution_assembly);
    symbolic_front_storage_plan_destroy(&symbolic->storage);
    memset(symbolic, 0, sizeof(*symbolic));
}
