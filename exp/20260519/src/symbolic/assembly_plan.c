#include "symbolic/assembly_plan.h"
#include "symbolic/symbolic_internal.h"

#include <stdlib.h>
#include <string.h>

static int find_front_containing_pair(const SymbolicFactorization *symbolic,
                                      const int *owner,
                                      int row,
                                      int col);

int symbolic_build_entry_assembly_plan(const CSCMatrix *a_perm_pattern,
                                       const SymbolicFactorization *symbolic,
                                       EntryAssemblyPlan *plan)
{
    int *owner = NULL;
    int write = 0;
    int rc;

    if (!plan || !symbolic ||
        symbolic_validate_csc_pattern(a_perm_pattern) != SDS_OK) {
        return SDS_ERR_BAD_INPUT;
    }
    memset(plan, 0, sizeof(*plan));

    owner = (int *)malloc((size_t)symbolic->n * sizeof(int));
    if (!owner) {
        return SDS_ERR_ALLOC;
    }
    rc = symbolic_build_front_owner_map(symbolic, owner);
    if (rc != SDS_OK) {
        free(owner);
        return rc;
    }

    plan->num_entries = a_perm_pattern->nnz;
    plan->source_index = (int *)malloc((size_t)plan->num_entries * sizeof(int));
    plan->target_front = (int *)malloc((size_t)plan->num_entries * sizeof(int));
    plan->local_row = (int *)malloc((size_t)plan->num_entries * sizeof(int));
    plan->local_col = (int *)malloc((size_t)plan->num_entries * sizeof(int));
    plan->F_offset = (size_t *)malloc((size_t)plan->num_entries * sizeof(size_t));
    if (!plan->source_index || !plan->target_front || !plan->local_row ||
        !plan->local_col || !plan->F_offset) {
        free(owner);
        symbolic_entry_assembly_plan_destroy(plan);
        return SDS_ERR_ALLOC;
    }

    for (int col = 0; col < a_perm_pattern->ncols; ++col) {
        for (int p = a_perm_pattern->colptr[col]; p < a_perm_pattern->colptr[col + 1]; ++p) {
            const int row = a_perm_pattern->rowind[p];
            const int front = find_front_containing_pair(symbolic, owner, row, col);
            int local_row;
            int local_col;
            int nfront;

            if (front < 0) {
                free(owner);
                return SDS_ERR_SYMBOLIC;
            }
            local_row = symbolic_find_front_var(&symbolic->fronts[front], row);
            local_col = symbolic_find_front_var(&symbolic->fronts[front], col);
            if (local_row < 0 || local_col < 0) {
                free(owner);
                return SDS_ERR_SYMBOLIC;
            }

            nfront = symbolic->storage.nfront[front];
            plan->source_index[write] = p;
            plan->target_front[write] = front;
            plan->local_row[write] = local_row;
            plan->local_col[write] = local_col;
            plan->F_offset[write] =
                symbolic->storage.F_offset[front] +
                (size_t)local_row + (size_t)local_col * (size_t)nfront;
            ++write;
        }
    }

    free(owner);
    return write == plan->num_entries ? SDS_OK : SDS_ERR_SYMBOLIC;
}

int symbolic_build_contribution_assembly_plan(const SymbolicFactorization *symbolic,
                                              ContributionAssemblyPlan *plan)
{
    if (!symbolic || !plan) {
        return SDS_ERR_BAD_INPUT;
    }
    memset(plan, 0, sizeof(*plan));

    for (int child = 0; child < symbolic->num_fronts; ++child) {
        const FrontSymbolic *front = &symbolic->fronts[child];
        const int nupd = symbolic->fronts[child].num_updates;
        if (front->parent == -1) {
            continue;
        }
        if (front->num_update_to_parent != nupd ||
            (nupd > 0 && !front->update_to_parent)) {
            return SDS_ERR_SYMBOLIC;
        }
        ++plan->num_child_fronts;
        plan->total_update_indices += nupd;
    }

    return SDS_OK;
}

void symbolic_entry_assembly_plan_destroy(EntryAssemblyPlan *plan)
{
    if (!plan) {
        return;
    }
    free(plan->source_index);
    free(plan->target_front);
    free(plan->local_row);
    free(plan->local_col);
    free(plan->F_offset);
    memset(plan, 0, sizeof(*plan));
}

void symbolic_contribution_assembly_plan_destroy(ContributionAssemblyPlan *plan)
{
    if (!plan) {
        return;
    }
    memset(plan, 0, sizeof(*plan));
}

/*
 * 비영값 (row, col)을 모두 포함하는 가장 낮은 프론트를 찾는다.
 *
 * 탐색 전략:
 *   1. inv_perm 기준으로 위치가 더 낮은 변수(= 더 먼저 소거되는 변수)의
 *      owner 프론트에서 시작한다. 그 프론트는 해당 변수를 pivot으로 갖고
 *      나머지 변수를 front_vars에 포함할 가능성이 가장 높다.
 *   2. 두 변수를 모두 포함하지 않으면 부모로 올라간다.
 *      separator 트리의 구조상 두 변수의 공통 조상 프론트가 반드시 존재한다.
 */
static int find_front_containing_pair(const SymbolicFactorization *symbolic,
                                      const int *owner,
                                      int row,
                                      int col)
{
    int front_id;

    if (!symbolic || !owner || row < 0 || row >= symbolic->n ||
        col < 0 || col >= symbolic->n) {
        return -1;
    }

    if (symbolic->separator_inv_perm[row] <= symbolic->separator_inv_perm[col]) {
        front_id = owner[row];
    } else {
        front_id = owner[col];
    }

    while (front_id != -1) {
        const FrontSymbolic *front = &symbolic->fronts[front_id];
        if (symbolic_find_front_var(front, row) >= 0 &&
            symbolic_find_front_var(front, col) >= 0) {
            return front_id;
        }
        front_id = front->parent;
    }

    return -1;
}
