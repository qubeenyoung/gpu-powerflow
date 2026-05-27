#include "symbolic/symbolic_validate.h"
#include "symbolic/symbolic_internal.h"

#include <stdlib.h>
#include <string.h>

static int validate_front_schedule(const SymbolicFactorization *symbolic);
static int validate_front_storage_plan(const SymbolicFactorization *symbolic);
static int validate_entry_assembly_plan(const SymbolicFactorization *symbolic);
static int validate_contribution_assembly_plan(const SymbolicFactorization *symbolic);
static int validate_symbolic_permutations(const SymbolicFactorization *symbolic);
static int validate_etree_arrays(const SymbolicFactorization *symbolic);

int symbolic_factorization_validate(const SymbolicFactorization *symbolic)
{
    int *seen = NULL;
    int rc;

    if (!symbolic || symbolic->n <= 0 || !symbolic->fronts ||
        !symbolic->etree_parent || !symbolic->etree_postorder ||
        !symbolic->input_perm || !symbolic->input_inv_perm ||
        !symbolic->separator_perm || !symbolic->separator_inv_perm ||
        !symbolic->final_perm || !symbolic->final_inv_perm) {
        return SDS_ERR_BAD_INPUT;
    }

    rc = validate_symbolic_permutations(symbolic);
    if (rc != SDS_OK) {
        return rc;
    }
    rc = validate_etree_arrays(symbolic);
    if (rc != SDS_OK) {
        return rc;
    }

    rc = separator_tree_validate(&symbolic->separator_tree, symbolic->n);
    if (rc != SDS_OK) {
        return rc;
    }

    seen = (int *)calloc((size_t)symbolic->n, sizeof(int));
    if (!seen) {
        return SDS_ERR_ALLOC;
    }

    for (int front_id = 0; front_id < symbolic->num_fronts; ++front_id) {
        const FrontSymbolic *front = &symbolic->fronts[front_id];
        if (front->front_id != front_id || front->separator_id != front_id ||
            front->num_front_vars != front->num_pivots + front->num_updates) {
            free(seen);
            return SDS_ERR_SYMBOLIC;
        }

        for (int i = 0; i < front->num_pivots; ++i) {
            const int var = front->pivot_vars[i];
            if (var < 0 || var >= symbolic->n || seen[var]) {
                free(seen);
                return SDS_ERR_SYMBOLIC;
            }
            seen[var] = 1;
        }

        for (int i = 1; i < front->num_updates; ++i) {
            if (symbolic->separator_inv_perm[front->update_vars[i - 1]] >=
                symbolic->separator_inv_perm[front->update_vars[i]]) {
                free(seen);
                return SDS_ERR_SYMBOLIC;
            }
        }

        if (front->parent != -1) {
            if (front->num_update_to_parent != front->num_updates ||
                (front->num_updates > 0 && !front->update_to_parent)) {
                free(seen);
                return SDS_ERR_SYMBOLIC;
            }
            for (int i = 0; i < front->num_updates; ++i) {
                const int local = front->update_to_parent[i];
                if (local < 0 ||
                    local >= symbolic->fronts[front->parent].num_front_vars ||
                    symbolic->fronts[front->parent].front_vars[local] != front->update_vars[i]) {
                    free(seen);
                    return SDS_ERR_SYMBOLIC;
                }
            }
        }
    }

    for (int var = 0; var < symbolic->n; ++var) {
        if (!seen[var]) {
            free(seen);
            return SDS_ERR_SYMBOLIC;
        }
    }

    free(seen);

    rc = validate_front_schedule(symbolic);
    if (rc != SDS_OK) {
        return rc;
    }
    rc = validate_front_storage_plan(symbolic);
    if (rc != SDS_OK) {
        return rc;
    }
    rc = validate_entry_assembly_plan(symbolic);
    if (rc != SDS_OK) {
        return rc;
    }
    return validate_contribution_assembly_plan(symbolic);
}

static int validate_symbolic_permutations(const SymbolicFactorization *symbolic)
{
    const int n = symbolic ? symbolic->n : 0;

    if (!symbolic || n <= 0 ||
        symbolic_validate_permutation_array(symbolic->input_perm, n) != SDS_OK ||
        symbolic_validate_permutation_array(symbolic->input_inv_perm, n) != SDS_OK ||
        symbolic_validate_permutation_array(symbolic->separator_perm, n) != SDS_OK ||
        symbolic_validate_permutation_array(symbolic->separator_inv_perm, n) != SDS_OK ||
        symbolic_validate_permutation_array(symbolic->final_perm, n) != SDS_OK ||
        symbolic_validate_permutation_array(symbolic->final_inv_perm, n) != SDS_OK) {
        return SDS_ERR_SYMBOLIC;
    }

    for (int i = 0; i < n; ++i) {
        if (symbolic->input_inv_perm[symbolic->input_perm[i]] != i ||
            symbolic->separator_inv_perm[symbolic->separator_perm[i]] != i ||
            symbolic->final_inv_perm[symbolic->final_perm[i]] != i) {
            return SDS_ERR_SYMBOLIC;
        }
    }

    for (int original = 0; original < n; ++original) {
        const int metis_var = symbolic->input_perm[original];
        if (symbolic->final_perm[original] !=
            symbolic->separator_inv_perm[metis_var]) {
            return SDS_ERR_SYMBOLIC;
        }
    }
    for (int final_pos = 0; final_pos < n; ++final_pos) {
        const int metis_var = symbolic->separator_perm[final_pos];
        if (symbolic->final_inv_perm[final_pos] !=
            symbolic->input_inv_perm[metis_var]) {
            return SDS_ERR_SYMBOLIC;
        }
    }

    return SDS_OK;
}

static int validate_etree_arrays(const SymbolicFactorization *symbolic)
{
    int *seen = NULL;
    const int n = symbolic ? symbolic->n : 0;

    if (!symbolic || n <= 0 || !symbolic->etree_parent ||
        !symbolic->etree_postorder) {
        return SDS_ERR_BAD_INPUT;
    }
    seen = (int *)calloc((size_t)n, sizeof(int));
    if (!seen) {
        return SDS_ERR_ALLOC;
    }
    for (int node = 0; node < n; ++node) {
        const int parent = symbolic->etree_parent[node];
        if (parent < -1 || parent >= n || parent == node) {
            free(seen);
            return SDS_ERR_SYMBOLIC;
        }
    }
    for (int pos = 0; pos < n; ++pos) {
        const int node = symbolic->etree_postorder[pos];
        if (node < 0 || node >= n || seen[node]) {
            free(seen);
            return SDS_ERR_SYMBOLIC;
        }
        seen[node] = 1;
    }
    free(seen);
    return SDS_OK;
}

static int validate_front_schedule(const SymbolicFactorization *symbolic)
{
    const FrontSchedule *schedule;
    int *seen = NULL;
    int *position = NULL;
    int *front_level = NULL;
    int rc = SDS_OK;

    if (!symbolic) {
        return SDS_ERR_BAD_INPUT;
    }
    schedule = &symbolic->schedule;
    if (schedule->num_fronts != symbolic->num_fronts ||
        schedule->num_levels <= 0 || !schedule->factor_order ||
        !schedule->forward_order || !schedule->backward_order ||
        !schedule->level_ptr || !schedule->level_fronts) {
        return SDS_ERR_SYMBOLIC;
    }

    seen = (int *)calloc((size_t)schedule->num_fronts, sizeof(int));
    position = (int *)malloc((size_t)schedule->num_fronts * sizeof(int));
    front_level = (int *)malloc((size_t)schedule->num_fronts * sizeof(int));
    if (!seen || !position || !front_level) {
        free(seen);
        free(position);
        free(front_level);
        return SDS_ERR_ALLOC;
    }

    for (int i = 0; i < schedule->num_fronts; ++i) {
        const int front = schedule->factor_order[i];
        if (front < 0 || front >= schedule->num_fronts || seen[front]) {
            rc = SDS_ERR_SYMBOLIC;
            goto done;
        }
        seen[front] = 1;
        position[front] = i;
    }

    for (int i = 0; i < schedule->num_fronts; ++i) {
        if (schedule->forward_order[i] != schedule->factor_order[i] ||
            schedule->backward_order[i] !=
                schedule->factor_order[schedule->num_fronts - 1 - i]) {
            rc = SDS_ERR_SYMBOLIC;
            goto done;
        }
    }

    if (schedule->level_ptr[0] != 0 ||
        schedule->level_ptr[schedule->num_levels] != schedule->num_fronts) {
        rc = SDS_ERR_SYMBOLIC;
        goto done;
    }
    memset(seen, 0, (size_t)schedule->num_fronts * sizeof(int));
    for (int level = 0; level < schedule->num_levels; ++level) {
        if (schedule->level_ptr[level] > schedule->level_ptr[level + 1]) {
            rc = SDS_ERR_SYMBOLIC;
            goto done;
        }
        for (int p = schedule->level_ptr[level]; p < schedule->level_ptr[level + 1]; ++p) {
            const int front = schedule->level_fronts[p];
            if (front < 0 || front >= schedule->num_fronts || seen[front]) {
                rc = SDS_ERR_SYMBOLIC;
                goto done;
            }
            seen[front] = 1;
            front_level[front] = level;
        }
    }

    for (int front = 0; front < symbolic->num_fronts; ++front) {
        const FrontSymbolic *node = &symbolic->fronts[front];
        if (node->left_child != -1) {
            if (position[node->left_child] >= position[front] ||
                position[node->right_child] >= position[front] ||
                front_level[node->left_child] >= front_level[front] ||
                front_level[node->right_child] >= front_level[front]) {
                rc = SDS_ERR_SYMBOLIC;
                goto done;
            }
        }
    }

done:
    free(seen);
    free(position);
    free(front_level);
    return rc;
}

static int validate_front_storage_plan(const SymbolicFactorization *symbolic)
{
    const FrontStoragePlan *storage;
    size_t cursor = 0;

    if (!symbolic) {
        return SDS_ERR_BAD_INPUT;
    }
    storage = &symbolic->storage;
    if (storage->num_fronts != symbolic->num_fronts || !storage->npiv ||
        !storage->nupd || !storage->nfront || !storage->F_offset ||
        !storage->L11_offset || !storage->U11_offset ||
        !storage->L21_offset || !storage->U12_offset || !storage->C_offset) {
        return SDS_ERR_SYMBOLIC;
    }

    for (int front = 0; front < symbolic->num_fronts; ++front) {
        const size_t npiv = (size_t)symbolic->fronts[front].num_pivots;
        const size_t nupd = (size_t)symbolic->fronts[front].num_updates;
        const size_t nfront = npiv + nupd;

        if (storage->npiv[front] != symbolic->fronts[front].num_pivots ||
            storage->nupd[front] != symbolic->fronts[front].num_updates ||
            storage->nfront[front] != symbolic->fronts[front].num_front_vars) {
            return SDS_ERR_SYMBOLIC;
        }

        if (storage->F_offset[front] != cursor) {
            return SDS_ERR_SYMBOLIC;
        }
        cursor += symbolic_square_size((int)nfront);
        if (storage->L11_offset[front] != cursor) {
            return SDS_ERR_SYMBOLIC;
        }
        cursor += symbolic_square_size((int)npiv);
        if (storage->U11_offset[front] != cursor) {
            return SDS_ERR_SYMBOLIC;
        }
        cursor += symbolic_square_size((int)npiv);
        if (storage->L21_offset[front] != cursor) {
            return SDS_ERR_SYMBOLIC;
        }
        cursor += nupd * npiv;
        if (storage->U12_offset[front] != cursor) {
            return SDS_ERR_SYMBOLIC;
        }
        cursor += npiv * nupd;
        if (storage->C_offset[front] != cursor) {
            return SDS_ERR_SYMBOLIC;
        }
        cursor += symbolic_square_size((int)nupd);
    }

    if (storage->total_dense_entries != cursor ||
        storage->total_dense_bytes != cursor * sizeof(double)) {
        return SDS_ERR_SYMBOLIC;
    }
    return SDS_OK;
}

static int validate_entry_assembly_plan(const SymbolicFactorization *symbolic)
{
    const EntryAssemblyPlan *plan;
    int *seen = NULL;
    int rc = SDS_OK;

    if (!symbolic) {
        return SDS_ERR_BAD_INPUT;
    }
    plan = &symbolic->entry_assembly;
    if (plan->num_entries < 0) {
        return SDS_ERR_SYMBOLIC;
    }
    if (plan->num_entries == 0) {
        return SDS_OK;
    }
    if (!plan->source_index || !plan->target_front || !plan->local_row ||
        !plan->local_col || !plan->F_offset) {
        return SDS_ERR_SYMBOLIC;
    }

    seen = (int *)calloc((size_t)plan->num_entries, sizeof(int));
    if (!seen) {
        return SDS_ERR_ALLOC;
    }

    for (int i = 0; i < plan->num_entries; ++i) {
        const int source = plan->source_index[i];
        const int front = plan->target_front[i];
        int nfront;
        size_t expected;

        if (source < 0 || source >= plan->num_entries || seen[source] ||
            front < 0 || front >= symbolic->num_fronts) {
            rc = SDS_ERR_SYMBOLIC;
            goto done;
        }
        seen[source] = 1;

        nfront = symbolic->storage.nfront[front];
        if (plan->local_row[i] < 0 || plan->local_row[i] >= nfront ||
            plan->local_col[i] < 0 || plan->local_col[i] >= nfront) {
            rc = SDS_ERR_SYMBOLIC;
            goto done;
        }
        expected = symbolic->storage.F_offset[front] +
                   (size_t)plan->local_row[i] +
                   (size_t)plan->local_col[i] * (size_t)nfront;
        if (plan->F_offset[i] != expected ||
            plan->F_offset[i] >= symbolic->storage.total_dense_entries) {
            rc = SDS_ERR_SYMBOLIC;
            goto done;
        }
    }

done:
    free(seen);
    return rc;
}

static int validate_contribution_assembly_plan(const SymbolicFactorization *symbolic)
{
    const ContributionAssemblyPlan *plan;
    int expected_child_fronts = 0;
    int expected_update_indices = 0;

    if (!symbolic) {
        return SDS_ERR_BAD_INPUT;
    }
    plan = &symbolic->contribution_assembly;

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
        ++expected_child_fronts;
        expected_update_indices += nupd;
    }

    return plan->num_child_fronts == expected_child_fronts &&
           plan->total_update_indices == expected_update_indices
        ? SDS_OK : SDS_ERR_SYMBOLIC;
}
