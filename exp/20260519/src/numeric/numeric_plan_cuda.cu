#include "numeric/numeric_cuda_internal.h"

#include <stdlib.h>
#include <string.h>

int numeric_build_compact_block_layout(const SymbolicFactorization *symbolic,
                                       NumericFactorization *numeric)
{
    size_t cursor = 0;

    if (!symbolic || !numeric || symbolic->num_fronts <= 0) {
        return SDS_ERR_BAD_INPUT;
    }

    numeric->h_L11_offset = (size_t *)malloc((size_t)symbolic->num_fronts * sizeof(size_t));
    numeric->h_U11_offset = (size_t *)malloc((size_t)symbolic->num_fronts * sizeof(size_t));
    numeric->h_L21_offset = (size_t *)malloc((size_t)symbolic->num_fronts * sizeof(size_t));
    numeric->h_U12_offset = (size_t *)malloc((size_t)symbolic->num_fronts * sizeof(size_t));
    numeric->h_C_offset = (size_t *)malloc((size_t)symbolic->num_fronts * sizeof(size_t));
    if (!numeric->h_L11_offset || !numeric->h_U11_offset ||
        !numeric->h_L21_offset || !numeric->h_U12_offset || !numeric->h_C_offset) {
        return SDS_ERR_ALLOC;
    }

    for (int front = 0; front < symbolic->num_fronts; ++front) {
        const size_t npiv = (size_t)symbolic->fronts[front].num_pivots;
        const size_t nupd = (size_t)symbolic->fronts[front].num_updates;

        numeric->h_L11_offset[front] = cursor;
        cursor += npiv * npiv;
        numeric->h_U11_offset[front] = cursor;
        cursor += npiv * npiv;
        numeric->h_L21_offset[front] = cursor;
        cursor += nupd * npiv;
        numeric->h_U12_offset[front] = cursor;
        cursor += npiv * nupd;
        numeric->h_C_offset[front] = cursor;
        cursor += nupd * nupd;
    }

    numeric->total_dense_entries = cursor;
    numeric->total_dense_bytes = cursor * sizeof(double);
    return SDS_OK;
}

int numeric_build_host_plans(const SymbolicFactorization *symbolic,
                             const NumericFactorization *numeric,
                             HostNumericPlans *plans)
{
    const EntryAssemblyPlan *entry_plan = &symbolic->entry_assembly;
    int update_cursor = 0;

    memset(plans, 0, sizeof(*plans));
    plans->num_fronts = symbolic->num_fronts;

    plans->entry_counts = (int *)calloc((size_t)plans->num_fronts, sizeof(int));
    plans->entry_ptr = (int *)calloc((size_t)plans->num_fronts + 1u, sizeof(int));
    plans->entry_cursor = (int *)calloc((size_t)plans->num_fronts, sizeof(int));
    plans->entry_source = (int *)malloc((size_t)entry_plan->num_entries * sizeof(int));
    plans->entry_target_offset =
        (size_t *)malloc((size_t)entry_plan->num_entries * sizeof(size_t));
    plans->update_to_parent_ptr =
        (int *)calloc((size_t)plans->num_fronts + 1u, sizeof(int));
    if (!plans->entry_counts || !plans->entry_ptr || !plans->entry_cursor ||
        (entry_plan->num_entries > 0 &&
         (!plans->entry_source || !plans->entry_target_offset)) ||
        !plans->update_to_parent_ptr) {
        numeric_destroy_host_plans(plans);
        return SDS_ERR_ALLOC;
    }

    for (int i = 0; i < entry_plan->num_entries; ++i) {
        ++plans->entry_counts[entry_plan->target_front[i]];
    }
    for (int front = 0; front < plans->num_fronts; ++front) {
        plans->entry_ptr[front + 1] =
            plans->entry_ptr[front] + plans->entry_counts[front];
        plans->entry_cursor[front] = plans->entry_ptr[front];
    }
    for (int i = 0; i < entry_plan->num_entries; ++i) {
        const int front = entry_plan->target_front[i];
        const int dst = plans->entry_cursor[front]++;
        const int row = entry_plan->local_row[i];
        const int col = entry_plan->local_col[i];
        const int npiv = symbolic->storage.npiv[front];
        const int nupd = symbolic->storage.nupd[front];
        plans->entry_source[dst] = entry_plan->source_index[i];
        if (row < npiv && col < npiv) {
            plans->entry_target_offset[dst] =
                numeric->h_U11_offset[front] + (size_t)row + (size_t)col * (size_t)npiv;
        } else if (row < npiv) {
            plans->entry_target_offset[dst] =
                numeric->h_U12_offset[front] +
                (size_t)row + (size_t)(col - npiv) * (size_t)npiv;
        } else if (col < npiv) {
            plans->entry_target_offset[dst] =
                numeric->h_L21_offset[front] +
                (size_t)(row - npiv) + (size_t)col * (size_t)nupd;
        } else {
            plans->entry_target_offset[dst] =
                numeric->h_C_offset[front] +
                (size_t)(row - npiv) + (size_t)(col - npiv) * (size_t)nupd;
        }
    }

    for (int front = 0; front < plans->num_fronts; ++front) {
        plans->update_to_parent_ptr[front + 1] =
            plans->update_to_parent_ptr[front] +
            symbolic->fronts[front].num_update_to_parent;
    }
    if (plans->update_to_parent_ptr[plans->num_fronts] > 0) {
        plans->update_to_parent =
            (int *)malloc((size_t)plans->update_to_parent_ptr[plans->num_fronts] *
                          sizeof(int));
        if (!plans->update_to_parent) {
            numeric_destroy_host_plans(plans);
            return SDS_ERR_ALLOC;
        }
    }
    for (int front = 0; front < plans->num_fronts; ++front) {
        for (int i = 0; i < symbolic->fronts[front].num_update_to_parent; ++i) {
            plans->update_to_parent[update_cursor++] =
                symbolic->fronts[front].update_to_parent[i];
        }
    }

    return SDS_OK;
}

void numeric_destroy_host_plans(HostNumericPlans *plans)
{
    if (!plans) {
        return;
    }
    free(plans->entry_counts);
    free(plans->entry_ptr);
    free(plans->entry_cursor);
    free(plans->entry_source);
    free(plans->entry_target_offset);
    free(plans->update_to_parent_ptr);
    free(plans->update_to_parent);
    memset(plans, 0, sizeof(*plans));
}

int numeric_upload_plans(const HostNumericPlans *plans,
                         NumericFactorization *numeric)
{
    int rc;
    const int num_fronts = plans->num_fronts;
    const int num_entries = plans->entry_ptr[num_fronts];
    const int num_update_to_parent = plans->update_to_parent_ptr[num_fronts];

    numeric->h_entry_ptr = (int *)malloc(((size_t)num_fronts + 1u) * sizeof(int));
    numeric->h_update_to_parent_ptr =
        (int *)malloc(((size_t)num_fronts + 1u) * sizeof(int));
    if (!numeric->h_entry_ptr || !numeric->h_update_to_parent_ptr) {
        return SDS_ERR_ALLOC;
    }
    memcpy(numeric->h_entry_ptr, plans->entry_ptr,
           ((size_t)num_fronts + 1u) * sizeof(int));
    memcpy(numeric->h_update_to_parent_ptr, plans->update_to_parent_ptr,
           ((size_t)num_fronts + 1u) * sizeof(int));

    rc = numeric_cuda_copy_to_device((void **)&numeric->d_entry_ptr,
                                     plans->entry_ptr,
                                     (size_t)num_fronts + 1u,
                                     sizeof(int), numeric);
    if (rc == SDS_OK) {
        rc = numeric_cuda_copy_to_device((void **)&numeric->d_entry_source_index,
                                         plans->entry_source,
                                         (size_t)num_entries,
                                         sizeof(int), numeric);
    }
    if (rc == SDS_OK) {
        rc = numeric_cuda_copy_to_device((void **)&numeric->d_entry_target_offset,
                                         plans->entry_target_offset,
                                         (size_t)num_entries,
                                         sizeof(size_t), numeric);
    }
    if (rc == SDS_OK) {
        rc = numeric_cuda_copy_to_device((void **)&numeric->d_update_to_parent,
                                         plans->update_to_parent,
                                         (size_t)num_update_to_parent,
                                         sizeof(int), numeric);
    }
    return rc;
}
