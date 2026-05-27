#include "symbolic/storage_plan.h"

#include <stdlib.h>
#include <string.h>

int symbolic_build_front_storage_plan(const FrontSymbolic *fronts,
                                      int num_fronts,
                                      FrontStoragePlan *storage)
{
    size_t cursor = 0;

    if (!fronts || num_fronts <= 0 || !storage) {
        return SDS_ERR_BAD_INPUT;
    }
    memset(storage, 0, sizeof(*storage));
    storage->num_fronts = num_fronts;

    storage->npiv = (int *)malloc((size_t)num_fronts * sizeof(int));
    storage->nupd = (int *)malloc((size_t)num_fronts * sizeof(int));
    storage->nfront = (int *)malloc((size_t)num_fronts * sizeof(int));
    storage->F_offset = (size_t *)malloc((size_t)num_fronts * sizeof(size_t));
    storage->L11_offset = (size_t *)malloc((size_t)num_fronts * sizeof(size_t));
    storage->U11_offset = (size_t *)malloc((size_t)num_fronts * sizeof(size_t));
    storage->L21_offset = (size_t *)malloc((size_t)num_fronts * sizeof(size_t));
    storage->U12_offset = (size_t *)malloc((size_t)num_fronts * sizeof(size_t));
    storage->C_offset = (size_t *)malloc((size_t)num_fronts * sizeof(size_t));
    if (!storage->npiv || !storage->nupd || !storage->nfront ||
        !storage->F_offset || !storage->L11_offset || !storage->U11_offset ||
        !storage->L21_offset || !storage->U12_offset || !storage->C_offset) {
        symbolic_front_storage_plan_destroy(storage);
        return SDS_ERR_ALLOC;
    }

    /* 프론트 0부터 순서대로 블록을 연속 배치한다.
     * 각 프론트 내부 순서: F → L11 → U11 → L21 → U12 → C */
    for (int front = 0; front < num_fronts; ++front) {
        const size_t npiv = (size_t)fronts[front].num_pivots;
        const size_t nupd = (size_t)fronts[front].num_updates;
        const size_t nfront = npiv + nupd;

        storage->npiv[front] = fronts[front].num_pivots;
        storage->nupd[front] = fronts[front].num_updates;
        storage->nfront[front] = fronts[front].num_front_vars;

        storage->F_offset[front] = cursor;
        cursor += nfront * nfront;
        storage->L11_offset[front] = cursor;
        cursor += npiv * npiv;
        storage->U11_offset[front] = cursor;
        cursor += npiv * npiv;
        storage->L21_offset[front] = cursor;
        cursor += nupd * npiv;
        storage->U12_offset[front] = cursor;
        cursor += npiv * nupd;
        storage->C_offset[front] = cursor;
        cursor += nupd * nupd;
    }

    storage->total_dense_entries = cursor;
    storage->total_dense_bytes = cursor * sizeof(double);
    return SDS_OK;
}

void symbolic_front_storage_plan_destroy(FrontStoragePlan *storage)
{
    if (!storage) {
        return;
    }
    free(storage->npiv);
    free(storage->nupd);
    free(storage->nfront);
    free(storage->F_offset);
    free(storage->L11_offset);
    free(storage->U11_offset);
    free(storage->L21_offset);
    free(storage->U12_offset);
    free(storage->C_offset);
    memset(storage, 0, sizeof(*storage));
}
