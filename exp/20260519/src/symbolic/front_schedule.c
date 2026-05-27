#include "symbolic/front_schedule.h"

#include <stdlib.h>
#include <string.h>

static int emit_front_postorder(const SeparatorTree *tree,
                                int front_id,
                                int *order,
                                int *count);
static int compute_front_level(const SeparatorTree *tree, int front_id, int *levels);

int symbolic_build_front_schedule(const SeparatorTree *tree,
                                  FrontSchedule *schedule)
{
    int *levels = NULL;
    int *next = NULL;
    int count = 0;
    int max_level = 0;

    if (!tree || !schedule || tree->num_separators <= 0 ||
        separator_tree_validate(tree, tree->sizes[tree->num_separators]) != SDS_OK) {
        return SDS_ERR_BAD_INPUT;
    }
    memset(schedule, 0, sizeof(*schedule));
    schedule->num_fronts = tree->num_separators;

    schedule->factor_order = (int *)malloc((size_t)schedule->num_fronts * sizeof(int));
    schedule->forward_order = (int *)malloc((size_t)schedule->num_fronts * sizeof(int));
    schedule->backward_order = (int *)malloc((size_t)schedule->num_fronts * sizeof(int));
    schedule->level_fronts = (int *)malloc((size_t)schedule->num_fronts * sizeof(int));
    levels = (int *)malloc((size_t)schedule->num_fronts * sizeof(int));
    if (!schedule->factor_order || !schedule->forward_order ||
        !schedule->backward_order || !schedule->level_fronts || !levels) {
        free(levels);
        symbolic_front_schedule_destroy(schedule);
        return SDS_ERR_ALLOC;
    }

    for (int i = 0; i < schedule->num_fronts; ++i) {
        levels[i] = -1;
    }
    if (emit_front_postorder(tree, tree->root, schedule->factor_order, &count) != SDS_OK ||
        count != schedule->num_fronts) {
        free(levels);
        symbolic_front_schedule_destroy(schedule);
        return SDS_ERR_SYMBOLIC;
    }
    for (int i = 0; i < schedule->num_fronts; ++i) {
        schedule->forward_order[i] = schedule->factor_order[i];
        schedule->backward_order[i] = schedule->factor_order[schedule->num_fronts - 1 - i];
    }

    /* compute_front_level는 메모이제이션으로 중복 계산을 피한다.
     * 자식이 먼저 처리되므로 factor_order 순서대로 호출하면
     * 모든 자식 레벨이 이미 채워진 상태에서 부모를 처리하게 된다. */
    for (int front = 0; front < schedule->num_fronts; ++front) {
        const int level = compute_front_level(tree, front, levels);
        if (level < 0) {
            free(levels);
            symbolic_front_schedule_destroy(schedule);
            return SDS_ERR_SYMBOLIC;
        }
        if (level > max_level) {
            max_level = level;
        }
    }

    /* 레벨별 프론트 목록을 CSR 형식으로 구성.
     * factor_order 순서대로 삽입하면 같은 레벨 안에서 포스트오더가 유지된다. */
    schedule->num_levels = max_level + 1;
    schedule->level_ptr = (int *)calloc((size_t)schedule->num_levels + 1u, sizeof(int));
    next = (int *)malloc(((size_t)schedule->num_levels + 1u) * sizeof(int));
    if (!schedule->level_ptr || !next) {
        free(levels);
        free(next);
        symbolic_front_schedule_destroy(schedule);
        return SDS_ERR_ALLOC;
    }
    for (int front = 0; front < schedule->num_fronts; ++front) {
        ++schedule->level_ptr[levels[front] + 1];
    }
    for (int level = 0; level < schedule->num_levels; ++level) {
        schedule->level_ptr[level + 1] += schedule->level_ptr[level];
        next[level] = schedule->level_ptr[level];
    }
    for (int i = 0; i < schedule->num_fronts; ++i) {
        const int front = schedule->factor_order[i];
        const int level = levels[front];
        schedule->level_fronts[next[level]++] = front;
    }

    free(next);
    free(levels);
    return SDS_OK;
}

void symbolic_front_schedule_destroy(FrontSchedule *schedule)
{
    if (!schedule) {
        return;
    }
    free(schedule->factor_order);
    free(schedule->forward_order);
    free(schedule->backward_order);
    free(schedule->level_ptr);
    free(schedule->level_fronts);
    memset(schedule, 0, sizeof(*schedule));
}

static int emit_front_postorder(const SeparatorTree *tree,
                                int front_id,
                                int *order,
                                int *count)
{
    if (!tree || !order || !count || front_id < 0 ||
        front_id >= tree->num_separators) {
        return SDS_ERR_BAD_INPUT;
    }
    if (tree->left_child[front_id] != -1) {
        int rc = emit_front_postorder(tree, tree->left_child[front_id],
                                      order, count);
        if (rc != SDS_OK) {
            return rc;
        }
        rc = emit_front_postorder(tree, tree->right_child[front_id],
                                  order, count);
        if (rc != SDS_OK) {
            return rc;
        }
    }
    order[(*count)++] = front_id;
    return SDS_OK;
}

/* 프론트 레벨을 재귀+메모이제이션으로 계산.
 * levels[i] >= 0 이면 이미 계산된 값이므로 즉시 반환한다.
 * 레벨 = max(왼쪽 자식, 오른쪽 자식) + 1. 리프는 레벨 0. */
static int compute_front_level(const SeparatorTree *tree, int front_id, int *levels)
{
    int left_level;
    int right_level;

    if (!tree || !levels || front_id < 0 || front_id >= tree->num_separators) {
        return -1;
    }
    if (levels[front_id] >= 0) {
        return levels[front_id];
    }
    if (tree->left_child[front_id] == -1) {
        levels[front_id] = 0;
        return 0;
    }

    left_level = compute_front_level(tree, tree->left_child[front_id], levels);
    right_level = compute_front_level(tree, tree->right_child[front_id], levels);
    if (left_level < 0 || right_level < 0) {
        return -1;
    }

    levels[front_id] = 1 + (left_level > right_level ? left_level : right_level);
    return levels[front_id];
}
