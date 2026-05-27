#include "symbolic/symbolic_print.h"
#include "symbolic/symbolic_internal.h"

#include <stdio.h>

void symbolic_factorization_print(const SymbolicFactorization *symbolic)
{
    if (!symbolic) {
        return;
    }

    printf("SymbolicFactorization: n=%d num_fronts=%d\n",
           symbolic->n, symbolic->num_fronts);
    symbolic_print_int_array("input_perm", symbolic->input_perm, symbolic->n);
    symbolic_print_int_array("input_inv_perm", symbolic->input_inv_perm, symbolic->n);
    symbolic_print_int_array("separator_perm", symbolic->separator_perm, symbolic->n);
    symbolic_print_int_array("separator_inv_perm", symbolic->separator_inv_perm, symbolic->n);
    symbolic_print_int_array("final_perm", symbolic->final_perm, symbolic->n);
    symbolic_print_int_array("final_inv_perm", symbolic->final_inv_perm, symbolic->n);
    symbolic_print_int_array("etree_parent", symbolic->etree_parent, symbolic->n);
    symbolic_print_int_array("etree_postorder", symbolic->etree_postorder, symbolic->n);
    separator_tree_print(&symbolic->separator_tree);

    printf("FrontSymbolic:\n");
    for (int i = 0; i < symbolic->num_fronts; ++i) {
        const FrontSymbolic *front = &symbolic->fronts[i];
        printf("  front=%d sep=%d parent=%d left=%d right=%d range=[%d,%d)\n",
               front->front_id, front->separator_id, front->parent,
               front->left_child, front->right_child,
               front->sep_begin, front->sep_end);
        symbolic_print_int_array("    pivot_vars", front->pivot_vars,
                                 front->num_pivots);
        symbolic_print_int_array("    update_vars", front->update_vars,
                                 front->num_updates);
        symbolic_print_int_array("    front_vars", front->front_vars,
                                 front->num_front_vars);
        symbolic_print_int_array("    update_to_parent", front->update_to_parent,
                                 front->num_update_to_parent);
    }

    printf("FrontSchedule: num_levels=%d\n", symbolic->schedule.num_levels);
    symbolic_print_int_array("  factor_order", symbolic->schedule.factor_order,
                             symbolic->schedule.num_fronts);
    symbolic_print_int_array("  forward_order", symbolic->schedule.forward_order,
                             symbolic->schedule.num_fronts);
    symbolic_print_int_array("  backward_order", symbolic->schedule.backward_order,
                             symbolic->schedule.num_fronts);
    symbolic_print_int_array("  level_ptr", symbolic->schedule.level_ptr,
                             symbolic->schedule.num_levels + 1);
    symbolic_print_int_array("  level_fronts", symbolic->schedule.level_fronts,
                             symbolic->schedule.num_fronts);

    printf("FrontStoragePlan: total_dense_entries=%zu total_dense_bytes=%zu\n",
           symbolic->storage.total_dense_entries,
           symbolic->storage.total_dense_bytes);
    printf("  id npiv nupd nfront F L11 U11 L21 U12 C\n");
    for (int i = 0; i < symbolic->storage.num_fronts; ++i) {
        printf("  %d %d %d %d %zu %zu %zu %zu %zu %zu\n",
               i,
               symbolic->storage.npiv[i],
               symbolic->storage.nupd[i],
               symbolic->storage.nfront[i],
               symbolic->storage.F_offset[i],
               symbolic->storage.L11_offset[i],
               symbolic->storage.U11_offset[i],
               symbolic->storage.L21_offset[i],
               symbolic->storage.U12_offset[i],
               symbolic->storage.C_offset[i]);
    }

    printf("EntryAssemblyPlan: num_entries=%d\n",
           symbolic->entry_assembly.num_entries);
    for (int i = 0; i < symbolic->entry_assembly.num_entries && i < 16; ++i) {
        printf("  entry=%d source=%d front=%d local=(%d,%d) F_offset=%zu\n",
               i,
               symbolic->entry_assembly.source_index[i],
               symbolic->entry_assembly.target_front[i],
               symbolic->entry_assembly.local_row[i],
               symbolic->entry_assembly.local_col[i],
               symbolic->entry_assembly.F_offset[i]);
    }
    if (symbolic->entry_assembly.num_entries > 16) {
        printf("  ...\n");
    }

    printf("ContributionAssemblyPlan: child_fronts=%d total_update_indices=%d\n",
           symbolic->contribution_assembly.num_child_fronts,
           symbolic->contribution_assembly.total_update_indices);
    printf("  dense contribution entries are not materialized; numeric assembly uses update_to_parent.\n");
}
