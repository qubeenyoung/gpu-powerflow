#ifndef EXP_20260520_SYMBOLIC_SUPERNODE_H
#define EXP_20260520_SYMBOLIC_SUPERNODE_H

#include "matrix/csc_matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    SYMBOLIC_SUPERNODE_PATTERN_FULL_COLUMN = 0,
    SYMBOLIC_SUPERNODE_PATTERN_STRICT_TAIL = 1
} SymbolicSupernodePatternMode;

typedef struct {
    int n;
    int num_supernodes;
    int max_width;

    /*
     * Supernode s owns columns:
     *   [supernode_ptr[s], supernode_ptr[s + 1])
     *
     * Length is num_supernodes + 1.
     */
    int *supernode_ptr;

    /*
     * column_to_supernode[col] gives the supernode id for each column.
     * Length is n.
     */
    int *column_to_supernode;
} SymbolicSupernodeSet;

/*
 * Find strict supernodes from a CSC factor pattern.
 *
 * The default strict rule compares each column's below-diagonal tail pattern
 * (`row > col`). This avoids splitting every column solely because L stores a
 * different diagonal row in each column.
 */
int symbolic_supernodes_find_strict(const CSCMatrix *factor_pattern,
                                    SymbolicSupernodeSet *supernodes);

/*
 * Same as symbolic_supernodes_find_strict(), but lets callers choose whether
 * to compare the full column pattern or the below-diagonal tail pattern.
 */
int symbolic_supernodes_find_strict_with_mode(
    const CSCMatrix *factor_pattern,
    SymbolicSupernodePatternMode mode,
    SymbolicSupernodeSet *supernodes);

int symbolic_supernodes_validate(const SymbolicSupernodeSet *supernodes);
void symbolic_supernodes_destroy(SymbolicSupernodeSet *supernodes);

#ifdef __cplusplus
}
#endif

#endif
