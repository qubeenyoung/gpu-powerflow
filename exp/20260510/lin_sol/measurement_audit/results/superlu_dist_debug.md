Reproduction of prior failure:

```text
Invalid ISPEC at line 556 in file /workspace/gpu-powerflow/exp/20260510/lin_sol/third_party/superlu_dist/src/SRC/prec-independent/get_perm_c.c
```

Installed SuperLU_DIST configuration excerpts:

```text
/* #undef HAVE_PARMETIS */
/* #undef HAVE_COLAMD */
typedef enum {NOROWPERM, LargeDiag_MC64, LargeDiag_HWPM, MY_PERMR} rowperm_t;
typedef enum {NATURAL, MMD_ATA, MMD_AT_PLUS_A, COLAMD,
```

`get_perm_c.c` accepts `METIS_AT_PLUS_A` only inside `#ifdef HAVE_PARMETIS`; otherwise it falls through to `ABORT("Invalid ISPEC")`:

```c
t = SuperLU_timer_();

    switch ( ispec ) {

        case NATURAL: /* Natural ordering */
	      for (i = 0; i < n; ++i) perm_c[i] = i;
#if ( PRNTlevel>=1 )
	      if ( !pnum ) printf(".. Use natural column ordering\n");
#endif
	      return;

        case MMD_AT_PLUS_A: /* Minimum degree ordering on A'+A */
	      if ( m != n ) ABORT("Matrix is not square");
	      at_plus_a_dist(n, Astore->nnz, Astore->colptr, Astore->rowind,
			     &bnz, &b_colptr, &b_rowind);
	      t = SuperLU_timer_() - t;
	      /*printf("Form A'+A time = %8.3f\n", t);*/
#if ( PRNTlevel>=1 )
	      if ( !pnum ) printf(".. Use minimum degree ordering on A'+A.\n");
#endif
	      break;

        case MMD_ATA: /* Minimum degree ordering on A'*A */
	      getata_dist(m, n, Astore->nnz, Astore->colptr, Astore->rowind,
			  &bnz, &b_colptr, &b_rowind);
	      t = SuperLU_timer_() - t;
	      /*printf("Form A'*A time = %8.3f\n", t);*/
#if ( PRNTlevel>=1 )
	      if ( !pnum ) printf(".. Use minimum degree ordering on A'*A\n");
#endif
	      break;

        case (COLAMD): /* Approximate minimum degree column ordering. */
	      get_colamd_dist(m, n, Astore->nnz, Astore->colptr, Astore->rowind,
			      perm_c);
#if ( PRNTlevel>=1 )
	      printf(".. Use approximate minimum degree column ordering.\n");
#endif
	      return;
#ifdef HAVE_PARMETIS
        case METIS_AT_PLUS_A: /* METIS ordering on A'+A */
	      if ( m != n ) ABORT("Matrix is not square");
	      at_plus_a_dist(n, Astore->nnz, Astore->colptr, Astore->rowind,
			     &bnz, &b_colptr, &b_rowind);

	      if ( bnz ) { /* non-empty adjacency structure */
		  get_metis_dist(n, bnz, b_colptr, b_rowind, perm_c);
	      } else { /* e.g., diagonal matrix */
		  for (i = 0; i < n; ++i) perm_c[i] = i;
		  SUPERLU_FREE(b_colptr);
		  /* b_rowind is not allocated in this case */
	      }

#if ( PRNTlevel>=1 )
	      if ( !pnum ) printf(".. Use METIS ordering on A'+A\n");
#endif
	      return;
#endif /* matching ifdef HAVE_PARMETIS */

        default:
	      ABORT("Invalid ISPEC");
    }

    if ( bnz ) {
	t = SuperLU_timer_();
```

Audit repair attempts:

- Built audit executables for `NATURAL`, `MMD_AT_PLUS_A`, and `MMD_ATA`, with both `LargeDiag_MC64` and `NOROWPERM` row permutation variants.
- `Invalid ISPEC` is fixed by avoiding `METIS_AT_PLUS_A` in this no-ParMETIS build.
- Correctness depends on row/column permutation and on not reusing the same ABglobal `SuperMatrix` across repeated in-process solves. The original wrapper therefore remains invalid for repeated timing.
