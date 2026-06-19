#include <cholmod.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace {

template <typename T>
T* checked(T* ptr, const char* what) {
  if (!ptr) throw std::runtime_error(what);
  return ptr;
}

cholmod_sparse* symmetrized_pattern(cholmod_sparse* input,
                                    cholmod_common* common) {
  cholmod_sparse* a =
      checked(cholmod_copy(input, 0, 0, common), "cholmod_copy");
  cholmod_sparse_xtype(CHOLMOD_PATTERN, a, common);
  cholmod_sparse* at =
      checked(cholmod_transpose(a, 0, common), "cholmod_transpose");
  double one[2] = {1.0, 0.0};
  cholmod_sparse* full =
      checked(cholmod_add(a, at, one, one, 0, 1, common), "cholmod_add");
  cholmod_free_sparse(&a, common);
  cholmod_free_sparse(&at, common);
  full->stype = 1;  // use the upper triangle of the explicit A + A' graph
  return full;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s MATRIX.mtx OUT.csv\n", argv[0]);
    return 2;
  }

  cholmod_common common;
  cholmod_start(&common);
  common.supernodal = CHOLMOD_SUPERNODAL;
  common.nmethods = 1;
  common.method[0].ordering = CHOLMOD_METIS;
  common.Postorder = true;
  common.final_super = true;

  try {
    FILE* f = std::fopen(argv[1], "r");
    if (!f) throw std::runtime_error("open input");
    cholmod_sparse* input =
        checked(cholmod_read_sparse(f, &common), "cholmod_read_sparse");
    std::fclose(f);
    if (input->nrow != input->ncol)
      throw std::runtime_error("matrix must be square");

    cholmod_sparse* graph = symmetrized_pattern(input, &common);
    cholmod_factor* factor =
        checked(cholmod_analyze(graph, &common), "cholmod_analyze");
    if (!factor->is_super)
      throw std::runtime_error("CHOLMOD did not produce supernodal analysis");

    FILE* out = std::fopen(argv[2], "w");
    if (!out) throw std::runtime_error("open output");
    std::fprintf(out, "front,fsz,nc,uc\n");
    int* super = static_cast<int*>(factor->super);
    int* pi = static_cast<int*>(factor->pi);
    for (size_t s = 0; s < factor->nsuper; ++s) {
      const int nc = super[s + 1] - super[s];
      const int fsz = pi[s + 1] - pi[s];
      std::fprintf(out, "%zu,%d,%d,%d\n", s, fsz, nc, fsz - nc);
    }
    std::fclose(out);

    std::fprintf(
        stderr, "[cholmod-fronts] n=%zu nsuper=%zu lnz=%.0f fl=%.0f wrote %s\n",
        input->nrow, factor->nsuper, common.lnz, common.fl, argv[2]);

    cholmod_free_factor(&factor, &common);
    cholmod_free_sparse(&graph, &common);
    cholmod_free_sparse(&input, &common);
    cholmod_finish(&common);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    cholmod_finish(&common);
    return 1;
  }
}
