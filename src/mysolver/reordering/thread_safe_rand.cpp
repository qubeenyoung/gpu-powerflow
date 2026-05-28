// cy168: thread-safe rand()/srand() for DETERMINISTIC parallel nested dissection.
// Build as a shared lib (libdet_rand.so) and LD_PRELOAD it. METIS (USE_GKRAND off) uses
// glibc rand()/srand() whose state is a single GLOBAL -> the parallel-ND worker threads
// race on it -> NON-deterministic ordering. LD_PRELOAD reliably interposes the versioned
// glibc rand@GLIBC_2.2.5 reference inside libmetis.so (unlike link-time interposition, which
// is versioned-symbol-fragile). thread-local rand_r + the per-call srand(42) reseed in
// metis_nd.cpp -> each subgraph ordered from a fixed RNG state independent of thread/timing
// -> fully REPRODUCIBLE parallel ordering (verified: fill identical across runs), at METIS
// quality (fill ~= serial), NO system-library rebuild, fully reversible (drop LD_PRELOAD).
//   Usage:  LD_PRELOAD=<build>/libdet_rand.so ./benchmark --solver mysolver-gpu ...
#include <cstdlib>

namespace { thread_local unsigned int ts_rand_state = 0u; }

extern "C" int rand(void) { return static_cast<int>(rand_r(&ts_rand_state) & 0x7fffffff); }
extern "C" void srand(unsigned int seed) { ts_rand_state = seed; }
