#pragma once

// ---------------------------------------------------------------------------
// storage_convert.hpp
//
// Device-side conversion launchers for storage upload/download, replacing the
// per-element host cast/trig loops. Inputs/outputs that are complex are passed
// as interleaved double arrays (length 2*count: re,im,re,im,...), layout-
// compatible with std::complex<double>. Defined in storage_convert.cu.
// ---------------------------------------------------------------------------

#ifdef CUPF_WITH_CUDA

#include <cstdint>

// Seed rectangular + polar voltage from an interleaved complex V0 (device):
//   v_re=Re, v_im=Im, va=atan2(Im,Re), vm=hypot(Re,Im)  (cast to StorageT).
template <typename StorageT>
void launch_seed_state_from_v0(const double* v0_interleaved,
                               StorageT* v_re, StorageT* v_im,
                               StorageT* va, StorageT* vm,
                               int32_t count);

// Split an interleaved complex array (device) into StorageT re/im buffers.
template <typename StorageT>
void launch_split_complex(const double* src_interleaved,
                          StorageT* dst_re, StorageT* dst_im,
                          int32_t count);

// Pack StorageT re/im buffers into an interleaved complex<double> output
// (device), for a single bulk D2H into the host result vector.
template <typename StorageT>
void launch_pack_complex_to_double(const StorageT* re, const StorageT* im,
                                   double* dst_interleaved,
                                   int32_t count);

#endif  // CUPF_WITH_CUDA
