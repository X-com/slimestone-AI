#pragma once

// Shared qualifier for functions callable from both host and device code, and from device
// kernels. All of src/kernel/ is written to build twice: once as ordinary host C++ (so it can
// be unit-validated against src/reference/ with a plain compiler before touching a GPU), and
// once compiled by nvcc for the device. No exceptions, no std::vector/std::string/std::sort on
// the HD path - see world.cuh's PosKeySet/FixedWorld for the fixed-capacity + sticky-error-flag
// replacement for "throw std::out_of_range".
#if defined(__CUDACC__)
#define MCPGPU_HD __host__ __device__
#else
#define MCPGPU_HD
#endif
