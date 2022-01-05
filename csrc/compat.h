#ifndef TORCH_CHECK
#define TORCH_CHECK AT_CHECK
#endif

#ifdef VERSION_GE_1_3
#define DATA_PTR data_ptr
#else
#define DATA_PTR data
#endif

#ifdef __HIP_PLATFORM_HCC__
constexpr int GPU_WARP_SIZE = 64;
#else
constexpr int GPU_WARP_SIZE = 32;
#endif
