#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>

#include "type_shim.h"
#include "multi_tensor_apply.cuh"

#define BLOCK_SIZE 512
#define ILP 4

template<typename T>
struct LARCFunctor
{
   __device__ __forceinline__ void operator()(
    int chunk_size,
    volatile int* noop_gmem,
    TensorListMetadata<2>* tl,
    float *grad_norms,
    float *param_norms,
    const float lr,
    const float trust_coefficient,
    const float epsilon,
    const float weight_decay,
    const bool clip) {
    int tensor_loc = tl->block_to_tensor[blockIdx.x];

    int chunk_idx = tl->block_to_chunk[blockIdx.x];
    int n = tl->sizes[tensor_loc];
    int num_chunks = (n * chunk_size - 1) / chunk_size;
    if (num_chunks - 1 == chunk_idx && n % chunk_size != 0) {
      chunk_size = n % chunk_size;
    }
    n = min(n, chunk_size);

    T* g = (T*) tl->addresses[0][tensor_loc];
    g += chunk_idx * chunk_size;

    T* p = (T*) tl->addresses[1][tensor_loc];
    p += chunk_idx * chunk_size;

    int tensor_offset = tl->start_tensor_this_launch + tensor_loc;
    float g_norm = grad_norms[tensor_offset];
    float p_norm = param_norms[tensor_offset];

    float adaptive_lr = trust_coefficient * p_norm / (g_norm + p_norm * weight_decay + epsilon);
    if (clip) {
      adaptive_lr = min(adaptive_lr / lr, 1);
    }

    if (weight_decay != 0.0f) {
      for (int i_start = 0; i_start < n; i_start += blockDim.x * ILP) {
#pragma unroll
        for (int i = i_start + threadIdx.x;
            i < i_start + threadIdx.x + ILP * blockDim.x && i < n;
            i += blockDim.x) {
          g[i] += weight_decay * p[i];
          g[i] = static_cast<float>(g[i]) * adaptive_lr;
        }
      }
    }
    else {
      // Avoid reading p when weight decay = 0.0f
      for (int i_start = 0; i_start < n; i_start += blockDim.x * ILP) {
#pragma unroll
        for (int i = i_start + threadIdx.x;
            i < i_start + threadIdx.x + ILP * blockDim.x && i < n;
            i += blockDim.x) {
          g[i] = static_cast<float>(g[i]) * adaptive_lr;
        }
      }
    }
  }
};

void multi_tensor_larc_cuda(
  int chunk_size,
  at::Tensor noop_flag,
  std::vector<std::vector<at::Tensor>> tensor_lists,
  at::Tensor grad_norms,
  at::Tensor param_norms,
  const float lr,
  const float trust_coefficient,
  const float epsilon,
  const float weight_decay,
  const bool clip)
{
  using namespace at;

  DISPATCH_DOUBLE_FLOAT_AND_HALF_AND_BFLOAT16(
    tensor_lists[0][0].scalar_type(), 0, "larc",
    multi_tensor_apply<2>(
      BLOCK_SIZE,
      chunk_size,
      noop_flag,
      tensor_lists,
      LARCFunctor<scalar_t_0>(),
      grad_norms.DATA_PTR<float>(),
      param_norms.DATA_PTR<float>(),
      lr,
      trust_coefficient,
      epsilon,
      weight_decay,
      clip);)

  AT_CUDA_CHECK(cudaGetLastError());
}
