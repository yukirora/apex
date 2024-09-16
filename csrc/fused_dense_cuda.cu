#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <torch/torch.h>

/* Includes, cuda */
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <rocblas/rocblas.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 11000 || defined(USE_ROCM)
#include <cublasLt.h>
#endif

#include "type_shim.h"

inline void _checkCublasStatus(char const *function, char const *file, long line, int status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("%s[%s:%ld]: ", function, file, line);
        printf("hipBLASlt API failed with status %d\n", status);
        throw std::logic_error("hipBLASlt API failed");
    }
}

#ifndef CHECK_HIP_ERROR
#define CHECK_HIP_ERROR(error)                    \
    if(error != hipSuccess)                       \
    {                                             \
        fprintf(stderr,                           \
                "Hip error: '%s'(%d) at %s:%d\n", \
                hipGetErrorString(error),         \
                error,                            \
                __FILE__,                         \
                __LINE__);                        \
        exit(EXIT_FAILURE);                       \
    }
#endif

#ifndef CHECK_HIPBLASLT_ERROR
#define CHECK_HIPBLASLT_ERROR(error)                                                      \
    if(error != HIPBLAS_STATUS_SUCCESS)                                                   \
    {                                                                                     \
        fprintf(stderr, "hipBLASLt error(Err=%d) at %s:%d\n", error, __FILE__, __LINE__); \
        fprintf(stderr, "\n");                                                            \
        exit(EXIT_FAILURE);                                                               \
    }
#endif




#define checkCublasStatus(status) _checkCublasStatus(__FUNCTION__, __FILE__, __LINE__, status)
/*
Ref: /var/lib/jenkins/pytorch/aten/src/ATen/hip/HIPBlas.cpp
      |   aType    |   bType    |   cType    |     computeType     |
      | ---------- | ---------- | ---------- | ------------------- |
      | HIP_R_16F  | HIP_R_16F  | HIP_R_16F  | HIPBLAS_COMPUTE_16F | Half
      | HIP_R_16F  | HIP_R_16F  | HIP_R_16F  | HIPBLAS_COMPUTE_32F |
      | HIP_R_16F  | HIP_R_16F  | HIP_R_32F  | HIPBLAS_COMPUTE_32F |

      | HIP_R_16BF | HIP_R_16BF | HIP_R_16BF | HIPBLAS_COMPUTE_32F | BF16
      | HIP_R_16BF | HIP_R_16BF | HIP_R_32F  | HIPBLAS_COMPUTE_32F |

      | HIP_R_32F  | HIP_R_32F  | HIP_R_32F  | HIPBLAS_COMPUTE_32F | float

      | HIP_R_64F  | HIP_R_64F  | HIP_R_64F  | HIPBLAS_COMPUTE_64F | double
      
      | HIP_R_8I   | HIP_R_8I   | HIP_R_32I  | HIPBLAS_COMPUTE_32I |
      | HIP_C_32F  | HIP_C_32F  | HIP_C_32F  | HIPBLAS_COMPUTE_32F |
      | HIP_C_64F  | HIP_C_64F  | HIP_C_64F  | HIPBLAS_COMPUTE_64F |



############################################################################################################################################
#
#							   
############################################################################################################################################
uint32_t CUBLASLT_MATMUL_PREF_SEARCH_MODE:             Search mode. See cublasLtMatmulSearch_t. Default is CUBLASLT_SEARCH_BEST_FIT.
uint64_t CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES:     Maximum allowed workspace memory. Default is 0 (no workspace memory allowed).
uint32_t CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK:   Reduction scheme mask. See cublasLtReductionScheme_t. Only algorithm configurations 
                                                       specifying CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME that is not masked out by this attribute 
						       are allowed. For example, a mask value of 0x03 will allow only INPLACE and COMPUTE_TYPE 
						       reduction schemes. Default is CUBLASLT_REDUCTION_SCHEME_MASK (i.e., allows all reduction schemes).
uint32_t CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES:   Minimum buffer alignment for matrix A (in bytes). Selecting a smaller value will exclude algorithms 
                                                       that can not work with matrix A, which is not as strictly aligned as the algorithms need. Default is 256 bytes.
uint32_t CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES:   Minimum buffer alignment for matrix B (in bytes). Selecting a smaller value will exclude algorithms 
                                                       that can not work with matrix B, which is not as strictly aligned as the algorithms need. Default is 256 bytes.
uint32_t CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES:   Minimum buffer alignment for matrix C (in bytes). Selecting a smaller value will exclude algorithms 
                                                       that can not work with matrix C, which is not as strictly aligned as the algorithms need. Default is 256 bytes.
uint32_t CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES:   Minimum buffer alignment for matrix D (in bytes). Selecting a smaller value will exclude algorithms that 
                                                       can not work with matrix D, which is not as strictly aligned as the algorithms need. Default is 256 bytes.
float    CUBLASLT_MATMUL_PREF_MAX_WAVES_COUNT:         Maximum wave count. See cublasLtMatmulHeuristicResult_t::wavesCount. Selecting a non-zero value will exclude 
                                                       algorithms that report device utilization higher than specified. Default is 0.0f.
uint64_t CUBLASLT_MATMUL_PREF_IMPL_MASK:               Numerical implementation details mask. See cublasLtNumericalImplFlags_t. Filters heuristic result to only 
                                                       include algorithms that use the allowed implementations. default: uint64_t(-1) (allow everything)

         CUBLASLT_SEARCH_BEST_FIT:                     Request heuristics for the best algorithm for the given use case.
         CUBLASLT_SEARCH_LIMITED_BY_ALGO_ID:           Request heuristics only for the pre-configured algo id.



############################################################################################################################################
#
#							   
############################################################################################################################################
CUBLASLT_EPILOGUE_DEFAULT = 1                              No special postprocessing, just scale and quantize the results if necessary.

CUBLASLT_EPILOGUE_RELU    = 2                              Apply ReLU point-wise transform to the results (x := max(x, 0)).

CUBLASLT_EPILOGUE_RELU_AUX = 
         CUBLASLT_EPILOGUE_RELU | 128                      Apply ReLU point-wise transform to the results (x := max(x, 0)). 
                                                           This epilogue mode produces an extra output, see CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER of cublasLtMatmulDescAttributes_t.

CUBLASLT_EPILOGUE_BIAS    = 4                              Apply (broadcast) bias from the bias vector. Bias vector length must match matrix D rows, and it must be packed 
                                                           (such as stride between vector elements is 1). Bias vector is broadcast to all columns and added before applying the 
							   final postprocessing.
CUBLASLT_EPILOGUE_RELU_BIAS = 
      CUBLASLT_EPILOGUE_RELU | CUBLASLT_EPILOGUE_BIAS      Apply bias and then ReLU transform.
      
CUBLASLT_EPILOGUE_RELU_AUX_BIAS = 
      CUBLASLT_EPILOGUE_RELU_AUX | CUBLASLT_EPILOGUE_BIAS  Apply bias and then ReLU transform. This epilogue mode produces an extra output, see 
                                                           CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER of cublasLtMatmulDescAttributes_t.

CUBLASLT_EPILOGUE_DRELU = 8 | 128                          Apply ReLu gradient to matmul output. Store ReLu gradient in the output matrix. This epilogue mode requires an extra 
                                                           input, see CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER of cublasLtMatmulDescAttributes_t.
CUBLASLT_EPILOGUE_DRELU_BGRAD = 
      CUBLASLT_EPILOGUE_DRELU | 16                         Apply independently ReLu and Bias gradient to matmul output. Store ReLu gradient in the output matrix, 
                                                           and Bias gradient in the bias buffer (see CUBLASLT_MATMUL_DESC_BIAS_POINTER). This epilogue mode requires an 
							   extra input, see CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER of cublasLtMatmulDescAttributes_t.

CUBLASLT_EPILOGUE_GELU = 32                                Apply GELU point-wise transform to the results (x := GELU(x)).

CUBLASLT_EPILOGUE_GELU_AUX = CUBLASLT_EPILOGUE_GELU | 128  Apply GELU point-wise transform to the results (x := GELU(x)). This epilogue mode outputs GELU input as a separate matrix 
                                                           (useful for training). See CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER of cublasLtMatmulDescAttributes_t.

CUBLASLT_EPILOGUE_GELU_BIAS = 
      CUBLASLT_EPILOGUE_GELU | CUBLASLT_EPILOGUE_BIAS      Apply Bias and then GELU transform 4.

CUBLASLT_EPILOGUE_GELU_AUX_BIAS = 
      CUBLASLT_EPILOGUE_GELU_AUX | CUBLASLT_EPILOGUE_BIAS  Apply Bias and then GELU transform 4. This epilogue mode outputs GELU input as a separate matrix (useful for training). 
                                                           See CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER of cublasLtMatmulDescAttributes_t.

CUBLASLT_EPILOGUE_DGELU = 64 | 128                         Apply GELU gradient to matmul output. Store GELU gradient in the output matrix. This epilogue mode requires an extra 
                                                           input, see CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER of cublasLtMatmulDescAttributes_t.

CUBLASLT_EPILOGUE_DGELU_BGRAD = 
      CUBLASLT_EPILOGUE_DGELU | 16                         Apply independently GELU and Bias gradient to matmul output. Store GELU gradient in the output matrix, and Bias gradient 
                                                           in the bias buffer (see CUBLASLT_MATMUL_DESC_BIAS_POINTER). This epilogue mode requires an extra input, see 
							   CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER of cublasLtMatmulDescAttributes_t.

CUBLASLT_EPILOGUE_BGRADA = 256                             Apply Bias gradient to the input matrix A. The bias size corresponds to the number of rows of the matrix D. 
                                                           The reduction happens over the GEMM’s “k” dimension. Store Bias gradient in the bias buffer, see 
							   CUBLASLT_MATMUL_DESC_BIAS_POINTER of cublasLtMatmulDescAttributes_t.

CUBLASLT_EPILOGUE_BGRADB = 512                             Apply Bias gradient to the input matrix B. The bias size corresponds to the number of columns of the matrix D. 
                                                           The reduction happens over the GEMM’s “k” dimension. Store Bias gradient in the bias buffer, see 
							   CUBLASLT_MATMUL_DESC_BIAS_POINTER of cublasLtMatmulDescAttributes_t.

############################################################################################################################################
#
#							   
############################################################################################################################################
CUBLASLT_MATMUL_DESC_COMPUTE_TYPE: int32_t                 Compute type. Defines the data type used for multiply and accumulate operations, and the accumulator during the matrix multiplication. See cublasComputeType_t.

CUBLASLT_MATMUL_DESC_SCALE_TYPE: int32_t                   Scale type. Defines the data type of the scaling factors alpha and beta. The accumulator value and the value from matrix C are typically converted to scale 
                                                           type before final scaling. The value is then converted from scale type to the type of matrix D before storing in memory. Default value is aligned with 
							   CUBLASLT_MATMUL_DESC_COMPUTE_TYPE. See cudaDataType_t.

CUBLASLT_MATMUL_DESC_POINTER_MODE: int32_t                 Specifies alpha and beta are passed by reference, whether they are scalars on the host or on the device, or device vectors. 
                                                           Default value is: CUBLASLT_POINTER_MODE_HOST (i.e., on the host). See cublasLtPointerMode_t.

CUBLASLT_MATMUL_DESC_TRANSA/TRANSB/TRANSB: int32_t         Specifies the type of transformation operation that should be performed on matrix A/B/C. 
                                                           Default value is: CUBLAS_OP_N (i.e., non-transpose operation).

CUBLASLT_MATMUL_DESC_FILL_MODE: int32_t                    Indicates whether the lower or upper part of the dense matrix was filled, and consequently should be used by the function. 
                                                           Default value is: CUBLAS_FILL_MODE_FULL.See cublasFillMode_t.

CUBLASLT_MATMUL_DESC_EPILOGUE: uint32_t                    Epilogue function. See cublasLtEpilogue_t. Default value is: CUBLASLT_EPILOGUE_DEFAULT.

CUBLASLT_MATMUL_DESC_BIAS_POINTER: void * / const void *   Bias or Bias gradient vector pointer in the device memory. 
                                                           > Input vector with length that matches the number of rows of matrix D when one of the following epilogues is used: 
							     CUBLASLT_EPILOGUE_BIAS, CUBLASLT_EPILOGUE_RELU_BIAS, CUBLASLT_EPILOGUE_RELU_AUX_BIAS, CUBLASLT_EPILOGUE_GELU_BIAS, CUBLASLT_EPILOGUE_GELU_AUX_BIAS.
                                                           > Output vector with length that matches the number of rows of matrix D when one of the following epilogues is used: 
							     CUBLASLT_EPILOGUE_DRELU_BGRAD, CUBLASLT_EPILOGUE_DGELU_BGRAD, CUBLASLT_EPILOGUE_BGRADA.
                                                           > Output vector with length that matches the number of columns of matrix D when one of the following epilogues is used: 
							     CUBLASLT_EPILOGUE_BGRADB.
       
							   Bias vector elements are the same type as alpha and beta (see CUBLASLT_MATMUL_DESC_SCALE_TYPE in this table) when matrix D datatype 
							   is CUDA_R_8I and same as matrix D datatype otherwise. See the datatypes table under cublasLtMatmul() for detailed mapping. Default value is: NULL.

CUBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE: int64_t           Stride (in elements) to the next bias or bias gradient vector for strided batch operations. The default value is 0.

CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER:                Pointer for epilogue auxiliary buffer.
                    void * / const void                   > Output vector for ReLu bit-mask in forward pass when CUBLASLT_EPILOGUE_RELU_AUX or CUBLASLT_EPILOGUE_RELU_AUX_BIAS epilogue is used.
                                                          > Input vector for ReLu bit-mask in backward pass when CUBLASLT_EPILOGUE_DRELU or CUBLASLT_EPILOGUE_DRELU_BGRAD epilogue is used.
                                                          > Output of GELU input matrix in forward pass when CUBLASLT_EPILOGUE_GELU_AUX_BIAS epilogue is used.
                                                          > Input of GELU input matrix for backward pass when CUBLASLT_EPILOGUE_DGELU or CUBLASLT_EPILOGUE_DGELU_BGRAD epilogue is used.
                                                          For aux data type, see CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE. Routines that don’t dereference this pointer, like 
							  cublasLtMatmulAlgoGetHeuristic() depend on its value to determine expected pointer alignment. Requires setting the 
							  CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD attribute.

CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD: int64_t             Leading dimension for epilogue auxiliary buffer. 
                                                          > ReLu bit-mask matrix leading dimension in elements (i.e. bits) when CUBLASLT_EPILOGUE_RELU_AUX, CUBLASLT_EPILOGUE_RELU_AUX_BIAS, 
							    CUBLASLT_EPILOGUE_DRELU_BGRAD, or CUBLASLT_EPILOGUE_DRELU_BGRAD epilogue is used. Must be divisible by 128 and be no less than the number of rows 
							    in the output matrix.
                                                          > GELU input matrix leading dimension in elements when CUBLASLT_EPILOGUE_GELU_AUX_BIAS, CUBLASLT_EPILOGUE_DGELU, or CUBLASLT_EPILOGUE_DGELU_BGRAD 
							    epilogue used. Must be divisible by 8 and be no less than the number of rows in the output matrix.

CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE: int64_t   Batch stride for epilogue auxiliary buffer.
                                                          > ReLu bit-mask matrix batch stride in elements (i.e. bits) when CUBLASLT_EPILOGUE_RELU_AUX, CUBLASLT_EPILOGUE_RELU_AUX_BIAS or 
							    CUBLASLT_EPILOGUE_DRELU_BGRAD epilogue is used. Must be divisible by 128.
                                                          > GELU input matrix batch stride in elements when CUBLASLT_EPILOGUE_GELU_AUX_BIAS, CUBLASLT_EPILOGUE_DRELU, or CUBLASLT_EPILOGUE_DGELU_BGRAD 
							    epilogue used. Must be divisible by 8.
                                                          Default value: 0.

CUBLASLT_MATMUL_DESC_ALPHA_VECTOR_BATCH_STRIDE: int64_t   Batch stride for alpha vector. Used together with CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST when matrix D’s CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT 
                                                          is greater than 1. If CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO is set then CUBLASLT_MATMUL_DESC_ALPHA_VECTOR_BATCH_STRIDE must be set to 0 
							  as this mode doesn’t support batched alpha vector. Default value: 0.

CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET: int32_t             Number of SMs to target for parallel execution. Optimizes heuristics for execution on a different number of SMs when user expects a concurrent stream 
                                                          to be using some of the device resources. Default value: 0.

CUBLASLT_MATMUL_DESC_A_SCALE_POINTER: const void*         Device pointer to the scale factor value that converts data in matrix A to the compute data type range. The scaling factor must have the same type as 
                                                          the compute type. If not specified, or set to NULL, the scaling factor is assumed to be 1. If set for an unsupported matrix data, scale, and compute 
							  type combination, calling cublasLtMatmul() will return CUBLAS_INVALID_VALUE. Default value: NULL

CUBLASLT_MATMUL_DESC_B_SCALE_POINTER: const void*         Equivalent to CUBLASLT_MATMUL_DESC_A_SCALE_POINTER for matrix B. Default value: NULL

CUBLASLT_MATMUL_DESC_C_SCALE_POINTER: const void*         Equivalent to CUBLASLT_MATMUL_DESC_A_SCALE_POINTER for matrix C. Default value: NULL

CUBLASLT_MATMUL_DESC_D_SCALE_POINTER: const void*         Equivalent to CUBLASLT_MATMUL_DESC_A_SCALE_POINTER for matrix D. Default value: NULL

CUBLASLT_MATMUL_DESC_AMAX_D_POINTER: void*                Device pointer to the memory location that on completion will be set to the maximum of absolute values in the output matrix. The computed value has 
                                                          the same type as the compute type. If not specified, or set to NULL, the maximum absolute value is not computed. If set for an unsupported matrix data, 
							  scale, and compute type combination, calling cublasLtMatmul() will return CUBLAS_INVALID_VALUE. Default value: NULL

CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE               The type of the data that will be stored in CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER. If unset (or set to the default value of -1), the data type is 
              int32_t based on cudaDataType               set to be the output matrix element data type (DType) with some exceptions:
                                                          > ReLu uses a bit-mask.
                                                          > For FP8 kernels with an output type (DType) of CUDA_R_8F_E4M3, the data type can be set to a non-default value if:
                                                          1. AType and BType are CUDA_R_8F_E4M3.
                                                          2. Bias Type is CUDA_R_16F.
                                                          3. CType is CUDA_R_16BF or CUDA_R_16F
                                                          4. CUBLASLT_MATMUL_DESC_EPILOGUE is set to CUBLASLT_EPILOGUE_GELU_AUX

                                                          When CType is CUDA_R_16BF, the data type may be set to CUDA_R_16BF or CUDA_R_8F_E4M3. When CType is CUDA_R_16F, the data type may be set to CUDA_R_16F. 
							  Otherwise, the data type should be left unset or set to the default value of -1.

                                                          If set for an unsupported matrix data, scale, and compute type combination, calling cublasLtMatmul() will return CUBLAS_INVALID_VALUE. Default value: -1

CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER: void *   Device pointer to the scaling factor value to convert results from compute type data range to storage data range in the auxiliary matrix that is set 
                                                          via CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER. The scaling factor value must have the same type as the compute type. If not specified, or set to NULL, 
							  the scaling factor is assumed to be 1. If set for an unsupported matrix data, scale, and compute type combination, calling cublasLtMatmul() will return 
							  CUBLAS_INVALID_VALUE. Default value: NULL

CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_AMAX_POINTER: void *    Device pointer to the memory location that on completion will be set to the maximum of absolute values in the buffer that is set via 
                                                          CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER. The computed value has the same type as the compute type. If not specified, or set to NULL, 
							  the maximum absolute value is not computed. If set for an unsupported matrix data, scale, and compute type combination, calling cublasLtMatmul() 
	                                                  will return CUBLAS_INVALID_VALUE. Default value: NULL

CUBLASLT_MATMUL_DESC_FAST_ACCUM: int8_t                   Flag for managing FP8 fast accumulation mode. When enabled, problem execution might be faster but at the cost of lower accuracy because intermediate 
                                                          results will not periodically be promoted to a higher precision. Default value: 0 - fast accumulation mode is disabled

CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE                       Type of the bias or bias gradient vector in the device memory. Bias case: see CUBLASLT_EPILOGUE_BIAS. If unset (or set to the default value of -1), 
   int32_t based on cudaDataTypes                         are the same type as the elements of the output matrix (Dtype) with the following exceptions: IMMA kernels with 
                                                          computeType=CUDA_R_32I and Ctype=CUDA_R_8I where the bias vector elements are the same type as alpha, beta (CUBLASLT_MATMUL_DESC_SCALE_TYPE=CUDA_R_32F)
                                                          For FP8 kernels with an output type of CUDA_R_32F, CUDA_R_8F_E4M3 or CUDA_R_8F_E5M2. See cublasLtMatmul() for more details. Default value: -1

CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_IN_COUNTERS_POINTER      Pointer to a device array of input atomic counters consumed by a matmul. When a counter reaches zero, computation of the corresponding chunk of the 
                                        int32_t *         output tensor is allowed to start. Default: NULL. See Atomics Synchronization.

CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_OUT_COUNTERS_POINTER     Pointer to a device array of output atomic counters produced by a matmul. A matmul kernel sets a counter to zero when the computations of the 
                                        int32_t *         corresponding chunk of the output tensor have completed. All the counters must be initialized to 1 before a matmul kernel is run. Default: NULL. 
							  See Atomics Synchronization.

CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_NUM_CHUNKS_D_ROWS        Number of atomic synchronization chunks in the row dimension of the output matrix D. Each chunk corresponds to a single atomic counter. Default: 0 
                                        int32_t           (atomics synchronization disabled). See Atomics Synchronization.

CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_NUM_CHUNKS_D_COLS        Number of atomic synchronization chunks in the column dimension of the output matrix D. Each chunk corresponds to a single atomic counter. Default: 0 
                                        int32_t           (atomics synchronization disabled). See Atomics Synchronization.
############################################################################################################################################
#
#							   
############################################################################################################################################
*/



int gemm_bias(
                cublasHandle_t handle, cublasOperation_t transa,  cublasOperation_t transb, int m, int n, int k,
                const float* alpha, double *A, int lda, double *B, int ldb, const float* beta, double *C, int ldc) {
// HIP_R_64F  | HIP_R_64F  | HIP_R_64F  | HIPBLAS_COMPUTE_64F
  return cublasGemmEx(handle, transa, transb, m, n, k, alpha, A, CUDA_R_64F, lda, B, CUDA_R_64F, ldb,
                  beta,   C, CUDA_R_64F, ldc, CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT);
}

int gemm_bias(
                cublasHandle_t handle, cublasOperation_t transa,  cublasOperation_t transb, int m, int n, int k,
                const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc) {
// HIP_R_32F  | HIP_R_32F  | HIP_R_32F  | HIPBLAS_COMPUTE_32F
  return cublasGemmEx(handle, transa, transb, m, n, k, alpha, A, CUDA_R_32F, lda, B, CUDA_R_32F, ldb,
                  beta,    C, CUDA_R_32F, ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

int gemm_bias(
                cublasHandle_t handle, cublasOperation_t transa,  cublasOperation_t transb, int m, int n, int k,
                const float* alpha, at::Half * A, int lda, at::Half * B, int ldb, const float* beta, at::Half * C, int ldc) {
//  HIP_R_16F  | HIP_R_16F  | HIP_R_16F  | HIPBLAS_COMPUTE_16F
  return cublasGemmEx(handle, transa, transb, m, n, k, alpha, A, CUDA_R_16F, lda, B, CUDA_R_16F, ldb,
                  beta,    C,  CUDA_R_16F,  ldc,  CUBLAS_COMPUTE_16F,  CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

int gemm_bias(
                cublasHandle_t handle, cublasOperation_t transa,  cublasOperation_t transb, int m, int n, int k,
                const float* alpha, at::BFloat16 *A, int lda, at::BFloat16 *B, int ldb, const float* beta, at::BFloat16 *C, int ldc) {
// HIP_R_16BF | HIP_R_16BF | HIP_R_16BF | HIPBLAS_COMPUTE_32F
  return cublasGemmEx(handle, transa, transb, m, n, k, alpha, A, CUDA_R_16BF, lda, B, CUDA_R_16BF, ldb,
                  beta,    C,  CUDA_R_16BF,  ldc,  CUBLAS_COMPUTE_32F,  CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}


hipDataType get_dtype (at::Tensor A)
{
    hipDataType dataType;

    if (A.scalar_type() == c10::ScalarType::BFloat16) {
           dataType = HIP_R_16F;

    }
    if (A.scalar_type() == at::ScalarType::Half) {
            dataType = HIP_R_16F; 
    }

    if (A.scalar_type() == at::ScalarType::Float) {
           dataType = HIP_R_32F; 
    }
    if (A.scalar_type() == at::ScalarType::Double) {
           dataType = HIP_R_64F;
    }
    // The E4M3 is mainly used for the weights, and the E5M2 is for the gradient.
    if (A.scalar_type() == at::ScalarType::Float8_e5m2fnuz) {
           dataType = HIP_R_8F_E5M2_FNUZ;
    }
    if (A.scalar_type() == at::ScalarType::Float8_e4m3fnuz) {
           dataType = HIP_R_8F_E4M3_FNUZ;
    }

    /*
     Ref. from torch/csrc/TypeInfo.cpp
                  at::ScalarType::Float8_e5m2,                \
                  at::ScalarType::Float8_e5m2fnuz,            \
                  at::ScalarType::Float8_e4m3fn,              \
                  at::ScalarType::Float8_e4m3fnuz,            \

        torch/include/ATen/hip/tunable/GemmHipblaslt.h
                 constexpr hipblasDatatype_t HipBlasDataTypeFor<c10::Float8_e4m3fnuz>() { return HIP_R_8F_E4M3_FNUZ; }
                 constexpr hipblasDatatype_t HipBlasDataTypeFor<c10::Float8_e5m2fnuz>() { return HIP_R_8F_E5M2_FNUZ; }
   */
    return dataType;
}

/********************************************************************************************************************************************************
  *
  * D = Epilogue{  (alpha_s * (A * B) +  beta_s * C) +  bias_v } * scaleD_v
  *
  ******************************************************************************************************************************************************/
int gemm_lt(
           cublasOperation_t   trans_a,
           cublasOperation_t   trans_b,
           const float         *alpha,
           const float         *beta,
           at::Tensor          A,
           at::Tensor          B,
           at::Tensor          C,
           at::Tensor          bias,
           at::Tensor          gelu,
           bool                use_bias,
	   bool                use_grad,
	   bool                use_gelu)
{
    cudaStream_t stream;
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cublasGetStream(handle, &stream);

    hipDataType dtype_a    = get_dtype(A);
    hipDataType dtype_b    = get_dtype(B);
    hipDataType dtype_c    = get_dtype(C);
    hipDataType dtype_bias = get_dtype(bias);
    hipDataType dtype_gelu = get_dtype(gelu);


    const void * d_a    = static_cast<const void*>(A.data_ptr());
    const void * d_b    = static_cast<const void*>(B.data_ptr());
          void * d_c    = static_cast<void *>(C.data_ptr());
    
    auto   d_gelu       = static_cast<void *>(gelu.data_ptr());
    auto   d_bias       = static_cast<void *>(bias.data_ptr());

    hipblasLtEpilogue_t      epilogue   = HIPBLASLT_EPILOGUE_DEFAULT;
    hipblasLtMatrixLayout_t  matA= nullptr, matB= nullptr, matC= nullptr;
    hipblasLtMatmulDesc_t    matmulDesc = nullptr;

    int64_t ld_gelu = (int64_t) C.size(0);

    const int m = trans_a == CUBLAS_OP_T ? A.size(0) : A.size(1);
    const int k = trans_a == CUBLAS_OP_T ? A.size(1) : A.size(0);
    const int n = trans_b == CUBLAS_OP_T ? B.size(1) : B.size(0);

    int lda, ldb, ldd;
    if (trans_a ==CUBLAS_OP_T && trans_b == CUBLAS_OP_N) {  // TN
        lda = k;
        ldb = k;
        ldd = m;
    } else if (trans_a ==CUBLAS_OP_N && trans_b == CUBLAS_OP_N) {  // NN
        lda = m;
        ldb = k;
        ldd = m;
    } else if (trans_a ==CUBLAS_OP_N && trans_b == CUBLAS_OP_T) {  // NT
        lda = m;
        ldb = n;
        ldd = m;
    } 
    else {  // TT
        std::cout << "layout not allowed." << std::endl;
    }

    // std::cout <<"lda: " << lda << "\tldb: " << ldb << "\tldd: " << ldd << "\tm: " << m << "\tk: " << k << "\tn: " << n << std::endl;

    /* ============================================================================================
     *   Matrix layout
     Best combination yet 
    int m = A.size(0);
    int k = A.size(1);
    int n = B.size(1);
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matA, dtype_a, trans_a == CUBLAS_OP_N ? m : k, trans_a == CUBLAS_OP_N ? k : m, k));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matB, dtype_b, trans_b == CUBLAS_OP_N ? k : n, trans_b == CUBLAS_OP_N ? n : k, n));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matC, dtype_c, m, n, m));
     */
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matA, dtype_a, m , k, lda));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matB, dtype_b, k,  n, ldb));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matC, dtype_c, m,  n, ldd));

   /*
    hipblasLtOrder_t rowOrder = HIPBLASLT_ORDER_ROW;
    hipblasLtOrder_t colOrder = HIPBLASLT_ORDER_COL;

    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(matA, HIPBLASLT_MATRIX_LAYOUT_ORDER, &colOrder, sizeof(colOrder)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(matB, HIPBLASLT_MATRIX_LAYOUT_ORDER, &colOrder, sizeof(colOrder)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(matC, HIPBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof(rowOrder)));
    */
    /* ============================================================================================
    * default to 32F except for e5m2 inputs where the config is not supported
    */
    hipDataType desc_dataType=HIP_R_32F;
    hipblasComputeType_t    computeType=HIPBLAS_COMPUTE_32F, desc_computeType=HIPBLAS_COMPUTE_32F;

    if (A.scalar_type() == at::ScalarType::Double) {
           computeType = HIPBLAS_COMPUTE_64F; desc_dataType = HIP_R_64F; desc_computeType = HIPBLAS_COMPUTE_64F;
    }

    /* ============================================================================================
     *   Matmul desc
     */
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescCreate(&matmulDesc, desc_computeType, desc_dataType));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute( matmulDesc, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(int32_t)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute( matmulDesc, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(int32_t)));

    /* ============================================================================================
     *   Configure epilogue
     */
    if (d_bias == nullptr) { use_bias=false; }  if (d_gelu == nullptr) { use_gelu=false; }
  
    if (use_bias && use_gelu) {
         if (use_grad) {  epilogue = HIPBLASLT_EPILOGUE_DGELU_BGRAD;   } 
         else          {  epilogue = HIPBLASLT_EPILOGUE_GELU_AUX_BIAS; }
         CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(matmulDesc,  HIPBLASLT_MATMUL_DESC_BIAS_POINTER,         &d_bias,   sizeof(d_bias)));
         CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(matmulDesc,  HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &d_gelu,   sizeof(d_gelu)));
         CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(matmulDesc,  HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD,      &ld_gelu,  sizeof(ld_gelu)));
    } 
    else if (use_bias) {
         if (use_grad) { epilogue = HIPBLASLT_EPILOGUE_BGRADA; } 
         else          { epilogue = HIPBLASLT_EPILOGUE_BIAS;   }
         CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(matmulDesc,  HIPBLASLT_MATMUL_DESC_BIAS_POINTER,         &d_bias,     sizeof(d_bias)));
         CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(matmulDesc,  HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,       &dtype_bias, sizeof(hipDataType)));
    } 
    else if (use_gelu) {
         if (use_grad) { epilogue = HIPBLASLT_EPILOGUE_DGELU; } 
	 else          { epilogue = HIPBLASLT_EPILOGUE_GELU_AUX; }
         CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(matmulDesc, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &d_gelu,  sizeof(d_gelu)));
         CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(matmulDesc, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD,      &ld_gelu, sizeof(ld_gelu)));
    }
  
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(matmulDesc, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue,  sizeof(epilogue)));
    
    /* ============================================================================================
     *   Algo Get Heuristic
     */
    hipblasLtMatmulPreference_t pref;
    const int   request_solutions = 1;
    int         returnedAlgoCount = 0;
    uint64_t    workspace_size    = 0;
    void*       workspace         = nullptr;
    hipblasLtMatmulHeuristicResult_t heuristicResult[request_solutions];

    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceCreate(&pref));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulAlgoGetHeuristic(handle, matmulDesc, matA, matB, matC, matC, pref, request_solutions, heuristicResult, &returnedAlgoCount));

    if(returnedAlgoCount == 0)  { std::cerr << "No valid solution found!" << std::endl; return 1;  }

    for(int i = 0; i < returnedAlgoCount; i++) { workspace_size = max(workspace_size, heuristicResult[i].workspaceSize); }

    hipMalloc(&workspace, workspace_size);
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace, sizeof(workspace_size)));

    /* ============================================================================================
     * Matmul 
    */
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmul(handle, matmulDesc, 
			                  alpha, d_a, matA, 
					         d_b, matB, beta,
                                          static_cast<const void*>(d_c), matC, 
					                           d_c,  matC, 
					  &heuristicResult[0].algo, 
					  workspace, workspace_size, 
					  stream));
    
    // std::cout << "\nTensor-A:\n" << A << "\nTensor-B:\n" << B << "\nTensor-C:\n" << C << "\nTensor-Bias:\n" << bias << std::endl;

    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matA));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matB));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matC));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescDestroy(matmulDesc));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceDestroy(pref));

    return 0;
}

/****************************************************************************
  *
  *
  **************************************************************************/
template <typename T>
int linear_bias_forward_cuda(
		at::Tensor  input, 
		at::Tensor  weight, 
		at::Tensor  bias,
	        at::Tensor  output)
{
    int         status = HIPBLAS_STATUS_NOT_INITIALIZED;
    const float alpha  = 1.0, beta = 0.0;

    at::Tensor dummy_gelu = at::empty({0}, torch::device(torch::kCUDA).dtype(input.scalar_type()));
    //  y = torch.matmul(inputs, weight) + bias
    status = gemm_lt(CUBLAS_OP_N, CUBLAS_OP_N, &alpha, &beta, weight, input, output, bias, dummy_gelu, true, false, false);

    return status;
}
template int linear_bias_forward_cuda <at::BFloat16>(at::Tensor  input, at::Tensor  weight, at::Tensor  bias, at::Tensor  output); 
template int linear_bias_forward_cuda <c10::Float8_e5m2fnuz>(at::Tensor  input, at::Tensor  weight, at::Tensor  bias, at::Tensor  output); 
template int linear_bias_forward_cuda <c10::Float8_e4m3fnuz>(at::Tensor  input, at::Tensor  weight, at::Tensor  bias, at::Tensor  output); 
template int linear_bias_forward_cuda <at::Half>(at::Tensor  input, at::Tensor  weight, at::Tensor  bias, at::Tensor  output);
template int linear_bias_forward_cuda <float>(at::Tensor  input, at::Tensor  weight, at::Tensor  bias, at::Tensor  output);
template int linear_bias_forward_cuda <double>(at::Tensor  input, at::Tensor  weight, at::Tensor  bias, at::Tensor  output); 

/****************************************************************************
 * In the backward pass, we compute the gradients of the loss with respect to input, weight, and bias.
 * The key matrix operations are:
 *  1. Gradient of Input   (dX): dX  = dY ⋅ WT: Pass `dY`  as matrix `A`, `W`  as matrix `B`, and compute the result into `dX`.
 *  2. Gradient of Weights (dW): dW  = XT ⋅ dY: Pass `X^T` as matrix `A`  `dY` as matrix `B`, and compute the result into `dW`.
 *  3. Gradient of Bias    (db): db=sum(dY)
 *  
 **************************************************************************/
template <typename T>
int linear_bias_backward_cuda(
		at::Tensor    input, 
		at::Tensor    weight, 
		at::Tensor    d_output,  
		at::Tensor    d_weight,  
		at::Tensor    d_bias, 
		at::Tensor    d_input)
{
    int status = HIPBLAS_STATUS_NOT_INITIALIZED;
    const float alpha = 1.0, beta = 0.0;

    at::Tensor dummy_gelu      = at::empty({0}, torch::device(torch::kCUDA).dtype(input.scalar_type()));

    // dW  = XT ⋅ dY and db=sum(dY)
    status = gemm_lt( CUBLAS_OP_T, CUBLAS_OP_N, &alpha, &beta, d_output, input, d_weight, d_bias, dummy_gelu, true, true, false);
    // std::cout << "\ninput:\n" << input << "\nd_output:\n" << d_output <<  "\nd_weight: input x d_output: \n" << d_weight <<  std::endl;

    // std::cout << "\nd_output:\n" << weight << "\nweight:\n" << d_output <<  std::endl;
    // dX  = dY ⋅ WT: (Transposed in Python layer before sending here) 
    status = gemm_lt(CUBLAS_OP_T, CUBLAS_OP_N, &alpha, &beta, weight, d_output, d_input, d_bias, dummy_gelu, false, false, false);
    // std::cout << "\nd_output:\n" << weight << "\nweight:\n" << d_output <<  "\nd_input: d_outputi x weight\n" << d_input <<  std::endl;

    return status;
}

template int linear_bias_backward_cuda<at::BFloat16>(at::Tensor input, at::Tensor weight, at::Tensor d_output, at::Tensor d_weight, at::Tensor d_bias, at::Tensor d_input);
template int linear_bias_backward_cuda<c10::Float8_e5m2fnuz>(at::Tensor input, at::Tensor weight, at::Tensor d_output, at::Tensor d_weight, at::Tensor d_bias, at::Tensor d_input);
template int linear_bias_backward_cuda<c10::Float8_e4m3fnuz>(at::Tensor input, at::Tensor weight, at::Tensor d_output, at::Tensor d_weight, at::Tensor d_bias, at::Tensor d_input);
template int linear_bias_backward_cuda<at::Half>(at::Tensor input, at::Tensor weight, at::Tensor d_output, at::Tensor d_weight, at::Tensor d_bias, at::Tensor d_input);
template int linear_bias_backward_cuda<float>(at::Tensor input, at::Tensor weight, at::Tensor d_output, at::Tensor d_weight, at::Tensor d_bias, at::Tensor d_input);
template int linear_bias_backward_cuda<double>(at::Tensor input, at::Tensor weight, at::Tensor d_output, at::Tensor d_weight, at::Tensor d_bias, at::Tensor d_input);
/****************************************************************************
  *
  *
  **************************************************************************/
template <typename T>
int linear_gelu_linear_forward_cuda( 
		at::Tensor input,
		at::Tensor weight1,  
		at::Tensor bias1, 
		at::Tensor weight2, 
		at::Tensor bias2, 
		at::Tensor output1, 
		at::Tensor output2, 
		at::Tensor gelu_in)
{
    auto batch_size      = input.size(0);
    auto in_features     = input.size(1);
    int  hidden_features = weight1.size(0);
    int  out_features    = weight2.size(0);

    at::Tensor dummy_gelu      = at::empty({0}, torch::device(torch::kCUDA).dtype(input.scalar_type()));

    const float alpha      = 1.0, beta_zero  = 0.0;
    int status  = HIPBLAS_STATUS_NOT_INITIALIZED;

    status = gemm_lt(CUBLAS_OP_T, CUBLAS_OP_N, &alpha, &beta_zero, weight1, input,   output1, bias1,   gelu_in,    true, false, true);
    status = gemm_lt(CUBLAS_OP_T, CUBLAS_OP_N, &alpha, &beta_zero, weight2, output1, bias2,   output2, dummy_gelu, true, false, false);

    return status;
}
template int linear_gelu_linear_forward_cuda<at::BFloat16>(at::Tensor input, at::Tensor weight1, at::Tensor bias1, at::Tensor weight2, at::Tensor bias2, at::Tensor output1, at::Tensor output2, at::Tensor gelu_in);
template int linear_gelu_linear_forward_cuda<at::Half>(at::Tensor input, at::Tensor weight1, at::Tensor bias1, at::Tensor weight2, at::Tensor bias2, at::Tensor output1, at::Tensor output2, at::Tensor gelu_in);
template int linear_gelu_linear_forward_cuda<c10::Float8_e5m2fnuz>(at::Tensor input, at::Tensor weight1, at::Tensor bias1, at::Tensor weight2, at::Tensor bias2, at::Tensor output1, at::Tensor output2, at::Tensor gelu_in);
template int linear_gelu_linear_forward_cuda<c10::Float8_e4m3fnuz>(at::Tensor input, at::Tensor weight1, at::Tensor bias1, at::Tensor weight2, at::Tensor bias2, at::Tensor output1, at::Tensor output2, at::Tensor gelu_in);
template int linear_gelu_linear_forward_cuda<float>(at::Tensor input, at::Tensor weight1, at::Tensor bias1, at::Tensor weight2, at::Tensor bias2, at::Tensor output1, at::Tensor output2, at::Tensor gelu_in);
template int linear_gelu_linear_forward_cuda<double>(at::Tensor input, at::Tensor weight1, at::Tensor bias1, at::Tensor weight2, at::Tensor bias2, at::Tensor output1, at::Tensor output2, at::Tensor gelu_in);
/****************************************************************************
  *
  *
  **************************************************************************/
template <typename T>
int linear_gelu_linear_backward_cuda(
		at::Tensor input, 
		at::Tensor gelu_in, 
		at::Tensor output1, 
		at::Tensor weight1, 
		at::Tensor weight2, 
		at::Tensor d_output1, 
		at::Tensor d_output2, 
		at::Tensor d_weight1, 
		at::Tensor d_weight2, 
		at::Tensor d_bias1, 
		at::Tensor d_bias2, 
		at::Tensor d_input)
{

    auto batch_size  = input.size(0);
    auto in_features = input.size(1);
    int hidden_features = weight1.size(0);
    int out_features    = weight2.size(0);

    const float alpha      = 1.0, beta_zero  = 0.0, beta_one   = 1.0;
    int status  = HIPBLAS_STATUS_NOT_INITIALIZED;
    
    at::Tensor dummy_gelu      = at::empty({0}, torch::device(torch::kCUDA).dtype(input.scalar_type()));

    //wgrad for first gemm
    status = gemm_lt(CUBLAS_OP_N, CUBLAS_OP_T,  &alpha, &beta_zero,  output1,  d_output2, d_weight2, d_bias2, dummy_gelu, true, true, false);

    // hidden_features, batch_size, out_features,
    //dgrad for second GEMM
    status = gemm_lt( CUBLAS_OP_N, CUBLAS_OP_N, &alpha,  &beta_zero,  weight2, d_output2, d_output1, d_bias1, gelu_in, true, true, false);
    return status;
}

 template int  linear_gelu_linear_backward_cuda<at::BFloat16>(at::Tensor input, at::Tensor gelu_in, at::Tensor output1, at::Tensor weight1, at::Tensor weight2,
                at::Tensor d_output1, at::Tensor d_output2, at::Tensor d_weight1, at::Tensor d_weight2, at::Tensor d_bias1, at::Tensor d_bias2, at::Tensor d_input);
 template int  linear_gelu_linear_backward_cuda<at::Half>(at::Tensor input, at::Tensor gelu_in, at::Tensor output1, at::Tensor weight1, at::Tensor weight2,
                at::Tensor d_output1, at::Tensor d_output2, at::Tensor d_weight1, at::Tensor d_weight2, at::Tensor d_bias1, at::Tensor d_bias2, at::Tensor d_input);
 template int  linear_gelu_linear_backward_cuda<c10::Float8_e5m2fnuz>(at::Tensor input, at::Tensor gelu_in, at::Tensor output1, at::Tensor weight1, at::Tensor weight2,
                at::Tensor d_output1, at::Tensor d_output2, at::Tensor d_weight1, at::Tensor d_weight2, at::Tensor d_bias1, at::Tensor d_bias2, at::Tensor d_input);
 template int  linear_gelu_linear_backward_cuda<c10::Float8_e4m3fnuz>(at::Tensor input, at::Tensor gelu_in, at::Tensor output1, at::Tensor weight1, at::Tensor weight2,
                at::Tensor d_output1, at::Tensor d_output2, at::Tensor d_weight1, at::Tensor d_weight2, at::Tensor d_bias1, at::Tensor d_bias2, at::Tensor d_input);
 template int  linear_gelu_linear_backward_cuda<float>(at::Tensor input, at::Tensor gelu_in, at::Tensor output1, at::Tensor weight1, at::Tensor weight2,
                at::Tensor d_output1, at::Tensor d_output2, at::Tensor d_weight1, at::Tensor d_weight2, at::Tensor d_bias1, at::Tensor d_bias2, at::Tensor d_input);
 template int  linear_gelu_linear_backward_cuda<double>(at::Tensor input, at::Tensor gelu_in, at::Tensor output1, at::Tensor weight1, at::Tensor weight2,
                at::Tensor d_output1, at::Tensor d_output2, at::Tensor d_weight1, at::Tensor d_weight2, at::Tensor d_bias1, at::Tensor d_bias2, at::Tensor d_input);
