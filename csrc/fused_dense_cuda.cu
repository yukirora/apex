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

/********************************************************************************************************************************************************
  *
  * In the forward pass of a neural network layer, the input is multiplied by the weight  matrix, and an activation function is applied:
  * Y = XW + b
  * where X is the input matrix, W is the weight matrix, b is the bias, and Y is the output.
  *
  * D = Epilogue{  alpha_s * (A * B) +  beta_s * C +  bias_vi } * scaleD_v
  *
  ******************************************************************************************************************************************************/
int gemm_bias_lt(
                cublasOperation_t   trans_a,
                cublasOperation_t   trans_b,
                const float         *alpha,
                const float         *beta,
                at::Tensor          A,
                at::Tensor          B,
                at::Tensor          bias,
                at::Tensor          C,
                bool                use_bias)
{

    hipDataType             dataType, desc_dataType;
    hipblasComputeType_t    computeType, desc_computeType;

    if (A.scalar_type() == c10::ScalarType::BFloat16) {
           dataType         = HIP_R_16F;           computeType      = HIPBLAS_COMPUTE_32F;
           desc_dataType    = HIP_R_32F;           desc_computeType = HIPBLAS_COMPUTE_32F;

    }
    if (A.scalar_type() == at::ScalarType::Half) {
           dataType         = HIP_R_16F;           computeType      = HIPBLAS_COMPUTE_32F;
           desc_dataType    = HIP_R_32F;           desc_computeType = HIPBLAS_COMPUTE_32F;
    } 

    if (A.scalar_type() == at::ScalarType::Float) {
           dataType         = HIP_R_32F;           computeType      = HIPBLAS_COMPUTE_32F;
           desc_dataType    = HIP_R_32F;           desc_computeType = HIPBLAS_COMPUTE_32F;
    }
    if (A.scalar_type() == at::ScalarType::Double) {
           dataType         = HIP_R_64F;           computeType      = HIPBLAS_COMPUTE_64F;
           desc_dataType    = HIP_R_64F;           desc_computeType = HIPBLAS_COMPUTE_64F;
    }

    cudaStream_t stream;
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cublasGetStream(handle, &stream);

    /* ============================================================================================
     *   Matmul desc
     */
    hipblasLtMatmulDesc_t matmul   = nullptr;

    /* cublasStatus_t  hipblasLtMatmulDescCreate(cublasLtMatmulDesc_t *matmulDesc,
     *                                           cublasComputeType_t   computeType,
     *                                           cudaDataType_t        scaleType);
     *
     * This function creates a matrix multiply descriptor by allocating the memory needed to hold its opaque structure.

     * matmulDesc:  Output ==> Pointer to the structure holding the matrix multiply descriptor created by this function. See cublasLtMatmulDesc_t.
     * computeType: Input  ==> Enumerant that specifies the data precision for the matrix multiply descriptor this function creates. See cublasComputeType_t.
     * scaleType:   Input  ==> Enumerant that specifies the data precision for the matrix transform descriptor this function creates. See cudaDataType.
     */
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescCreate(&matmul, desc_computeType, desc_dataType));


    /* cublasStatus_t cublasLtMatmulDescSetAttribute( cublasLtMatmulDesc_t matmulDesc,
     *                                                cublasLtMatmulDescAttributes_t attr,
     *                                                const void *buf,
     *                                                size_t sizeInBytes);
     *
     * matmulDesc:  Input ==> Pointer to the previously created structure holding the matrix multiply descriptor queried by this function. See cublasLtMatmulDesc_t.
     * attr:        Input ==> The attribute that will be set by this function. See cublasLtMatmulDescAttributes_t.
     * buf:         Input ==> The value to which the specified attribute should be set.
     * sizeInBytes: Input ==> Size of buf buffer (in bytes) for verification.
     * Ref: https://docs.nvidia.com/cuda/cublas/#cublasltmatmuldescattributes-t
    */
    /*
     * CUBLASLT_MATMUL_DESC_TRANSA/CUBLASLT_MATMUL_DESC_TRANSB: int32_t  
     *
     * Specifies the type of transformation operation that should be performed on matrix A/matrix B. 
     * Default value is: CUBLAS_OP_N  (i.e., non-transpose operation).
    */

    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute( matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(int32_t)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute( matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(int32_t)));


     /* ============================================================================================
     *   Configure epilogue
     */

     hipblasLtEpilogue_t   epilogue = HIPBLASLT_EPILOGUE_DEFAULT;
   
     auto   d_bias = static_cast<void *>(bias.data_ptr());
     if ((use_bias) && (d_bias != nullptr)) {
    /* CUBLASLT_EPILOGUE_BIAS  Apply (broadcast) bias from the bias vector. Bias vector length must match matrix D rows, and it must be packed
     *                             (such as stride between vector elements is 1). Bias vector is broadcast to all columns and added before applying the final postprocessing.
    */
        epilogue = HIPBLASLT_EPILOGUE_BIAS;


        CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute( matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE,        &epilogue,  sizeof(epilogue)));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute( matmul, HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,  &dataType,  sizeof(&dataType)));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute( matmul, HIPBLASLT_MATMUL_DESC_BIAS_POINTER,    &d_bias,     sizeof(void*)));
    }
    
    /* ============================================================================================
     *   Matrix layout
     */
    hipblasLtMatrixLayout_t matA= nullptr, matB= nullptr, matC= nullptr;

    /*  cublasStatus_t  cublasLtMatrixLayoutCreate( cublasLtMatrixLayout_t *matLayout,
     *                                              cudaDataType           type,
     *                                              uint64_t               rows,
     *                                              uint64_t               cols,
     *                                              int64_t                ld);
     *  This function creates a matrix layout descriptor by allocating the memory needed to hold its opaque structure.
     *
     *  matLayout:  Output ==> Pointer to the structure holding the matrix layout descriptor created by this function. See cublasLtMatrixLayout_t.
     *  type:       Input  ==> Enumerant that specifies the data precision for the matrix layout descriptor this function creates. See cudaDataType.
     *  rows, cols: Input  ==> Number of rows and columns of the matrix.
     *  ld:         Input  ==> The leading dimension of the matrix. In column major layout, this is the number of elements to jump to reach the next column.
     *                         Thus ld >= m (number of rows).
    */
    int m = A.size(1);
    int n = A.size(0);
    int k = B.size(1);

    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matA, dataType, trans_a == CUBLAS_OP_N ? k : m, trans_a == CUBLAS_OP_N ? m : k, k));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matB, dataType, trans_b == CUBLAS_OP_N ? m : n, trans_b == CUBLAS_OP_N ? n : m, m));

    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matC, dataType, k, n, k));

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
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulAlgoGetHeuristic(handle, matmul, matA, matB, matC, matC, pref, request_solutions, heuristicResult, &returnedAlgoCount));

    if(returnedAlgoCount == 0)  { std::cerr << "No valid solution found!" << std::endl; return 1;  }

    for(int i = 0; i < returnedAlgoCount; i++) { workspace_size = max(workspace_size, heuristicResult[i].workspaceSize); }

    hipMalloc(&workspace, workspace_size);
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace, sizeof(workspace_size)));

    /* ============================================================================================
     * Matmul 
     *    lightHandle:                     Input  ==> Pointer to the allocated cuBLASLt handle for the cuBLASLt context. See cublasLtHandle_t.
     *    computeDesc:                     Input  ==> Handle to a previously created matrix multiplication descriptor of type cublasLtMatmulDesc_t.
     *    alpha, beta: Device/host memory: Input  ==> Pointers to the scalars used in the multiplication.
     *    A, B, and C: Device memory:      Input  ==> Pointers to the GPU memory associated with the corresponding descriptors Adesc, Bdesc and Cdesc.
     *    Adesc, Bdesc and Cdesc :         Input  ==> Handles to the previous created descriptors of the type cublasLtMatrixLayout_t.
     *    D:           Device memory       Output ==> Pointer to the GPU memory associated with the descriptor Ddesc.
     *    Ddesc:                           Input  ==> Handle to the previous created descriptor of the type cublasLtMatrixLayout_t.
     *    algo:                            Input  ==> Handle for matrix multiplication algorithm to be used. See cublasLtMatmulAlgo_t. When NULL,
     *                                                an implicit heuritics query with default search preferences will be performed to determine actual algorithm to use.
     *    workspace: Device memory:               ==> Pointer to the workspace buffer allocated in the GPU memory. Must be 256B aligned (i.e. lowest 8 bits of address must be 0).
     *    workspaceSizeInBytes:            Input  ==> Size of the workspace.
     *    stream:      Host memory         Input  ==> The CUDA stream where all the GPU work will be submitted.
    */
 
    // std::cout << "\n========================\ngemm_bias_lt\n";
    // std::cout << "\nTensor-A:\n" << A << "\nTensor-B:\n" << B << "\nTensor-C:\n" << C << "\nTensor-Bias:\n" << bias << std::endl;
    // std::cout << "\n========================\n";
    const void * d_a    = static_cast<const void*>(A.data_ptr());
    const void * d_b    = static_cast<const void*>(B.data_ptr());
          void * d_c    = static_cast<void *>(C.data_ptr());

    CHECK_HIPBLASLT_ERROR(hipblasLtMatmul(handle, matmul, alpha, d_b, matA, d_a, matB, beta,
                                          static_cast<const void*>(d_c), matC, d_c, matC, &heuristicResult[0].algo, workspace, workspace_size, stream));

    // std::cout << "\n========================\ngemm_bias_lt\n"; 
    // std::cout << "\nTensor-A:\n" << A << "\nTensor-B:\n" << B << "\nTensor-C:\n" << C << "\nTensor-Bias:\n" << bias << std::endl;
    // std::cout << "\n========================\n";

    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matA));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matB));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matC));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescDestroy(matmul));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceDestroy(pref));

    return 0;

}

/********************************************************************************************************************************************************
  * In the backward pass, we compute the gradients of the loss with respect to X, W, and b. The key matrix operations are:
  *  1. Gradient of Input (dX): dX = dY ⋅ WT:    Pass `dY`  as matrix `A`, `W` as matrix `B`, and compute the result into `dX`.
  *  2. Gradient of Weights (dW): dWi = XT ⋅ dY: Pass `X^T` as matrix `A` (or use cuBLAS `transpose` option), `dY` as matrix `B`, and compute the result into `dW`.
  *  3. Gradient of Bias (db): db=sum(dY)
  ******************************************************************************************************************************************************/
int gemm_bgradb_lt(
                cublasOperation_t  trans_a,
                cublasOperation_t  trans_b,
                const float        *alpha,
                const float        *beta,
                at::Tensor         A,
                at::Tensor         B,
		at::Tensor         bias,
                at::Tensor         C,
                bool               use_bias)
{
    hipDataType             dataType, desc_dataType;
    hipblasComputeType_t    computeType, desc_computeType;

    if (A.scalar_type() == c10::ScalarType::BFloat16) {
           dataType         = HIP_R_16F;           computeType      = HIPBLAS_COMPUTE_32F;
           desc_dataType    = HIP_R_32F;           desc_computeType = HIPBLAS_COMPUTE_32F;

    }
    if (A.scalar_type() == at::ScalarType::Half) {
           dataType         = HIP_R_16F;           computeType      = HIPBLAS_COMPUTE_32F;
           desc_dataType    = HIP_R_32F;           desc_computeType = HIPBLAS_COMPUTE_32F;
    }

    if (A.scalar_type() == at::ScalarType::Float) {
           dataType         = HIP_R_32F;           computeType      = HIPBLAS_COMPUTE_32F;
           desc_dataType    = HIP_R_32F;           desc_computeType = HIPBLAS_COMPUTE_32F;
    }
    if (A.scalar_type() == at::ScalarType::Double) {
           dataType         = HIP_R_64F;           computeType      = HIPBLAS_COMPUTE_64F;
           desc_dataType    = HIP_R_64F;           desc_computeType = HIPBLAS_COMPUTE_64F;
    }

    const void * d_a    = static_cast<const void*>(A.data_ptr());
    const void * d_b    = static_cast<const void*>(B.data_ptr());
          void * d_c    = static_cast<void *>(C.data_ptr());

    int m = A.size(1);
    int n = A.size(0);
    int k = B.size(1);

    cudaStream_t stream;
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cublasGetStream(handle, &stream);

    /* ============================================================================================
     *   Matmul desc
     */
    hipblasLtMatmulDesc_t matmul   = nullptr;

    /* cublasStatus_t  hipblasLtMatmulDescCreate(cublasLtMatmulDesc_t *matmulDesc,
     *                                           cublasComputeType_t   computeType,
     *                                           cudaDataType_t        scaleType);
     *
     * This function creates a matrix multiply descriptor by allocating the memory needed to hold its opaque structure.

     * matmulDesc:  Output ==> Pointer to the structure holding the matrix multiply descriptor created by this function. See cublasLtMatmulDesc_t.
     * computeType: Input  ==> Enumerant that specifies the data precision for the matrix multiply descriptor this function creates. See cublasComputeType_t.
     * scaleType:   Input  ==> Enumerant that specifies the data precision for the matrix transform descriptor this function creates. See cudaDataType.
     */
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescCreate(&matmul, desc_computeType, desc_dataType));

    /* cublasStatus_t cublasLtMatmulDescSetAttribute( cublasLtMatmulDesc_t matmulDesc,
     *                                                cublasLtMatmulDescAttributes_t attr,
     *                                                const void *buf,
     *                                                size_t sizeInBytes);
     *
     * matmulDesc:  Input ==> Pointer to the previously created structure holding the matrix multiply descriptor queried by this function. See cublasLtMatmulDesc_t.
     * attr:        Input ==> The attribute that will be set by this function. See cublasLtMatmulDescAttributes_t.
     * buf:         Input ==> The value to which the specified attribute should be set.
     * sizeInBytes: Input ==> Size of buf buffer (in bytes) for verification.
     * Ref: https://docs.nvidia.com/cuda/cublas/#cublasltmatmuldescattributes-t
    */
    /*
     * CUBLASLT_MATMUL_DESC_TRANSA/CUBLASLT_MATMUL_DESC_TRANSB: int32_t
     *
     * Specifies the type of transformation operation that should be performed on matrix A/matrix B.
     * Default value is: CUBLAS_OP_N  (i.e., non-transpose operation).
    */

    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute( matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(int32_t)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute( matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(int32_t)));


     /* ============================================================================================
     *   Configure epilogue
     */

     hipblasLtEpilogue_t   epilogue = HIPBLASLT_EPILOGUE_DEFAULT;
     auto   d_bias = static_cast<void *>(bias.data_ptr());
     if ((use_bias) && (d_bias != nullptr)) {
      /*
      CUBLASLT_EPILOGUE_BGRADB = 512    Apply Bias gradient to the input matrix B. The bias size corresponds to the number of columns of the matrix D.
                                        The reduction happens over the GEMM’s “k” dimension. Store Bias gradient in the bias buffer, see
                                        CUBLASLT_MATMUL_DESC_BIAS_POINTER of cublasLtMatmulDescAttributes_t.
      */
	    std::cout << " \nConfig epilogue\n" << std::endl;    
        epilogue = HIPBLASLT_EPILOGUE_BGRADB;

        CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute( matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE,        &epilogue,  sizeof(epilogue)));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute( matmul, HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,  &dataType,  sizeof(&dataType)));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute( matmul, HIPBLASLT_MATMUL_DESC_BIAS_POINTER,    &d_bias,     sizeof(void*)));
    }

    /* ============================================================================================
     *   Matrix layout
     */
    hipblasLtMatrixLayout_t matA= nullptr, matB= nullptr, matC= nullptr;

    /*  cublasStatus_t  cublasLtMatrixLayoutCreate( cublasLtMatrixLayout_t *matLayout,
     *                                              cudaDataType           type,
     *                                              uint64_t               rows,
     *                                              uint64_t               cols,
     *                                              int64_t                ld);
     *  This function creates a matrix layout descriptor by allocating the memory needed to hold its opaque structure.
     *
     *  matLayout:  Output ==> Pointer to the structure holding the matrix layout descriptor created by this function. See cublasLtMatrixLayout_t.
     *  type:       Input  ==> Enumerant that specifies the data precision for the matrix layout descriptor this function creates. See cudaDataType.
     *  rows, cols: Input  ==> Number of rows and columns of the matrix.
     *  ld:         Input  ==> The leading dimension of the matrix. In column major layout, this is the number of elements to jump to reach the next column.
     *                         Thus ld >= m (number of rows).
    */

    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matA, dataType, k, m, k));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matB, dataType, m, n, m));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matC, dataType, k, n, k));

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
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulAlgoGetHeuristic(handle, matmul, matA, matB, matC, matC, pref, request_solutions, heuristicResult, &returnedAlgoCount));

    if(returnedAlgoCount == 0)  { std::cerr << "No valid solution found!" << std::endl; return 1;  }

    for(int i = 0; i < returnedAlgoCount; i++) { workspace_size = max(workspace_size, heuristicResult[i].workspaceSize); }

    hipMalloc(&workspace, workspace_size);
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace, sizeof(workspace_size)));

    /* ============================================================================================
     * Matmul
     *    lightHandle:                     Input  ==> Pointer to the allocated cuBLASLt handle for the cuBLASLt context. See cublasLtHandle_t.
     *    computeDesc:                     Input  ==> Handle to a previously created matrix multiplication descriptor of type cublasLtMatmulDesc_t.
     *    alpha, beta: Device/host memory: Input  ==> Pointers to the scalars used in the multiplication.
     *    A, B, and C: Device memory:      Input  ==> Pointers to the GPU memory associated with the corresponding descriptors Adesc, Bdesc and Cdesc.
     *    Adesc, Bdesc and Cdesc :         Input  ==> Handles to the previous created descriptors of the type cublasLtMatrixLayout_t.
     *    D:           Device memory       Output ==> Pointer to the GPU memory associated with the descriptor Ddesc.
     *    Ddesc:                           Input  ==> Handle to the previous created descriptor of the type cublasLtMatrixLayout_t.
     *    algo:                            Input  ==> Handle for matrix multiplication algorithm to be used. See cublasLtMatmulAlgo_t. When NULL,
     *                                                an implicit heuritics query with default search preferences will be performed to determine actual algorithm to use.
     *    workspace: Device memory:               ==> Pointer to the workspace buffer allocated in the GPU memory. Must be 256B aligned (i.e. lowest 8 bits of address must be 0).
     *    workspaceSizeInBytes:            Input  ==> Size of the workspace.
     *    stream:      Host memory         Input  ==> The CUDA stream where all the GPU work will be submitted.
    */

    // std::cout << "\ngemm_bias_lt\n" << "Tensor-A:\n" << A << "\nTensor-B:\n" << B << "\nTensor-C:\n" << C << "\nTensor-Bias:\n" << bias << std::endl;

    CHECK_HIPBLASLT_ERROR(hipblasLtMatmul(handle, matmul, alpha, d_b, matA, d_a, matB, beta,
                                          static_cast<const void*>(d_c), matC, d_c, matC, &heuristicResult[0].algo, workspace, workspace_size, stream));


    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matA));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matB));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matC));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescDestroy(matmul));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceDestroy(pref));

    return 0;

}


/********************************************************************************************************************************************************
  * gemm_bias_gelu_lt
  * 
  *
  *
  ******************************************************************************************************************************************************/
int gemm_bias_gelu_lt(
		cublasLtHandle_t     handle, 
		cublasOperation_t    trans_a, 
		cublasOperation_t    trans_b, 
		int                  m, int n, int k, 
		const float          *alpha,
		const float          *beta,
                at::Tensor           A,  
		at::Tensor           B, 
		at::Tensor           C,
	        at::Tensor           gelu_in,
	        at::Tensor           bias,  	
		void                 *d_workspace,  
		size_t               max_workspace_size, 
		cudaStream_t         stream,
                bool                 use_bias)
{
   hipDataType             dataType, desc_dataType;
   hipblasComputeType_t    computeType, desc_computeType;

   // std::cout << "\nTensor-A: " << A << "\nTensor-B: " << B << "\nTensor-C: " << C << "\nTensor-Bias: " << bias << std::endl;


   if (A.scalar_type() == c10::ScalarType::BFloat16) {
           dataType         = HIP_R_16F;           computeType      = HIPBLAS_COMPUTE_32F;
           desc_dataType    = HIP_R_32F;           desc_computeType = HIPBLAS_COMPUTE_32F;

   }
   if (A.scalar_type() == at::ScalarType::Half) {
           dataType         = HIP_R_16F;           computeType      = HIPBLAS_COMPUTE_32F;
           desc_dataType    = HIP_R_32F;           desc_computeType = HIPBLAS_COMPUTE_32F;
   }

   if (A.scalar_type() == at::ScalarType::Float) {
           dataType         = HIP_R_32F;           computeType      = HIPBLAS_COMPUTE_32F;
           desc_dataType    = HIP_R_32F;           desc_computeType = HIPBLAS_COMPUTE_32F;
   }
   if (A.scalar_type() == at::ScalarType::Double) {
           dataType         = HIP_R_64F;           computeType      = HIPBLAS_COMPUTE_32F;
           desc_dataType    = HIP_R_64F;           desc_computeType = HIPBLAS_COMPUTE_64F;
   }

    hipblasLtMatmulDesc_t       matmul;
    hipblasLtMatmulPreference_t pref;
    hipblasLtEpilogue_t         epilogue = HIPBLASLT_EPILOGUE_DEFAULT;
    hipblasLtMatrixLayout_t     matA, matB, matC;

    int       returnedAlgoCount = 0;

    hipblasLtMatmulHeuristicResult_t heuristicResult = {};

    const void * A_data       = static_cast<const void*>(A.data_ptr());
    const void * B_data       = static_cast<const void*>(B.data_ptr());
    const void * C_data       = static_cast<const void*>(C.data_ptr());
    void       * D_data       = static_cast<void*>(C.data_ptr());
    const void * bias_data    = static_cast<const void*>(bias.data_ptr());
    const void * gelu_in_data = static_cast<const void*>(gelu_in.data_ptr());

    /* cublasStatus_t
       hipblasLtMatmulDescCreate(cublasLtMatmulDesc_t *matmulDesc,
                                 cublasComputeType_t   computeType,
                                 cudaDataType_t        scaleType);

       This function creates a matrix multiply descriptor by allocating the memory needed to hold its opaque structure.

       matmulDesc:  Output ==> Pointer to the structure holding the matrix multiply descriptor created by this function. See cublasLtMatmulDesc_t.
       computeType: Input  ==> Enumerant that specifies the data precision for the matrix multiply descriptor this function creates. See cublasComputeType_t.
       scaleType:   Input  ==> Enumerant that specifies the data precision for the matrix transform descriptor this function creates. See cudaDataType.
    */
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescCreate(&matmul, desc_computeType, desc_dataType));

    /*  cublasStatus_t
        cublasLtMatrixLayoutCreate( cublasLtMatrixLayout_t *matLayout,
                                           cudaDataType type,
                                           uint64_t rows,
                                           uint64_t cols,
                                           int64_t ld);
        This function creates a matrix layout descriptor by allocating the memory needed to hold its opaque structure.

        matLayout:  Output ==> Pointer to the structure holding the matrix layout descriptor created by this function. See cublasLtMatrixLayout_t.
        type:       Input  ==> Enumerant that specifies the data precision for the matrix layout descriptor this function creates. See cudaDataType.
        rows, cols: Input  ==> Number of rows and columns of the matrix.
        ld:         Input  ==> The leading dimension of the matrix. In column major layout, this is the number of elements to jump to reach the next column.
                               Thus ld >= m (number of rows).
    */

    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matA, dataType, trans_a == CUBLAS_OP_N ? m : k, trans_a == CUBLAS_OP_N ? k : m, k));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matB, dataType, trans_b == CUBLAS_OP_N ? k : n, trans_b == CUBLAS_OP_N ? n : k, k));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matC, dataType, m, n, m));

    /* cublasStatus_t
       cublasLtMatmulDescSetAttribute( cublasLtMatmulDesc_t matmulDesc,
                                       cublasLtMatmulDescAttributes_t attr,
                                       const void *buf,
                                       size_t sizeInBytes);

       matmulDesc:  Input ==> Pointer to the previously created structure holding the matrix multiply descriptor queried by this function. See cublasLtMatmulDesc_t.
       attr:        Input ==> The attribute that will be set by this function. See cublasLtMatmulDescAttributes_t.
       buf:         Input ==> The value to which the specified attribute should be set.
       sizeInBytes: Input ==> Size of buf buffer (in bytes) for verification.
       Ref: https://docs.nvidia.com/cuda/cublas/#cublasltmatmuldescattributes-t
    */

    // CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute( matmul, HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_A_EXT, &computeType, sizeof(computeType)));
    // CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute( matmul, HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_B_EXT, &computeType, sizeof(computeType)));

    if (use_bias)  {
      // Set Desc Bias Data Type
      // CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute( matmul, HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,  &computeType,    sizeof(&computeType)));
      CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute( matmul, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_data, sizeof(bias_data)));
      epilogue = HIPBLASLT_EPILOGUE_GELU_AUX_BIAS;
    }

    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute( matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE,             &epilogue,  sizeof(epilogue)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute( matmul, HIPBLASLT_MATMUL_DESC_TRANSA,               &trans_a,   sizeof(trans_a)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute( matmul, HIPBLASLT_MATMUL_DESC_TRANSB,               &trans_b,   sizeof(trans_b)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute( matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &gelu_in_data,   sizeof(gelu_in_data)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute( matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD,      &m,         sizeof(m)));

    // Set User Preference attributes
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceCreate(&pref));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_workspace_size, sizeof(max_workspace_size)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulAlgoGetHeuristic(handle, matmul, matA, matB, matC, matC, pref, 1, &heuristicResult, &returnedAlgoCount));


    if(returnedAlgoCount == 0) { std::cerr << "No valid solution found!" << std::endl; return HIPBLAS_STATUS_EXECUTION_FAILED;  }

    CHECK_HIPBLASLT_ERROR(hipblasLtMatmul(handle, matmul, alpha, A_data, matA, B_data, matB, beta, C_data, matC, D_data, matC, NULL, d_workspace, max_workspace_size, stream));

    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matA));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matB));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matC));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescDestroy(matmul));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceDestroy(pref));

  return HIPBLAS_STATUS_SUCCESS;
}
/********************************************************************************************************************************************************
  * gemm_dgelu_bgradb_lt
  *
  *
  *
  ******************************************************************************************************************************************************/
int gemm_dgelu_bgradb_lt(
		cublasLtHandle_t  handle, 
		cublasOperation_t trans_a, 
		cublasOperation_t trans_b, 
		int               m,  int n,  int k,   
		const float       *alpha,
		const float       *beta,
                at::Tensor        A, 
		at::Tensor        B, 
		at::Tensor        C,
		at::Tensor        gelu_in,
		at::Tensor        bgrad,
		void              *d_workspace,  
		size_t            max_workspace_size, 
		cudaStream_t      stream)
{
   hipDataType             dataType, desc_dataType;
   hipblasComputeType_t    computeType, desc_computeType;

   // std::cout << "\nTensor-A: " << A << "\nTensor-B: " << B << "\nTensor-C: " << C << "\nTensor-Bias: " << bias << std::endl;


   if (A.scalar_type() == c10::ScalarType::BFloat16) {
           dataType         = HIP_R_16F;           computeType      = HIPBLAS_COMPUTE_32F;
           desc_dataType    = HIP_R_32F;           desc_computeType = HIPBLAS_COMPUTE_32F;

   }
   if (A.scalar_type() == at::ScalarType::Half) {
           dataType         = HIP_R_16F;           computeType      = HIPBLAS_COMPUTE_32F;
           desc_dataType    = HIP_R_32F;           desc_computeType = HIPBLAS_COMPUTE_32F;
   }

   if (A.scalar_type() == at::ScalarType::Float) {
           dataType         = HIP_R_32F;           computeType      = HIPBLAS_COMPUTE_32F;
           desc_dataType    = HIP_R_32F;           desc_computeType = HIPBLAS_COMPUTE_32F;
   }
   if (A.scalar_type() == at::ScalarType::Double) {
           dataType         = HIP_R_64F;           computeType      = HIPBLAS_COMPUTE_32F;
           desc_dataType    = HIP_R_64F;           desc_computeType = HIPBLAS_COMPUTE_64F;
   }

    hipblasLtMatmulDesc_t       matmul;
    hipblasLtMatmulPreference_t pref;
    hipblasLtEpilogue_t         epilogue = HIPBLASLT_EPILOGUE_DGELU_BGRAD;
    hipblasLtMatrixLayout_t     matA, matB, matC;

    int                              returnedAlgoCount = 0;
    hipblasLtMatmulHeuristicResult_t heuristicResult = {};

    const void * A_data       = static_cast<const void*>(A.data_ptr());
    const void * B_data       = static_cast<const void*>(B.data_ptr());
    const void * C_data       = static_cast<const void*>(C.data_ptr());
    void       * D_data       = static_cast<void*>(C.data_ptr());
    const void * bgrad_data   = static_cast<const void*>(bgrad.data_ptr());
    const void * gelu_in_data = static_cast<const void*>(gelu_in.data_ptr());

    /* cublasStatus_t
       hipblasLtMatmulDescCreate(cublasLtMatmulDesc_t *matmulDesc,
                                 cublasComputeType_t   computeType,
                                 cudaDataType_t        scaleType);

       This function creates a matrix multiply descriptor by allocating the memory needed to hold its opaque structure.

       matmulDesc:  Output ==> Pointer to the structure holding the matrix multiply descriptor created by this function. See cublasLtMatmulDesc_t.
       computeType: Input  ==> Enumerant that specifies the data precision for the matrix multiply descriptor this function creates. See cublasComputeType_t.
       scaleType:   Input  ==> Enumerant that specifies the data precision for the matrix transform descriptor this function creates. See cudaDataType.
    */
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescCreate(&matmul, desc_computeType, desc_dataType));

    /*  cublasStatus_t
        cublasLtMatrixLayoutCreate( cublasLtMatrixLayout_t *matLayout,
                                           cudaDataType type,
                                           uint64_t rows,
                                           uint64_t cols,
                                           int64_t ld);
        This function creates a matrix layout descriptor by allocating the memory needed to hold its opaque structure.

        matLayout:  Output ==> Pointer to the structure holding the matrix layout descriptor created by this function. See cublasLtMatrixLayout_t.
        type:       Input  ==> Enumerant that specifies the data precision for the matrix layout descriptor this function creates. See cudaDataType.
        rows, cols: Input  ==> Number of rows and columns of the matrix.
        ld:         Input  ==> The leading dimension of the matrix. In column major layout, this is the number of elements to jump to reach the next column.
                               Thus ld >= m (number of rows).
    */

    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matA, dataType, trans_a == CUBLAS_OP_N ? m : k, trans_a == CUBLAS_OP_N ? k : m, k));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matB, dataType, trans_b == CUBLAS_OP_N ? k : n, trans_b == CUBLAS_OP_N ? n : k, k));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matC, dataType, m, n, m));

    /* cublasStatus_t
       cublasLtMatmulDescSetAttribute( cublasLtMatmulDesc_t matmulDesc,
                                       cublasLtMatmulDescAttributes_t attr,
                                       const void *buf,
                                       size_t sizeInBytes);

       matmulDesc:  Input ==> Pointer to the previously created structure holding the matrix multiply descriptor queried by this function. See cublasLtMatmulDesc_t.
       attr:        Input ==> The attribute that will be set by this function. See cublasLtMatmulDescAttributes_t.
       buf:         Input ==> The value to which the specified attribute should be set.
       sizeInBytes: Input ==> Size of buf buffer (in bytes) for verification.
       Ref: https://docs.nvidia.com/cuda/cublas/#cublasltmatmuldescattributes-t
    */

    // CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute( matmul, HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_A_EXT, &computeType, sizeof(computeType)));
    // CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute( matmul, HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_B_EXT, &computeType, sizeof(computeType)));

      // Set Desc Bias Data Type
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute( matmul, HIPBLASLT_MATMUL_DESC_TRANSA,               &trans_a,     sizeof(trans_a)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute( matmul, HIPBLASLT_MATMUL_DESC_TRANSB,               &trans_b,     sizeof(trans_b)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute( matmul, HIPBLASLT_MATMUL_DESC_BIAS_POINTER,         &bgrad_data,  sizeof(bgrad_data)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute( matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &gelu_in_data,sizeof(gelu_in_data)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute( matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD,      &m,           sizeof(m)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute( matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE,             &epilogue,    sizeof(epilogue)));

    // Set User Preference attributes
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceCreate(&pref));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_workspace_size, sizeof(max_workspace_size)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulAlgoGetHeuristic(handle, matmul, matA, matB, matC, matC, pref, 1, &heuristicResult, &returnedAlgoCount));

    if(returnedAlgoCount == 0) { std::cerr << "No valid solution found!" << std::endl; return HIPBLAS_STATUS_EXECUTION_FAILED;  }

    CHECK_HIPBLASLT_ERROR(hipblasLtMatmul(handle, matmul, alpha, A_data, matA, B_data, matB, beta, C_data, matC, D_data, matC, NULL, d_workspace, max_workspace_size, stream));

    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matA));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matB));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matC));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescDestroy(matmul));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceDestroy(pref));

  return HIPBLAS_STATUS_SUCCESS;
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

    status = gemm_bias_lt( CUBLAS_OP_N, CUBLAS_OP_N, &alpha, &beta, input, weight, bias, output, true);

    return status;
}
template int linear_bias_forward_cuda <at::BFloat16>(at::Tensor  input, at::Tensor  weight, at::Tensor  bias, at::Tensor  output); 
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

    std::cout << "Input Size:"   << input.sizes()    << std::endl;
    std::cout << "Weight Size:"  << weight.sizes()   << std::endl;
    std::cout << "d_output Size" << d_output.sizes() << std::endl;

    std::cout << "d_weight Size:"  << d_weight.sizes() << std::endl;
    std::cout << "d_bias Size:"    << d_bias.sizes()   << std::endl;
    std::cout << "d_input Size"    << d_input.sizes()  << std::endl;
    // Gradient of Input (dX): dX  = dY ⋅ WT: Pass `dY`  as matrix `A`, `W`  as matrix `B`, and compute the result into `dX`.
    status = gemm_bias_lt(   CUBLAS_OP_N,  CUBLAS_OP_N,   &alpha,  &beta, d_output, weight,  d_bias, d_input,  false);

    std::cout << "\nfinding d_weight\n" << std::endl;
    // dW  = XT ⋅ dY and db=sum(dY) 
    status = gemm_bgradb_lt( CUBLAS_OP_T,  CUBLAS_OP_N,   &alpha,  &beta, input, d_output,  d_bias,  d_weight,  true);

    return status;
}

template int linear_bias_backward_cuda<at::BFloat16>(at::Tensor input, at::Tensor weight, at::Tensor d_output, at::Tensor d_weight, at::Tensor d_bias, at::Tensor d_input);
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
		int in_features, int hidden_features, int batch_size, int out_features, 
		at::Tensor output1, 
		at::Tensor output2, 
		at::Tensor gelu_in, 
		void *lt_workspace)
{
    cudaStream_t stream;
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cublasGetStream(handle, &stream);

    const float alpha      = 1.0, beta_zero  = 0.0;
    int status  = HIPBLAS_STATUS_NOT_INITIALIZED;

    status = gemm_bias_gelu_lt(
                    (cublasLtHandle_t)handle,  
		    CUBLAS_OP_T,     
		    CUBLAS_OP_N,
                    hidden_features, batch_size, in_features,        
		    &alpha,
		    &beta_zero,
                    weight1,                   
		    input, 
                    output1,
		    gelu_in,
		    bias1,
		    lt_workspace,
                    1 << 22,                   
		    stream,          
		    true);

    status = gemm_bias_lt(
		    CUBLAS_OP_T,     
		    CUBLAS_OP_N,       
		    &alpha,
	            &beta_zero,	    
		    weight2,
		    output1,
	            bias2,	    
                    output2,                    
		    true);
    return status;
}
template int linear_gelu_linear_forward_cuda<at::BFloat16>(at::Tensor input, at::Tensor weight1, at::Tensor bias1, at::Tensor weight2, 	at::Tensor bias2, 
		int in_features, int hidden_features, int batch_size, int out_features, at::Tensor output1, at::Tensor output2, at::Tensor gelu_in, void *lt_workspace);
template int linear_gelu_linear_forward_cuda<at::Half>(at::Tensor input, at::Tensor weight1, at::Tensor bias1, at::Tensor weight2, 	at::Tensor bias2, 
		int in_features, int hidden_features, int batch_size, int out_features, at::Tensor output1, at::Tensor output2, at::Tensor gelu_in, void *lt_workspace);
template int linear_gelu_linear_forward_cuda<float>(at::Tensor input, at::Tensor weight1, at::Tensor bias1, at::Tensor weight2, 	at::Tensor bias2, 
		int in_features, int hidden_features, int batch_size, int out_features, at::Tensor output1, at::Tensor output2, at::Tensor gelu_in, void *lt_workspace);
template int linear_gelu_linear_forward_cuda<double>(at::Tensor input, at::Tensor weight1, at::Tensor bias1, at::Tensor weight2, 	at::Tensor bias2, 
		int in_features, int hidden_features, int batch_size, int out_features, at::Tensor output1, at::Tensor output2, at::Tensor gelu_in, void *lt_workspace);
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
		int in_features, int batch_size, int hidden_features, int out_features, 
		at::Tensor d_weight1, 
		at::Tensor d_weight2, 
		at::Tensor d_bias1, 
		at::Tensor d_bias2, 
		at::Tensor d_input,  
		void *lt_workspace)
{
    // Get the stream from cublas handle to reuse for biasReLU kernel.
    cudaStream_t stream;
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cublasGetStream(handle, &stream);

    const float alpha      = 1.0, beta_zero  = 0.0, beta_one   = 1.0;
    int status  = HIPBLAS_STATUS_NOT_INITIALIZED;

    //wgrad for first gemm
    status = gemm_bgradb_lt(
		    CUBLAS_OP_N, 
		    CUBLAS_OP_T, 
		    &alpha,
		    &beta_zero, 
		    output1, 
                    d_output2, 
		    d_weight2,
		    d_bias2, 
		    true);

    //dgrad for second GEMM
    status = gemm_dgelu_bgradb_lt(
		    (cublasLtHandle_t)handle, 
		    CUBLAS_OP_N, 
		    CUBLAS_OP_N, 
		    hidden_features, batch_size, out_features, 
		    &alpha, 
		    &beta_zero,
		    weight2,
                    d_output2, 
		    d_output1,
		    gelu_in,
		    d_bias1, 
		    lt_workspace, 
		    1 << 22, 
		    stream);
    return status;
}

 template int  linear_gelu_linear_backward_cuda<at::BFloat16>(at::Tensor input, at::Tensor gelu_in, at::Tensor output1, at::Tensor weight1, at::Tensor weight2, 
		                                              at::Tensor d_output1, at::Tensor d_output2, int in_features, int batch_size, int hidden_features, 
							      int out_features, at::Tensor d_weight1, at::Tensor d_weight2, at::Tensor d_bias1, at::Tensor d_bias2, 
							      at::Tensor d_input, void *lt_workspace);
 template int  linear_gelu_linear_backward_cuda<at::Half>(at::Tensor input, at::Tensor gelu_in, at::Tensor output1, at::Tensor weight1, at::Tensor weight2, 
		                                              at::Tensor d_output1, at::Tensor d_output2, int in_features, int batch_size, int hidden_features, 
							      int out_features, at::Tensor d_weight1, at::Tensor d_weight2, at::Tensor d_bias1, at::Tensor d_bias2, 
							      at::Tensor d_input, void *lt_workspace);
 template int  linear_gelu_linear_backward_cuda<float>(at::Tensor input, at::Tensor gelu_in, at::Tensor output1, at::Tensor weight1, at::Tensor weight2, 
		                                              at::Tensor d_output1, at::Tensor d_output2, int in_features, int batch_size, int hidden_features, 
							      int out_features, at::Tensor d_weight1, at::Tensor d_weight2, at::Tensor d_bias1, at::Tensor d_bias2, 
							      at::Tensor d_input, void *lt_workspace);
 template int  linear_gelu_linear_backward_cuda<double>(at::Tensor input, at::Tensor gelu_in, at::Tensor output1, at::Tensor weight1, at::Tensor weight2, 
		                                              at::Tensor d_output1, at::Tensor d_output2, int in_features, int batch_size, int hidden_features, 
							      int out_features, at::Tensor d_weight1, at::Tensor d_weight2, at::Tensor d_bias1, at::Tensor d_bias2, 
							      at::Tensor d_input, void *lt_workspace);
