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

lightHandle:                     Input  ==> Pointer to the allocated cuBLASLt handle for the cuBLASLt context. See cublasLtHandle_t.
computeDesc:                     Input  ==> Handle to a previously created matrix multiplication descriptor of type cublasLtMatmulDesc_t.
alpha, beta: Device/host memory: Input  ==> Pointers to the scalars used in the multiplication.
A, B, and C: Device memory:      Input  ==> Pointers to the GPU memory associated with the corresponding descriptors Adesc, Bdesc and Cdesc.
Adesc, Bdesc and Cdesc :         Input  ==> Handles to the previous created descriptors of the type cublasLtMatrixLayout_t.
D:           Device memory       Output ==> Pointer to the GPU memory associated with the descriptor Ddesc.
Ddesc:                           Input  ==> Handle to the previous created descriptor of the type cublasLtMatrixLayout_t.
algo:                            Input  ==> Handle for matrix multiplication algorithm to be used. See cublasLtMatmulAlgo_t. When NULL, 
                                            an implicit heuritics query with default search preferences will be performed to determine actual algorithm to use.
workspace: Device memory:               ==> Pointer to the workspace buffer allocated in the GPU memory. Must be 256B aligned (i.e. lowest 8 bits of address must be 0).
workspaceSizeInBytes:            Input  ==> Size of the workspace.
stream:      Host memory         Input  ==> The CUDA stream where all the GPU work will be submitted.


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
  *
  *
  *
  ******************************************************************************************************************************************************/
int gemm_bias_lt(
		cublasLtHandle_t    handle, 
		cublasOperation_t   trans_a, 
		cublasOperation_t   trans_b, 
		int                 m,  int n,  int k,  
		const float         *alpha,
		const float         *beta,
                at::Tensor          A, 
		at::Tensor          B, 
		at::Tensor          bias,
		at::Tensor          C,
		void                *d_workspace,  
		size_t              max_workspace_size,
                cudaStream_t        stream, 
		bool                use_bias)
{

   hipDataType             dataType, desc_dataType;
   hipblasComputeType_t    computeType, desc_computeType;

   // std::cout << "\nTensor-A: " << A << "\nTensor-B: " << B << "\nTensor-C: " << C << "\nTensor-Bias: " << bias << std::endl;


   if (A.scalar_type() == c10::ScalarType::BFloat16) { 
	   dataType         = HIP_R_16F; 	   computeType      = HIPBLAS_COMPUTE_32F; 
	   desc_dataType    = HIP_R_32F;	   desc_computeType = HIPBLAS_COMPUTE_32F; 

   }
   if (A.scalar_type() == at::ScalarType::Half) { 
	   dataType         = HIP_R_16F;	   computeType      = HIPBLAS_COMPUTE_32F;
	   desc_dataType    = HIP_R_32F;	   desc_computeType = HIPBLAS_COMPUTE_32F;
   }
   
   if (A.scalar_type() == at::ScalarType::Float) { 
	   dataType         = HIP_R_32F;	   computeType      = HIPBLAS_COMPUTE_32F;
	   desc_dataType    = HIP_R_32F;	   desc_computeType = HIPBLAS_COMPUTE_32F;
   } 
   if (A.scalar_type() == at::ScalarType::Double) { 
	   dataType         = HIP_R_64F;	   computeType      = HIPBLAS_COMPUTE_64F;
	   desc_dataType    = HIP_R_64F;	   desc_computeType = HIPBLAS_COMPUTE_64F;
   } 

    hipblasLtMatmulDesc_t       matmul;
    hipblasLtMatmulPreference_t pref;
    hipblasLtEpilogue_t         epilogue = HIPBLASLT_EPILOGUE_DEFAULT;
    hipblasLtMatrixLayout_t     matA, matB, matC;
    
    int       returnedAlgoCount = 0;
    uint64_t  workspace_size    = 0;
 
    hipblasLtMatmulHeuristicResult_t heuristicResult = {}; 

    const void * A_data    = static_cast<const void*>(A.data_ptr());
    const void * B_data    = static_cast<const void*>(B.data_ptr());
    const void * C_data    = static_cast<const void*>(C.data_ptr());
    void       * D_data    = static_cast<void*>(C.data_ptr());
    const void * bias_data = static_cast<const void*>(bias.data_ptr());

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
      epilogue = HIPBLASLT_EPILOGUE_BIAS;
    }
  
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute( matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE,  &epilogue,  sizeof(epilogue))); 
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute( matmul, HIPBLASLT_MATMUL_DESC_TRANSA,    &trans_a,   sizeof(trans_a)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute( matmul, HIPBLASLT_MATMUL_DESC_TRANSB,    &trans_b,   sizeof(trans_b))); 


    // Set User Preference attributes
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceCreate(&pref));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_workspace_size, sizeof(max_workspace_size)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulAlgoGetHeuristic(handle, matmul, matA, matB, matC, matC, pref, 1, &heuristicResult, &returnedAlgoCount));

    if(returnedAlgoCount == 0) {
        std::cerr << "No valid solution found!" << std::endl;
        return HIPBLAS_STATUS_EXECUTION_FAILED;
    }


    CHECK_HIPBLASLT_ERROR(hipblasLtMatmul(handle, 
			                  matmul, 
					  alpha, 
			                  A_data, 
					  matA, 
			                  B_data, 
					  matB, 
					  beta, 
					  C_data, 
					  matC, 
					  D_data, 
					  matC, 
					  NULL, 
					  d_workspace, 
					  max_workspace_size, 
					  stream));

    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matA));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matB));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matC));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescDestroy(matmul));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceDestroy(pref));

  return HIPBLAS_STATUS_SUCCESS;
}

/********************************************************************************************************************************************************
  *
  *
  *
  *
  ******************************************************************************************************************************************************/
int gemm_bgradb_lt(
                cublasLtHandle_t   handle,
                cublasOperation_t  trans_a,
                cublasOperation_t  trans_b,
                int                m, int n, int k,
                const float        *alpha,
                const float        *beta,
                at::Tensor         A,
                at::Tensor         B,
                at::Tensor         C,
                at::Tensor         bgrad,
		void               *d_workspace,  
		size_t             max_workspace_size,
                cudaStream_t       stream,
                bool               use_bias)
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
    uint64_t  workspace_size    = 0;

    hipblasLtMatmulHeuristicResult_t heuristicResult = {};

    const void * A_data    = static_cast<const void*>(A.data_ptr());
    const void * B_data    = static_cast<const void*>(B.data_ptr());
    const void * C_data    = static_cast<const void*>(C.data_ptr());
    void       * D_data    = static_cast<void*>(C.data_ptr());
    const void * bgrad_data = static_cast<const void*>(bgrad.data_ptr());

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
      CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute( matmul, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &bgrad_data, sizeof(bgrad_data)));
      epilogue = HIPBLASLT_EPILOGUE_BGRADB;
    }

    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute( matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE,  &epilogue,  sizeof(epilogue)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute( matmul, HIPBLASLT_MATMUL_DESC_TRANSA,    &trans_a,   sizeof(trans_a)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute( matmul, HIPBLASLT_MATMUL_DESC_TRANSB,    &trans_b,   sizeof(trans_b)));


    // Set User Preference attributes
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceCreate(&pref));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_workspace_size, sizeof(max_workspace_size)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulAlgoGetHeuristic(handle, matmul, matA, matB, matC, matC, pref, 1, &heuristicResult, &returnedAlgoCount));

    if(returnedAlgoCount == 0) {
        std::cerr << "No valid solution found!" << std::endl;
        return HIPBLAS_STATUS_EXECUTION_FAILED;
    }


    CHECK_HIPBLASLT_ERROR(hipblasLtMatmul(handle,
                                          matmul,
                                          alpha,
                                          A_data,
                                          matA,
                                          B_data,
                                          matB,
                                          beta,
                                          C_data,
                                          matC,
                                          D_data,
                                          matC,
                                          NULL,
                                          d_workspace,
                                          max_workspace_size,
                                          stream));

    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matA));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matB));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matC));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescDestroy(matmul));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceDestroy(pref));

  return HIPBLAS_STATUS_SUCCESS;
}


/********************************************************************************************************************************************************
  *
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
    uint64_t  workspace_size    = 0;

    hipblasLtMatmulHeuristicResult_t heuristicResult = {};

    const void * A_data    = static_cast<const void*>(A.data_ptr());
    const void * B_data    = static_cast<const void*>(B.data_ptr());
    const void * C_data    = static_cast<const void*>(C.data_ptr());
    void       * D_data    = static_cast<void*>(C.data_ptr());
    const void * bias_data = static_cast<const void*>(bias.data_ptr());
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


    if(returnedAlgoCount == 0) {
        std::cerr << "No valid solution found!" << std::endl;
        return HIPBLAS_STATUS_EXECUTION_FAILED;
    }


    CHECK_HIPBLASLT_ERROR(hipblasLtMatmul(handle,
                                          matmul,
                                          alpha,
                                          A_data,
                                          matA,
                                          B_data,
                                          matB,
                                          beta,
                                          C_data,
                                          matC,
                                          D_data,
                                          matC,
                                          NULL,
                                          d_workspace,
                                          max_workspace_size,
                                          stream));

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

    int       returnedAlgoCount = 0;
    uint64_t  workspace_size    = 0;

    hipblasLtMatmulHeuristicResult_t heuristicResult = {};

    const void * A_data    = static_cast<const void*>(A.data_ptr());
    const void * B_data    = static_cast<const void*>(B.data_ptr());
    const void * C_data    = static_cast<const void*>(C.data_ptr());
    void       * D_data    = static_cast<void*>(C.data_ptr());
    const void * bgrad_data = static_cast<const void*>(bgrad.data_ptr());
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


    if(returnedAlgoCount == 0) {
        std::cerr << "No valid solution found!" << std::endl;
        return HIPBLAS_STATUS_EXECUTION_FAILED;
    }


    CHECK_HIPBLASLT_ERROR(hipblasLtMatmul(handle,
                                          matmul,
                                          alpha,
                                          A_data,
                                          matA,
                                          B_data,
                                          matB,
                                          beta,
                                          C_data,
                                          matC,
                                          D_data,
                                          matC,
                                          NULL,
                                          d_workspace,
                                          max_workspace_size,
                                          stream));

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
	        at::Tensor  output,	
		int in_features, int batch_size, int out_features, 
		void *lt_workspace)
{
    int         status = HIPBLAS_STATUS_NOT_INITIALIZED;
    const float alpha  = 1.0, beta_zero = 0.0, beta_one  = 1.0;

    cudaStream_t stream;
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cublasGetStream(handle, &stream); // Get the stream from cublas handle to reuse for biasReLU kernel.

    std::cout << "Input dtype: "  << input.scalar_type() << std::endl;

    status = gemm_bias_lt(
                    (cublasLtHandle_t)handle, 
		    CUBLAS_OP_T, 
		    CUBLAS_OP_N, 
		    out_features, batch_size, in_features,
                    &alpha,
		    &beta_zero,
		    input,
		    weight,
		    bias, 
		    output,
		    lt_workspace, 
		    1 << 22, 
		    stream, 
		    true);
    return status;
}
template int linear_bias_forward_cuda <at::BFloat16>(at::Tensor  input, at::Tensor  weight, at::Tensor  bias, at::Tensor  output, 
		                                     int in_features, int batch_size, int out_features, void *lt_workspace);
template int linear_bias_forward_cuda <at::Half>(at::Tensor  input, at::Tensor  weight, at::Tensor  bias, at::Tensor  output, 
		                                     int in_features, int batch_size, int out_features, void *lt_workspace);
template int linear_bias_forward_cuda <float>(at::Tensor  input, at::Tensor  weight, at::Tensor  bias, at::Tensor  output, 
		                                     int in_features, int batch_size, int out_features, void *lt_workspace);
template int linear_bias_forward_cuda <double>(at::Tensor  input, at::Tensor  weight, at::Tensor  bias, at::Tensor  output, 
		                                     int in_features, int batch_size, int out_features, void *lt_workspace);

/****************************************************************************
  *
  *
  **************************************************************************/
template <typename T>
int linear_bias_backward_cuda(
		at::Tensor    input, 
		at::Tensor    weight, 
		at::Tensor    d_output,  
		int           in_features,  int batch_size, int out_features, 
		at::Tensor    d_weight,  
		at::Tensor    d_bias, 
		at::Tensor    d_input, 
		void          *lt_workspace)
{
    int status = HIPBLAS_STATUS_NOT_INITIALIZED;
    const float alpha = 1.0, beta_zero = 0.0, beta_one = 1.0;

    cudaStream_t stream;
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle(); 
    cublasGetStream(handle, &stream); // Get the stream from cublas handle to reuse for biasReLU kernel.

    printf("linear_bias_backward_cuda:1 status=%d\n",status);

    status = gemm_bgradb_lt(
                    (cublasLtHandle_t) handle, 
		    CUBLAS_OP_N, 
		    CUBLAS_OP_T, 
		    in_features, out_features, batch_size, 
		    &alpha,
		    &beta_zero,
                    input, 
		    d_output, 
		    d_weight, 
		    d_bias,
		    lt_workspace,
                    1 << 22, 
		    stream, 
		    true);

    return status;

}

template int linear_bias_backward_cuda<at::BFloat16>(at::Tensor input, at::Tensor weight, at::Tensor d_output,  int in_features,  int batch_size, int out_features, 
		                                     at::Tensor d_weight, at::Tensor d_bias, at::Tensor d_input, void *lt_workspace);
template int linear_bias_backward_cuda<at::Half>(at::Tensor input, at::Tensor weight, at::Tensor d_output,  int in_features,  int batch_size, int out_features, 
		                                     at::Tensor d_weight, at::Tensor d_bias, at::Tensor d_input, void *lt_workspace);
template int linear_bias_backward_cuda<float>(at::Tensor input, at::Tensor weight, at::Tensor d_output,  int in_features,  int batch_size, int out_features, 
		                                     at::Tensor d_weight, at::Tensor d_bias, at::Tensor d_input, void *lt_workspace);
template int linear_bias_backward_cuda<double>(at::Tensor input, at::Tensor weight, at::Tensor d_output,  int in_features,  int batch_size, int out_features, 
		                                     at::Tensor d_weight, at::Tensor d_bias, at::Tensor d_input, void *lt_workspace);
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
                    (cublasLtHandle_t)handle,   
		    CUBLAS_OP_T,     
		    CUBLAS_OP_N,       
		    out_features, batch_size, hidden_features, 
		    &alpha,
	            &beta_zero,	    
		    weight2,
		    output1,
	            bias2,	    
                    output2,                    
		    lt_workspace,      
		    1 << 22,
                    stream,                     
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
		    (cublasLtHandle_t)handle, 
		    CUBLAS_OP_N, 
		    CUBLAS_OP_T, 
		    hidden_features, out_features, batch_size, 
		    &alpha,
		    &beta_zero, 
		    output1, 
                    d_output2, 
		    d_weight2,
		    d_bias2, 
		    lt_workspace, 
		    1 << 22, 
		    stream, 
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
