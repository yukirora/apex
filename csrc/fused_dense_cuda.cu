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

inline void _checkCublasStatus(char const *function, char const *file, long line, cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("%s[%s:%ld]: ", function, file, line);
        printf("hipBLASlt API failed with status %d\n", status);
        throw std::logic_error("hipBLASlt API failed");
    }
}

#define checkCublasStatus(status) _checkCublasStatus(__FUNCTION__, __FILE__, __LINE__, status)
/*
      |   aType    |   bType    |   cType    |     computeType     |
      | ---------- | ---------- | ---------- | ------------------- |
      | HIP_R_16F  | HIP_R_16F  | HIP_R_16F  | HIPBLAS_COMPUTE_16F |
      | HIP_R_16F  | HIP_R_16F  | HIP_R_16F  | HIPBLAS_COMPUTE_32F |
      | HIP_R_16F  | HIP_R_16F  | HIP_R_32F  | HIPBLAS_COMPUTE_32F |
      | HIP_R_16BF | HIP_R_16BF | HIP_R_16BF | HIPBLAS_COMPUTE_32F |
      | HIP_R_16BF | HIP_R_16BF | HIP_R_32F  | HIPBLAS_COMPUTE_32F |
      | HIP_R_32F  | HIP_R_32F  | HIP_R_32F  | HIPBLAS_COMPUTE_32F |
      | HIP_R_64F  | HIP_R_64F  | HIP_R_64F  | HIPBLAS_COMPUTE_64F |
      | HIP_R_8I   | HIP_R_8I   | HIP_R_32I  | HIPBLAS_COMPUTE_32I |
      | HIP_C_32F  | HIP_C_32F  | HIP_C_32F  | HIPBLAS_COMPUTE_32F |
      | HIP_C_64F  | HIP_C_64F  | HIP_C_64F  | HIPBLAS_COMPUTE_64F |

*/
cublasStatus_t gemm_bias(
                cublasHandle_t handle, cublasOperation_t transa,  cublasOperation_t transb, int m, int n, int k,
                const float* alpha, double *A, int lda, double *B, int ldb, const float* beta, double *C, int ldc) {
// HIP_R_64F  | HIP_R_64F  | HIP_R_64F  | HIPBLAS_COMPUTE_64F
  return cublasGemmEx(handle, transa, transb, m, n, k, alpha, A, CUDA_R_64F, lda, B, CUDA_R_64F, ldb,
                  beta,   C, CUDA_R_64F, ldc, CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT);
}

cublasStatus_t gemm_bias(
                cublasHandle_t handle, cublasOperation_t transa,  cublasOperation_t transb, int m, int n, int k,
                const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc) {
// HIP_R_32F  | HIP_R_32F  | HIP_R_32F  | HIPBLAS_COMPUTE_32F
  return cublasGemmEx(handle, transa, transb, m, n, k, alpha, A, CUDA_R_32F, lda, B, CUDA_R_32F, ldb,
                  beta,    C, CUDA_R_32F, ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

cublasStatus_t gemm_bias(
                cublasHandle_t handle, cublasOperation_t transa,  cublasOperation_t transb, int m, int n, int k,
                const float* alpha, at::Half * A, int lda, at::Half * B, int ldb, const float* beta, at::Half * C, int ldc) {
//  HIP_R_16F  | HIP_R_16F  | HIP_R_16F  | HIPBLAS_COMPUTE_16F
  return cublasGemmEx(handle, transa, transb, m, n, k, alpha, A, CUDA_R_16F, lda, B, CUDA_R_16F, ldb,
                  beta,    C,  CUDA_R_16F,  ldc,  CUBLAS_COMPUTE_16F,  CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

cublasStatus_t gemm_bias(
                cublasHandle_t handle, cublasOperation_t transa,  cublasOperation_t transb, int m, int n, int k,
                const float* alpha, at::BFloat16 *A, int lda, at::BFloat16 *B, int ldb, const float* beta, at::BFloat16 *C, int ldc) {
// HIP_R_16BF | HIP_R_16BF | HIP_R_16BF | HIPBLAS_COMPUTE_32F
  return cublasGemmEx(handle, transa, transb, m, n, k, alpha, A, CUDA_R_16BF, lda, B, CUDA_R_16BF, ldb,
                  beta,    C,  CUDA_R_16BF,  ldc,  CUBLAS_COMPUTE_32F,  CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 11600 || defined(USE_ROCM)
/********************************************************************************************************************************************************
  *
  *
  *
  *
  ******************************************************************************************************************************************************/

/**************************************************************************
  * gemm_bias_lt: at::BFloat16
  ************************************************************************/

int gemm_bias_lt(cublasLtHandle_t ltHandle, cublasOperation_t transa, cublasOperation_t transb, int m,  int n,  int k,  const float *alpha,
                at::BFloat16 *A, int lda, at::BFloat16 *B, int ldb, const float *beta, at::BFloat16 *C, int ldc, void *workspace,  size_t workspaceSize,
                cudaStream_t stream, bool use_bias, const void* bias)
{
  cublasLtMatmulDesc_t               operationDesc = NULL;
  cublasLtMatrixLayout_t             Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
  cublasLtMatmulPreference_t         preference = NULL;
  cublasLtMatmulHeuristicResult_t    heuristicResult = {};
  cublasLtEpilogue_t                 epilogue = CUBLASLT_EPILOGUE_DEFAULT;

  int returnedResults = 0;

  checkCublasStatus(cublasLtMatmulDescCreate  (&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_16BF, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
  checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_16BF, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
  checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16BF, m, n, ldc));

  if (use_bias)  {
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
    epilogue = CUBLASLT_EPILOGUE_BIAS;
  }

  checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
  checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
  checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transa)));

  checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference));
  checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

  // we just need the best available heuristic to try and run matmul. There is no guarantee this will work, e.g. if A
  // is badly aligned, you can request more (e.g. 32) algos and try to run them one by one until something works
  checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults));

  if (returnedResults == 0) { checkCublasStatus(CUBLAS_STATUS_NOT_SUPPORTED);  }

  checkCublasStatus(cublasLtMatmul(ltHandle, operationDesc, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, C, Cdesc, NULL, workspace, workspaceSize, stream));

  // descriptors are no longer needed as all GPU work was already enqueued
  if (preference) checkCublasStatus(cublasLtMatmulPreferenceDestroy(preference));
  if (Cdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Cdesc));
  if (Bdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));
  if (Adesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));
  if (operationDesc) checkCublasStatus(cublasLtMatmulDescDestroy(operationDesc));
}

/**************************************************************************
  * gemm_bias_lt: at::Half
  ************************************************************************/
int gemm_bias_lt(cublasLtHandle_t ltHandle, cublasOperation_t transa, cublasOperation_t transb, int m,  int n,  int k,  const float *alpha,
                at::Half *A, int lda, at::Half *B, int ldb, const float *beta, at::Half *C, int ldc, void *workspace,  size_t workspaceSize, cudaStream_t stream,
                bool use_bias, const void* bias)
{
  cublasLtMatmulDesc_t               operationDesc = NULL;
  cublasLtMatrixLayout_t             Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
  cublasLtMatmulPreference_t         preference = NULL;
  cublasLtMatmulHeuristicResult_t    heuristicResult = {};
  cublasLtEpilogue_t                 epilogue = CUBLASLT_EPILOGUE_DEFAULT;

  int returnedResults = 0;

  // create operation desciriptor; see cublasLtMatmulDescAttributes_t for details about defaults; here we just need to
  // set the transforms for A and B
  checkCublasStatus(cublasLtMatmulDescCreate  (&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_16F, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
  checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_16F, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
  checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16F, m, n, ldc));

  if (use_bias)  {
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
    epilogue = CUBLASLT_EPILOGUE_BIAS;
  }

  checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
  checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
  checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transa)));

  checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference));
  checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

  checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults));

  if (returnedResults == 0) { checkCublasStatus(CUBLAS_STATUS_NOT_SUPPORTED);  }

  checkCublasStatus(cublasLtMatmul(ltHandle, operationDesc, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, C, Cdesc, NULL, workspace, workspaceSize, stream));

  // descriptors are no longer needed as all GPU work was already enqueued
  if (preference) checkCublasStatus(cublasLtMatmulPreferenceDestroy(preference));
  if (Cdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Cdesc));
  if (Bdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));
  if (Adesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));
  if (operationDesc) checkCublasStatus(cublasLtMatmulDescDestroy(operationDesc));
}


/**************************************************************************
  * gemm_bias_lt: float
  ************************************************************************/
int gemm_bias_lt(cublasLtHandle_t ltHandle, cublasOperation_t transa, cublasOperation_t transb, int m,  int n,  int k,  const float *alpha,
                float *A, int lda, float *B, int ldb, const float *beta, float *C, int ldc, void *workspace,  size_t workspaceSize, cudaStream_t stream,
                bool use_bias, const void* bias)
{
  cublasLtMatmulDesc_t               operationDesc = NULL;
  cublasLtMatrixLayout_t             Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
  cublasLtMatmulPreference_t         preference = NULL;
  cublasLtMatmulHeuristicResult_t    heuristicResult = {};
  cublasLtEpilogue_t                 epilogue = CUBLASLT_EPILOGUE_DEFAULT;

  int returnedResults = 0;

  checkCublasStatus(cublasLtMatmulDescCreate  (&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
  checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
  checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, m, n, ldc));

  if (use_bias)  {
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
    epilogue = CUBLASLT_EPILOGUE_BIAS;
  }

  checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
  checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
  checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transa)));

  checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference));
  checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

  checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults));

  if (returnedResults == 0) { checkCublasStatus(CUBLAS_STATUS_NOT_SUPPORTED);  }

  checkCublasStatus(cublasLtMatmul(ltHandle, operationDesc, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, C, Cdesc, NULL, workspace, workspaceSize, stream));

  // descriptors are no longer needed as all GPU work was already enqueued
  if (preference) checkCublasStatus(cublasLtMatmulPreferenceDestroy(preference));
  if (Cdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Cdesc));
  if (Bdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));
  if (Adesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));
  if (operationDesc) checkCublasStatus(cublasLtMatmulDescDestroy(operationDesc));
}

/**************************************************************************
  * gemm_bias_lt: double
  ************************************************************************/
int gemm_bias_lt(cublasLtHandle_t ltHandle, cublasOperation_t transa, cublasOperation_t transb, int m,  int n,  int k,  const float *alpha,
                double *A, int lda, double *B, int ldb, const float *beta, double *C, int ldc, void *workspace,  size_t workspaceSize, cudaStream_t stream,
                bool use_bias, const void* bias)
{
  cublasLtMatmulDesc_t               operationDesc = NULL;
  cublasLtMatrixLayout_t             Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
  cublasLtMatmulPreference_t         preference = NULL;
  cublasLtMatmulHeuristicResult_t    heuristicResult = {};
  cublasLtEpilogue_t                 epilogue = CUBLASLT_EPILOGUE_DEFAULT;

  int returnedResults = 0;

  checkCublasStatus(cublasLtMatmulDescCreate  (&operationDesc, CUBLAS_COMPUTE_64F, CUDA_R_64F));
  checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_64F, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
  checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_64F, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
  checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_64F, m, n, ldc));

  if (use_bias)  {
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
    epilogue = CUBLASLT_EPILOGUE_BIAS;
  }

  checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
  checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
  checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transa)));

  checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference));
  checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

  checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults));

  if (returnedResults == 0) { checkCublasStatus(CUBLAS_STATUS_NOT_SUPPORTED);  }

  checkCublasStatus(cublasLtMatmul(ltHandle, operationDesc, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, C, Cdesc, NULL, workspace, workspaceSize, stream));

  // descriptors are no longer needed as all GPU work was already enqueued
  if (preference) checkCublasStatus(cublasLtMatmulPreferenceDestroy(preference));
  if (Cdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Cdesc));
  if (Bdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));
  if (Adesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));
  if (operationDesc) checkCublasStatus(cublasLtMatmulDescDestroy(operationDesc));
}
/********************************************************************************************************************************************************
  *
  * 
  *
  *
  ******************************************************************************************************************************************************/

/****************************************************************
  * gemm_bias_gelu_lt: at::BFloat16
  *
  ***************************************************************/
int gemm_bias_gelu_lt(cublasLtHandle_t ltHandle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha,
                at::BFloat16 *A, int lda, at::BFloat16 *B, int ldb, const float *beta, at::BFloat16 *C,  int64_t ldc, void *workspace,  size_t workspaceSize, cudaStream_t stream,
                bool use_bias,  const void* gelu_in, const void* bias)
{
  cublasLtMatmulDesc_t               operationDesc = NULL;
  cublasLtMatrixLayout_t             Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
  cublasLtMatmulPreference_t         preference = NULL;
  cublasLtMatmulHeuristicResult_t    heuristicResult = {};
  cublasLtEpilogue_t                 epilogue = HIPBLASLT_EPILOGUE_GELU_AUX;

  int returnedResults = 0;

  checkCublasStatus(cublasLtMatmulDescCreate( &operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_16BF, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
  checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_16BF, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
  checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16BF, m, n, ldc));
  
  if (use_bias) {
      checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
      epilogue = HIPBLASLT_EPILOGUE_GELU_AUX_BIAS;
  }
  
  checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
  checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
  checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transa)));
  checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &gelu_in, sizeof(gelu_in)));
  checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &ldc, sizeof(ldc)));

  checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference));
  checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

  checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults));

  if (returnedResults == 0) { checkCublasStatus(CUBLAS_STATUS_NOT_SUPPORTED); }

  checkCublasStatus(cublasLtMatmul(ltHandle, operationDesc, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, C, Cdesc, NULL, workspace, workspaceSize, stream));
  // descriptors are no longer needed as all GPU work was already enqueued
  if (preference) checkCublasStatus(cublasLtMatmulPreferenceDestroy(preference));
  if (Cdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Cdesc));
  if (Bdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));
  if (Adesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));
  if (operationDesc) checkCublasStatus(cublasLtMatmulDescDestroy(operationDesc));
}

/****************************************************************
  * gemm_bias_gelu_lt: at::Half
  *
  ***************************************************************/
int gemm_bias_gelu_lt(cublasLtHandle_t ltHandle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha,
                at::Half *A, int lda, at::Half *B, int ldb, const float *beta, at::Half *C,  int64_t ldc, void *workspace,  size_t workspaceSize, cudaStream_t stream,
                bool use_bias,  const void* gelu_in, const void* bias)
{
  cublasLtMatmulDesc_t               operationDesc = NULL;
  cublasLtMatrixLayout_t             Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
  cublasLtMatmulPreference_t         preference = NULL;
  cublasLtMatmulHeuristicResult_t    heuristicResult = {};
  cublasLtEpilogue_t                 epilogue = HIPBLASLT_EPILOGUE_GELU_AUX;

  int returnedResults = 0;

  checkCublasStatus(cublasLtMatmulDescCreate( &operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_16F, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
  checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_16F, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
  checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16F, m, n, ldc));

  if (use_bias) {
      checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
      epilogue = HIPBLASLT_EPILOGUE_GELU_AUX_BIAS;
  }
  
  checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
  checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
  checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transa)));
  checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &gelu_in, sizeof(gelu_in)));
  checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &ldc, sizeof(ldc)));


  checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference));
  checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

  checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults));

  if (returnedResults == 0) { checkCublasStatus(CUBLAS_STATUS_NOT_SUPPORTED); }

  checkCublasStatus(cublasLtMatmul(ltHandle, operationDesc, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, C, Cdesc, NULL, workspace, workspaceSize, stream));
  // descriptors are no longer needed as all GPU work was already enqueued
  if (preference) checkCublasStatus(cublasLtMatmulPreferenceDestroy(preference));
  if (Cdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Cdesc));
  if (Bdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));
  if (Adesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));
  if (operationDesc) checkCublasStatus(cublasLtMatmulDescDestroy(operationDesc));
}
/****************************************************************
  * gemm_bias_gelu_lt: float
  *
  ***************************************************************/
int gemm_bias_gelu_lt(cublasLtHandle_t ltHandle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha,
                float *A, int lda, float *B, int ldb, const float *beta, float *C,  int64_t ldc, void *workspace,  size_t workspaceSize, cudaStream_t stream,
                bool use_bias,  const void* gelu_in, const void* bias)
{
  cublasLtMatmulDesc_t               operationDesc = NULL;
  cublasLtMatrixLayout_t             Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
  cublasLtMatmulPreference_t         preference = NULL;
  cublasLtMatmulHeuristicResult_t    heuristicResult = {};
  cublasLtEpilogue_t                 epilogue = HIPBLASLT_EPILOGUE_GELU_AUX;

  int returnedResults = 0;

  checkCublasStatus(cublasLtMatmulDescCreate( &operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
  checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
  checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, m, n, ldc));

  if (use_bias) {
      checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
      epilogue = HIPBLASLT_EPILOGUE_GELU_AUX_BIAS;
  }

  checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
  checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
  checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transa)));
  checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &gelu_in, sizeof(gelu_in)));
  checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &ldc, sizeof(ldc)));


  checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference));
  checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

  checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults));

  if (returnedResults == 0) { checkCublasStatus(CUBLAS_STATUS_NOT_SUPPORTED); }

  checkCublasStatus(cublasLtMatmul(ltHandle, operationDesc, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, C, Cdesc, NULL, workspace, workspaceSize, stream));
  // descriptors are no longer needed as all GPU work was already enqueued
  if (preference) checkCublasStatus(cublasLtMatmulPreferenceDestroy(preference));
  if (Cdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Cdesc));
  if (Bdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));
  if (Adesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));
  if (operationDesc) checkCublasStatus(cublasLtMatmulDescDestroy(operationDesc));
}


/****************************************************************
  * gemm_bias_gelu_lt: double
  *
  ***************************************************************/
int gemm_bias_gelu_lt(cublasLtHandle_t ltHandle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha,
                double *A, int lda, double *B, int ldb, const float *beta, double *C, int64_t ldc, void *workspace, size_t workspaceSize, 
		cudaStream_t stream, bool use_bias,  const void* gelu_in, const void* bias)
{
  cublasLtMatmulDesc_t               operationDesc = NULL;
  cublasLtMatrixLayout_t             Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
  cublasLtMatmulPreference_t         preference = NULL;
  cublasLtMatmulHeuristicResult_t    heuristicResult = {};
  cublasLtEpilogue_t                 epilogue = HIPBLASLT_EPILOGUE_GELU_AUX;

  int returnedResults = 0;

  checkCublasStatus(cublasLtMatmulDescCreate( &operationDesc, CUBLAS_COMPUTE_64F, CUDA_R_64F));
  checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_64F, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
  checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_64F, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
  checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_64F, m, n, ldc));

  if (use_bias) {
      checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
      epilogue = HIPBLASLT_EPILOGUE_GELU_AUX_BIAS;
  }

  checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
  checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
  checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transa)));
  checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &gelu_in, sizeof(gelu_in)));
  checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &ldc, sizeof(ldc)));


  checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference));
  checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

  checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults));

  if (returnedResults == 0) { checkCublasStatus(CUBLAS_STATUS_NOT_SUPPORTED); }

  checkCublasStatus(cublasLtMatmul(ltHandle, operationDesc, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, C, Cdesc, NULL, workspace, workspaceSize, stream));
  // descriptors are no longer needed as all GPU work was already enqueued
  if (preference) checkCublasStatus(cublasLtMatmulPreferenceDestroy(preference));
  if (Cdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Cdesc));
  if (Bdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));
  if (Adesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));
  if (operationDesc) checkCublasStatus(cublasLtMatmulDescDestroy(operationDesc));
}

/********************************************************************************************************************************************************
  *
  *
  *
  *
  ******************************************************************************************************************************************************/
/****************************************************************************
  * gemm_bgradb_lt: at::BFloat16
  *
  **************************************************************************/
int gemm_bgradb_lt(cublasLtHandle_t ltHandle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha,
                at::BFloat16 *A, int lda, at::BFloat16 *B, int ldb, const float *beta, at::BFloat16 *C, int ldc, void *workspace, size_t workspaceSize, cudaStream_t stream,
                bool use_bias,  const void* bgrad)
{
  cublasLtMatmulDesc_t             operationDesc = NULL;
  cublasLtMatrixLayout_t           Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
  cublasLtMatmulPreference_t       preference = NULL;
  cublasLtMatmulHeuristicResult_t  heuristicResult = {};
  cublasLtEpilogue_t               epilogue = CUBLASLT_EPILOGUE_DEFAULT;

  int returnedResults = 0;

  checkCublasStatus(cublasLtMatmulDescCreate( &operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_16BF, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
  checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_16BF, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
  checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16BF, m, n, ldc));
 
  if (use_bias) {
    checkCublasStatus(cublasLtMatmulDescSetAttribute( operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bgrad, sizeof(bgrad)));
    epilogue = HIPBLASLT_EPILOGUE_BGRADB;
  }

  checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE,  &epilogue,  sizeof(epilogue)));
  checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
  checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transa)));

  // Create matrix descriptors. Not setting any extra attributes.
  checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference));
  checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference));
  
  checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));



  checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults));

  if (returnedResults == 0) { checkCublasStatus(CUBLAS_STATUS_NOT_SUPPORTED); }

  checkCublasStatus(cublasLtMatmul(ltHandle, operationDesc, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, C, Cdesc, NULL, workspace, workspaceSize, stream));

  // descriptors are no longer needed as all GPU work was already enqueued
  if (preference) checkCublasStatus(cublasLtMatmulPreferenceDestroy(preference));
  if (Cdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Cdesc));
  if (Bdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));
  if (Adesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));
  if (operationDesc) checkCublasStatus(cublasLtMatmulDescDestroy(operationDesc));
}

/****************************************************************************
  * gemm_bgradb_lt: at::Half
  *
  **************************************************************************/
int gemm_bgradb_lt(cublasLtHandle_t ltHandle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha,
                at::Half *A, int lda, at::Half *B, int ldb, const float *beta, at::Half *C, int ldc, void *workspace, size_t workspaceSize, cudaStream_t stream,
                bool use_bias,  const void* bgrad)
{
  cublasLtMatmulDesc_t             operationDesc = NULL;
  cublasLtMatrixLayout_t           Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
  cublasLtMatmulPreference_t       preference = NULL;
  cublasLtMatmulHeuristicResult_t  heuristicResult = {};
  cublasLtEpilogue_t               epilogue = CUBLASLT_EPILOGUE_DEFAULT;

  int returnedResults = 0;

  checkCublasStatus(cublasLtMatmulDescCreate( &operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_16F, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
  checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_16F, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
  checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16F, m, n, ldc));

  if (use_bias) {
    checkCublasStatus(cublasLtMatmulDescSetAttribute( operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bgrad, sizeof(bgrad)));
    epilogue = HIPBLASLT_EPILOGUE_BGRADB;
  }

  checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE,  &epilogue,  sizeof(epilogue)));
  checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
  checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transa)));

  // Create matrix descriptors. Not setting any extra attributes.
  checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference));
  checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference));
  
  checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));



  checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults));

  if (returnedResults == 0) { checkCublasStatus(CUBLAS_STATUS_NOT_SUPPORTED); }

  checkCublasStatus(cublasLtMatmul(ltHandle, operationDesc, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, C, Cdesc, NULL, workspace, workspaceSize, stream));

  // descriptors are no longer needed as all GPU work was already enqueued
  if (preference) checkCublasStatus(cublasLtMatmulPreferenceDestroy(preference));
  if (Cdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Cdesc));
  if (Bdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));
  if (Adesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));
  if (operationDesc) checkCublasStatus(cublasLtMatmulDescDestroy(operationDesc));
}

/****************************************************************************
  * gemm_bgradb_lt: float
  *
  **************************************************************************/
int gemm_bgradb_lt(cublasLtHandle_t ltHandle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha,
                float *A, int lda, float *B, int ldb, const float *beta, float *C, int ldc, void *workspace, size_t workspaceSize, cudaStream_t stream,
                bool use_bias,  const void* bgrad)
{
  cublasLtMatmulDesc_t             operationDesc = NULL;
  cublasLtMatrixLayout_t           Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
  cublasLtMatmulPreference_t       preference = NULL;
  cublasLtMatmulHeuristicResult_t  heuristicResult = {};
  cublasLtEpilogue_t               epilogue = CUBLASLT_EPILOGUE_DEFAULT;

  int returnedResults = 0;

  checkCublasStatus(cublasLtMatmulDescCreate( &operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
  checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
  checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, m, n, ldc));

 if (use_bias) {
    checkCublasStatus(cublasLtMatmulDescSetAttribute( operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bgrad, sizeof(bgrad)));
    epilogue = HIPBLASLT_EPILOGUE_BGRADB;
  }

  checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE,  &epilogue,  sizeof(epilogue)));
  checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
  checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transa)));

  // Create matrix descriptors. Not setting any extra attributes.
  checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference));
  checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference));
  
  checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));



  checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults));

  if (returnedResults == 0) { checkCublasStatus(CUBLAS_STATUS_NOT_SUPPORTED); }

  checkCublasStatus(cublasLtMatmul(ltHandle, operationDesc, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, C, Cdesc, NULL, workspace, workspaceSize, stream));

  // descriptors are no longer needed as all GPU work was already enqueued
  if (preference) checkCublasStatus(cublasLtMatmulPreferenceDestroy(preference));
  if (Cdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Cdesc));
  if (Bdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));
  if (Adesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));
  if (operationDesc) checkCublasStatus(cublasLtMatmulDescDestroy(operationDesc));
}


/****************************************************************************
  * gemm_bgradb_lt: double
  *
  **************************************************************************/
int gemm_bgradb_lt(cublasLtHandle_t ltHandle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha,
                double *A, int lda, double *B, int ldb, const float *beta, double *C, int ldc, void *workspace, size_t workspaceSize, cudaStream_t stream,
                bool use_bias,  const void* bgrad)
{
  cublasLtMatmulDesc_t             operationDesc = NULL;
  cublasLtMatrixLayout_t           Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
  cublasLtMatmulPreference_t       preference = NULL;
  cublasLtMatmulHeuristicResult_t  heuristicResult = {};
  cublasLtEpilogue_t               epilogue = CUBLASLT_EPILOGUE_DEFAULT;

  int returnedResults = 0;

  checkCublasStatus(cublasLtMatmulDescCreate( &operationDesc, CUBLAS_COMPUTE_64F, CUDA_R_64F));
  checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_64F, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
  checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_64F, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
  checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_64F, m, n, ldc));
 if (use_bias) {
    checkCublasStatus(cublasLtMatmulDescSetAttribute( operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bgrad, sizeof(bgrad)));
    epilogue = HIPBLASLT_EPILOGUE_BGRADB;
  }

  checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE,  &epilogue,  sizeof(epilogue)));
  checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
  checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transa)));

  // Create matrix descriptors. Not setting any extra attributes.
  checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference));
  checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference));
  
  checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

  checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults));

  if (returnedResults == 0) { checkCublasStatus(CUBLAS_STATUS_NOT_SUPPORTED); }

  checkCublasStatus(cublasLtMatmul(ltHandle, operationDesc, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, C, Cdesc, NULL, workspace, workspaceSize, stream));

  // descriptors are no longer needed as all GPU work was already enqueued
  if (preference) checkCublasStatus(cublasLtMatmulPreferenceDestroy(preference));
  if (Cdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Cdesc));
  if (Bdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));
  if (Adesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));
  if (operationDesc) checkCublasStatus(cublasLtMatmulDescDestroy(operationDesc));
}

/********************************************************************************************************************************************************
  *
  *
  *
  *
  ******************************************************************************************************************************************************/


/****************************************************************************
  * gemm_dgelu_bgradb_lt: at::BFloat16
  *
  **************************************************************************/

int gemm_dgelu_bgradb_lt(cublasLtHandle_t ltHandle, cublasOperation_t transa, cublasOperation_t transb, int m,  int n,  int k,   const float *alpha,
                at::BFloat16 *A, int lda, at::BFloat16 *B, int ldb, const float *beta, at::BFloat16 *C, int64_t ldc, void *workspace, size_t workspaceSize, 
		cudaStream_t stream, const void *gelu_in, const void *bgrad)
{

  cublasLtMatmulDesc_t             operationDesc = NULL;
  cublasLtMatrixLayout_t           Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
  cublasLtMatmulPreference_t       preference = NULL;
  cublasLtMatmulHeuristicResult_t  heuristicResult  = {};
  cublasLtEpilogue_t               epilogue = HIPBLASLT_EPILOGUE_DGELU_BGRAD;

  int                              returnedResults  = 0;

  checkCublasStatus(cublasLtMatmulDescCreate( &operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_16BF, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
  checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_16BF, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
  checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16BF, m, n, ldc));

  checkCublasStatus(cublasLtMatmulDescSetAttribute( operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa,       sizeof(transa)));
  checkCublasStatus(cublasLtMatmulDescSetAttribute( operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb,       sizeof(transa)));
  checkCublasStatus(cublasLtMatmulDescSetAttribute( operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bgrad,  sizeof(bgrad)));
  checkCublasStatus(cublasLtMatmulDescSetAttribute( operationDesc, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &gelu_in,  sizeof(gelu_in)));
  checkCublasStatus(cublasLtMatmulDescSetAttribute( operationDesc, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD,      &ldc,      sizeof(ldc)));
  checkCublasStatus(cublasLtMatmulDescSetAttribute( operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE,             &epilogue, sizeof(epilogue)));

  checkCublasStatus(cublasLtMatmulPreferenceCreate( &preference));
  checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

  checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults));

  if (returnedResults == 0) { checkCublasStatus(CUBLAS_STATUS_NOT_SUPPORTED); }

  checkCublasStatus(cublasLtMatmul(ltHandle, operationDesc, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, C, Cdesc, NULL, workspace, workspaceSize, stream));
  // descriptors are no longer needed as all GPU work was already enqueued
  if (preference) checkCublasStatus(cublasLtMatmulPreferenceDestroy(preference));
  if (Cdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Cdesc));
  if (Bdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));
  if (Adesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));
  if (operationDesc) checkCublasStatus(cublasLtMatmulDescDestroy(operationDesc));
}

/****************************************************************************
  * gemm_dgelu_bgradb_lt: at::Half
  *
  **************************************************************************/
int gemm_dgelu_bgradb_lt(cublasLtHandle_t ltHandle, cublasOperation_t transa, cublasOperation_t transb, int m,  int n,  int k,   const float *alpha,
                at::Half *A, int lda, at::Half *B, int ldb, const float *beta, at::Half *C, int64_t ldc, void *workspace, size_t workspaceSize,
                cudaStream_t stream, const void *gelu_in, const void *bgrad)
{

  cublasLtMatmulDesc_t             operationDesc = NULL;
  cublasLtMatrixLayout_t           Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
  cublasLtMatmulPreference_t       preference = NULL;
  cublasLtMatmulHeuristicResult_t  heuristicResult  = {};
  cublasLtEpilogue_t               epilogue = HIPBLASLT_EPILOGUE_DGELU_BGRAD;

  int                              returnedResults  = 0;

  checkCublasStatus(cublasLtMatmulDescCreate( &operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_16F, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
  checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_16F, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
  checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16F, m, n, ldc));

  checkCublasStatus(cublasLtMatmulDescSetAttribute( operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa,       sizeof(transa)));
  checkCublasStatus(cublasLtMatmulDescSetAttribute( operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb,       sizeof(transa)));
  checkCublasStatus(cublasLtMatmulDescSetAttribute( operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bgrad,  sizeof(bgrad)));
  checkCublasStatus(cublasLtMatmulDescSetAttribute( operationDesc, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &gelu_in,  sizeof(gelu_in)));
  checkCublasStatus(cublasLtMatmulDescSetAttribute( operationDesc, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD,      &ldc,      sizeof(ldc)));
  checkCublasStatus(cublasLtMatmulDescSetAttribute( operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE,             &epilogue, sizeof(epilogue)));

  checkCublasStatus(cublasLtMatmulPreferenceCreate( &preference));
  checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

  checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults));

  if (returnedResults == 0) { checkCublasStatus(CUBLAS_STATUS_NOT_SUPPORTED); }

  checkCublasStatus(cublasLtMatmul(ltHandle, operationDesc, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, C, Cdesc, NULL, workspace, workspaceSize, stream));
  // descriptors are no longer needed as all GPU work was already enqueued
  if (preference) checkCublasStatus(cublasLtMatmulPreferenceDestroy(preference));
  if (Cdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Cdesc));
  if (Bdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));
  if (Adesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));
  if (operationDesc) checkCublasStatus(cublasLtMatmulDescDestroy(operationDesc));
}

/****************************************************************************
  * gemm_dgelu_bgradb_lt: float
  *
  **************************************************************************/
int gemm_dgelu_bgradb_lt(cublasLtHandle_t ltHandle, cublasOperation_t transa, cublasOperation_t transb, int m,  int n,  int k,   const float *alpha,
                float *A, int lda, float *B, int ldb, const float *beta, float *C, int64_t ldc, void *workspace, size_t workspaceSize,
                cudaStream_t stream, const void *gelu_in, const void *bgrad)
{

  cublasLtMatmulDesc_t             operationDesc = NULL;
  cublasLtMatrixLayout_t           Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
  cublasLtMatmulPreference_t       preference = NULL;
  cublasLtMatmulHeuristicResult_t  heuristicResult  = {};
  cublasLtEpilogue_t               epilogue = HIPBLASLT_EPILOGUE_DGELU_BGRAD;

  int                              returnedResults  = 0;

  checkCublasStatus(cublasLtMatmulDescCreate( &operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
  checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
  checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, m, n, ldc));

  checkCublasStatus(cublasLtMatmulDescSetAttribute( operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa,       sizeof(transa)));
  checkCublasStatus(cublasLtMatmulDescSetAttribute( operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb,       sizeof(transa)));
  checkCublasStatus(cublasLtMatmulDescSetAttribute( operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bgrad,  sizeof(bgrad)));
  checkCublasStatus(cublasLtMatmulDescSetAttribute( operationDesc, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &gelu_in,  sizeof(gelu_in)));
  checkCublasStatus(cublasLtMatmulDescSetAttribute( operationDesc, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD,      &ldc,      sizeof(ldc)));
  checkCublasStatus(cublasLtMatmulDescSetAttribute( operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE,             &epilogue, sizeof(epilogue)));

  checkCublasStatus(cublasLtMatmulPreferenceCreate( &preference));
  checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

  checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults));

  if (returnedResults == 0) { checkCublasStatus(CUBLAS_STATUS_NOT_SUPPORTED); }

  checkCublasStatus(cublasLtMatmul(ltHandle, operationDesc, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, C, Cdesc, NULL, workspace, workspaceSize, stream));
  // descriptors are no longer needed as all GPU work was already enqueued
  if (preference) checkCublasStatus(cublasLtMatmulPreferenceDestroy(preference));
  if (Cdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Cdesc));
  if (Bdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));
  if (Adesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));
  if (operationDesc) checkCublasStatus(cublasLtMatmulDescDestroy(operationDesc));
}
  
/****************************************************************************
  * gemm_dgelu_bgradb_lt: double
  *
  **************************************************************************/
int gemm_dgelu_bgradb_lt(cublasLtHandle_t ltHandle, cublasOperation_t transa, cublasOperation_t transb, int m,  int n,  int k,   const float *alpha,
                double *A, int lda, double *B, int ldb, const float *beta, double *C, int64_t ldc, void *workspace, size_t workspaceSize,
                cudaStream_t stream, const void *gelu_in, const void *bgrad)
{

  cublasLtMatmulDesc_t             operationDesc = NULL;
  cublasLtMatrixLayout_t           Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
  cublasLtMatmulPreference_t       preference = NULL;
  cublasLtMatmulHeuristicResult_t  heuristicResult  = {};
  cublasLtEpilogue_t               epilogue = HIPBLASLT_EPILOGUE_DGELU_BGRAD;

  int                              returnedResults  = 0;


  checkCublasStatus(cublasLtMatmulDescCreate( &operationDesc, CUBLAS_COMPUTE_64F, CUDA_R_64F));
  checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_64F, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
  checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_64F, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
  checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_64F, m, n, ldc));

  checkCublasStatus(cublasLtMatmulDescSetAttribute( operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa,       sizeof(transa)));
  checkCublasStatus(cublasLtMatmulDescSetAttribute( operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb,       sizeof(transa)));
  checkCublasStatus(cublasLtMatmulDescSetAttribute( operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bgrad,  sizeof(bgrad)));
  checkCublasStatus(cublasLtMatmulDescSetAttribute( operationDesc, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &gelu_in,  sizeof(gelu_in)));
  checkCublasStatus(cublasLtMatmulDescSetAttribute( operationDesc, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD,      &ldc,      sizeof(ldc)));
  checkCublasStatus(cublasLtMatmulDescSetAttribute( operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE,             &epilogue, sizeof(epilogue)));

  checkCublasStatus(cublasLtMatmulPreferenceCreate( &preference));
  checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

  checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults));

  if (returnedResults == 0) { checkCublasStatus(CUBLAS_STATUS_NOT_SUPPORTED); }

  checkCublasStatus(cublasLtMatmul(ltHandle, operationDesc, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, C, Cdesc, NULL, workspace, workspaceSize, stream));
  // descriptors are no longer needed as all GPU work was already enqueued
  if (preference) checkCublasStatus(cublasLtMatmulPreferenceDestroy(preference));
  if (Cdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Cdesc));
  if (Bdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));
  if (Adesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));
  if (operationDesc) checkCublasStatus(cublasLtMatmulDescDestroy(operationDesc));
}


#endif
/****************************************************************************
  *
  *
  **************************************************************************/
template <typename T>
int linear_bias_forward_cuda(at::Tensor input, T *weight, at::Tensor bias, int in_features, int batch_size, int out_features, at::Tensor output, void *lt_workspace)
{
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

    // Get the stream from cublas handle to reuse for biasReLU kernel.
    cudaStream_t stream;
    cublasGetStream(handle, &stream);
    int status = HIPBLAS_STATUS_NOT_INITIALIZED;

    const float alpha     = 1.0, beta_zero = 0.0, beta_one  = 1.0;

#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 11600 || defined(USE_ROCM)
     status = gemm_bias_lt(
                    (cublasLtHandle_t)handle, CUBLAS_OP_T, CUBLAS_OP_N, out_features, batch_size, in_features,
                    &alpha, weight, in_features, input.data_ptr<T>(), in_features, &beta_zero,  output.data_ptr<T>(),
                    out_features, lt_workspace, 1 << 22, stream, true, static_cast<const void*>(bias.data_ptr<T>()));

#endif
    output.copy_(bias);
    if(status!=HIPBLAS_STATUS_SUCCESS)
    {
            status = gemm_bias(
                            handle,                CUBLAS_OP_T,           CUBLAS_OP_N,
                            out_features,          batch_size,            in_features,   &alpha,          weight,
                            in_features,           input.data_ptr<T>(),   in_features,   &beta_one,
                            output.data_ptr<T>(),  out_features);
    }

    return status;
}
template int linear_bias_forward_cuda<at::BFloat16>(at::Tensor, at::BFloat16 *,  at::Tensor, int, int, int, at::Tensor, void *);
template int linear_bias_forward_cuda<at::Half>    (at::Tensor, at::Half *,      at::Tensor, int, int, int, at::Tensor, void *);
template int linear_bias_forward_cuda<float>       (at::Tensor, float *,         at::Tensor, int, int, int, at::Tensor, void *);
template int linear_bias_forward_cuda<double>      (at::Tensor, double *,        at::Tensor, int, int, int, at::Tensor, void *);

/****************************************************************************
  *
  *
  **************************************************************************/
template <typename T>
int linear_bias_backward_cuda(T *input, T *weight, T *d_output,  int in_features,  int batch_size, int out_features, T *d_weight,  T *d_bias, T *d_input, void *lt_workspace)
{
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

    // Get the stream from cublas handle to reuse for biasReLU kernel.
    cudaStream_t stream;
    cublasGetStream(handle, &stream);

    const float alpha = 1.0, beta_zero = 0.0, beta_one = 1.0;
    int status = HIPBLAS_STATUS_NOT_INITIALIZED;
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 11600 || defined(USE_ROCM)
    status = gemm_bgradb_lt(
                    //cublasLtHandle_t ltHandle    transa          transb
                    (cublasLtHandle_t)handle,      CUBLAS_OP_N,    CUBLAS_OP_T,
                    // m                           n               k                alpha
                    in_features,                   out_features,   batch_size,      &alpha,
                    // A                            lda            B                ldb
                    input,                         in_features,    d_output,        out_features,
                    // beta                        C               ldc              void *workspace
                    &beta_zero,                    d_weight,       in_features,     lt_workspace,
                    // size_t workspaceSize  cudaStream_t stream   bool use_bias,   const void* bias
                    1 << 22,                       stream,         true,            static_cast<const void*>(d_bias));
#endif

    if (status != HIPBLAS_STATUS_SUCCESS)
    {
            status = gemm_bias(
                            handle,         CUBLAS_OP_N,    CUBLAS_OP_T,
                            in_features,    out_features,   batch_size,     &alpha,     input,
                            in_features,    d_output,       out_features,   &beta_zero, d_weight,
                            in_features);
    }

    status = gemm_bias(
                    handle,         CUBLAS_OP_N,   CUBLAS_OP_N,
                    in_features,    batch_size,    out_features,    &alpha,       weight,
                    in_features,    d_output,      out_features,    &beta_zero,   d_input,
                    in_features);
    return status;

}
template int linear_bias_backward_cuda<at::BFloat16>(at::BFloat16 *, at::BFloat16 *, at::BFloat16 *, int, int, int,  at::BFloat16 *, at::BFloat16 *,   at::BFloat16 *, void *);
template int linear_bias_backward_cuda<at::Half>    (at::Half *,     at::Half *,     at::Half *,     int, int, int,  at::Half *,     at::Half *,       at::Half *,     void *);
template int linear_bias_backward_cuda<float>       (float *,        float *,        float *,        int, int, int,  float *,        float *,          float *,        void *);
template int linear_bias_backward_cuda<double>      (double *,       double *,       double *,       int, int, int,  double *,       double *,         double *,       void *);
/****************************************************************************
  *
  *
  **************************************************************************/
template <typename T>
int linear_gelu_linear_forward_cuda(T *input, T *weight1, T *bias1, T *weight2, T *bias2, int in_features, int hidden_features, int batch_size,
                int out_features, T *output1, T *output2, T *gelu_in, void *lt_workspace)
{
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

    // Get the stream from cublas handle to reuse for biasReLU kernel.
    cudaStream_t stream;
    cublasGetStream(handle, &stream);

    const float alpha      = 1.0, beta_zero  = 0.0;
    int         status     = HIPBLAS_STATUS_NOT_INITIALIZED;


#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 11600 || defined(USE_ROCM)
    status = gemm_bias_gelu_lt(
                    (cublasLtHandle_t)handle,  CUBLAS_OP_T,     CUBLAS_OP_N,
                    hidden_features,           batch_size,      in_features,        &alpha,
                    weight1,                   in_features,     input,              in_features,
                    &beta_zero,                output1,         hidden_features,    lt_workspace,
                    1 << 22,                   stream,          true,               static_cast<const void*>(gelu_in),
                    static_cast<const void*>(bias1));
    status = gemm_bias_lt(
                    (cublasLtHandle_t)handle,   CUBLAS_OP_T,     CUBLAS_OP_N,       out_features,
                    batch_size,                 hidden_features, &alpha,            weight2,
                    hidden_features,            output1,         hidden_features,   &beta_zero,
                    output2,                    out_features,    lt_workspace,      1 << 22,
                    stream,                     true,            static_cast<const void*>(bias2));
#endif
    return status;
}

template int linear_gelu_linear_forward_cuda<at::BFloat16>(at::BFloat16 *, at::BFloat16 *, at::BFloat16 *, at::BFloat16 *, at::BFloat16 *,
                                                           int, int, int, int, at::BFloat16 *,  at::BFloat16 *,  at::BFloat16 *, void *);
template int linear_gelu_linear_forward_cuda<at::Half>    (at::Half *,     at::Half *,     at::Half *,      at::Half *,      at::Half *,
                                                           int, int, int, int, at::Half *,      at::Half *,      at::Half *,     void *);
template int linear_gelu_linear_forward_cuda<float>       (float *,        float *,        float *,         float *,          float *,
                                                           int, int, int, int, float *,         float *,         float *,        void *);
template int linear_gelu_linear_forward_cuda<double>      (double *,       double *,       double *,        double *,         double *,
                                                           int, int, int, int,  double *,        double *,        double *,      void *);

/****************************************************************************
  *
  *
  **************************************************************************/
template <typename T>
int linear_gelu_linear_backward_cuda(T *input, T *gelu_in, T *output1, T *weight1, T *weight2, T *d_output1, T *d_output2, int in_features, int batch_size,
                int hidden_features, int out_features, T *d_weight1, T *d_weight2, T *d_bias1, T *d_bias2, T *d_input,  void *lt_workspace)
{
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

    // Get the stream from cublas handle to reuse for biasReLU kernel.
    cudaStream_t stream;
    cublasGetStream(handle, &stream);

    const float alpha      = 1.0, beta_zero  = 0.0, beta_one   = 1.0;
    int         status     = HIPBLAS_STATUS_NOT_INITIALIZED;

#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 11000 || defined(USE_ROCM)
    //wgrad for first gemm
    status = gemm_bgradb_lt((cublasLtHandle_t)handle, CUBLAS_OP_N, CUBLAS_OP_T, hidden_features, out_features, batch_size, &alpha, output1, hidden_features,
                            d_output2, out_features, &beta_zero, d_weight2, hidden_features, lt_workspace, 1 << 22, stream, true, static_cast<const void*>(d_bias2));

    //dgrad for second GEMM
    status = gemm_dgelu_bgradb_lt((cublasLtHandle_t)handle, CUBLAS_OP_N, CUBLAS_OP_N, hidden_features, batch_size, out_features, &alpha, weight2, hidden_features,
                    d_output2, out_features, &beta_zero, d_output1, hidden_features, lt_workspace, 1 << 22, stream, static_cast<const void*>(gelu_in),
                    static_cast<const void*>(d_bias1));
#else
    //wgrad for the first GEMM
    status = gemm_bias( handle, CUBLAS_OP_N, CUBLAS_OP_T, in_features, hidden_features, batch_size, &alpha, input, in_features, d_output1, hidden_features, &beta_zero,
                        d_weight1, in_features);

    //dgrad for the first GEMM
    status = gemm_bias(handle, CUBLAS_OP_N, CUBLAS_OP_N, in_features, batch_size, hidden_features, &alpha, weight1, in_features, d_output1, hidden_features, &beta_zero,
                       d_input, in_features);
#endif
    return status;

}

template int linear_gelu_linear_backward_cuda<at::BFloat16>(at::BFloat16 *, at::BFloat16 *, at::BFloat16 *, at::BFloat16 *, at::BFloat16 *, at::BFloat16 *, at::BFloat16 *,
                                                            int, int, int, int, at::BFloat16 *, at::BFloat16 *, at::BFloat16 *,  at::BFloat16 *, at::BFloat16 *,   void *);
template int linear_gelu_linear_backward_cuda<at::Half>    (at::Half *,     at::Half *,     at::Half *,      at::Half *,     at::Half *,     at::Half *,     at::Half *,
                                                            int, int, int, int, at::Half *,      at::Half *,    at::Half *,      at::Half *,      at::Half *,   void *);
template int linear_gelu_linear_backward_cuda<float>       (float *,        float *,        float *,          float *,        float *,       float *,         float *,
                                                            int, int, int, int, float *,         float *,       float *,         float *,          float *,   void *);
template int linear_gelu_linear_backward_cuda<double>      (double *,       double *,       double *,         double *,       double *,       double *,        double *,
                                                            int, int, int, int,  double *,        double *,     double *,         double *,         double *,  void *);

