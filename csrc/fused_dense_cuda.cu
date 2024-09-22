#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <torch/torch.h>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <rocblas/rocblas.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <cublasLt.h>

#define DEBUG 0

#include "type_shim.h"

#ifndef CHECK_HIPBLASLT_ERROR
#define CHECK_HIPBLASLT_ERROR(error)                                                      \
    if(error != HIPBLAS_STATUS_SUCCESS)                                                   \
    {                                                                                     \
        fprintf(stderr, "hipBLASLt error(Err=%d) at %s:%d\n", error, __FILE__, __LINE__); \
        fprintf(stderr, "\n");                                                            \
        exit(EXIT_FAILURE);                                                               \
    }
#endif


#define DISPATCH_TYPES(TYPE, NAME, ...)                                        \
  switch (TYPE) {                                                              \
  case at::ScalarType::Half: {                                                 \
    constexpr auto compute_t = CUBLAS_COMPUTE_32F;                             \
    constexpr auto compute_datatype_t = CUDA_R_32F;                            \
    constexpr auto datatype_t = CUDA_R_16F;                                    \
    using scalar_t = at::Half;                                                 \
    __VA_ARGS__();                                                             \
    break;                                                                     \
  }                                                                            \
  case at::ScalarType::BFloat16: {                                             \
    constexpr auto compute_t = CUBLAS_COMPUTE_32F;                             \
    constexpr auto compute_datatype_t = CUDA_R_32F;                            \
    constexpr auto datatype_t = CUDA_R_16BF;                                   \
    using scalar_t = at::BFloat16;                                             \
    __VA_ARGS__();                                                             \
    break;                                                                     \
  }                                                                            \
  case at::ScalarType::Float: {                                                \
    constexpr auto compute_t = CUBLAS_COMPUTE_32F;                             \
    constexpr auto compute_datatype_t = CUDA_R_32F;                            \
    constexpr auto datatype_t = CUDA_R_32F;                                    \
    using scalar_t = float;                                                    \
    __VA_ARGS__();                                                             \
    break;                                                                     \
  }                                                                            \
  case at::ScalarType::Double: {                                               \
    constexpr auto compute_t = CUBLAS_COMPUTE_64F;                             \
    constexpr auto compute_datatype_t = CUDA_R_64F;                            \
    constexpr auto datatype_t = CUDA_R_64F;                                    \
    using scalar_t = double;                                                   \
    __VA_ARGS__();                                                             \
    break;                                                                     \
  }                                                                            \
  default:                                                                     \
    AT_ERROR(#NAME, " not implemented type ");                                 \
  }


template <cublasComputeType_t ComputeType, typename TensorType, cudaDataType_t DataType>
cublasStatus_t gemm_bias(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const float *alpha,
    const TensorType *A, int lda,
    const TensorType *B, int ldb,
    const float *beta,
    TensorType *C, int ldc)
{
  return cublasGemmEx( handle,   transa, transb,     m,        n,    k, 
		       alpha,    A,      DataType,   lda,      B,    DataType,  
		       ldb,      beta,   C,          DataType, ldc,  ComputeType, 
		       CUBLAS_GEMM_DEFAULT);
}


hipDataType get_dtype (at::Tensor A)
{
    hipDataType dataType;

    if (A.scalar_type() == c10::ScalarType::BFloat16) { dataType = HIP_R_16F; }
    if (A.scalar_type() == at::ScalarType::Half)      { dataType = HIP_R_16F; }
    if (A.scalar_type() == at::ScalarType::Float)     { dataType = HIP_R_32F; }
    if (A.scalar_type() == at::ScalarType::Double)    { dataType = HIP_R_64F; }
    // The E4M3 is mainly used for the weights, and the E5M2 is for the gradient.
    if (A.scalar_type() == at::ScalarType::Float8_e5m2fnuz) { dataType = HIP_R_8F_E5M2_FNUZ; }
    if (A.scalar_type() == at::ScalarType::Float8_e4m3fnuz) { dataType = HIP_R_8F_E4M3_FNUZ; }

    return dataType;
}

/********************************************************************************************************************************************************
  *
  * D = Epilogue{  (alpha_s * (A * B) +  beta_s * C) +  bias_v } * scaleD_v
  *
  ******************************************************************************************************************************************************/
int gemm_lt(
           hipblasOperation_t   trans_a,
           hipblasOperation_t   trans_b,
           const float          *alpha,
           const float          *beta,
           at::Tensor           A,
           at::Tensor           B,
           at::Tensor           C,
           at::Tensor           bias,
           at::Tensor           gelu,
           bool                 use_bias,
	   bool                 use_grad,
	   bool                 use_gelu)
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

    const int m = trans_a == HIPBLAS_OP_T ? A.size(0) : A.size(1);
    const int k = trans_a == HIPBLAS_OP_T ? A.size(1) : A.size(0);
    const int n = trans_b == HIPBLAS_OP_T ? B.size(1) : B.size(0);

    int lda, ldb, ldd;
    if (trans_a == HIPBLAS_OP_T && trans_b == HIPBLAS_OP_N)       {  lda = A.size(1);  ldb = B.size(1);  ldd = m; // TN
    } else if (trans_a ==HIPBLAS_OP_N && trans_b == HIPBLAS_OP_N) {  lda = m;  ldb = k;  ldd = m; // NN
    } else if (trans_a ==HIPBLAS_OP_N && trans_b == HIPBLAS_OP_T) {  lda = m;  ldb = n;  ldd = m; // NT
    } else {  std::cout << "layout not allowed." << std::endl; return 0; // TT
    }

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
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute( matmulDesc, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute( matmulDesc, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)));


    /* ============================================================================================
     *   Matrix layout
     */
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matA, dtype_a, m , k, lda));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matB, dtype_b, k,  n, ldb));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matC, dtype_c, m,  n, ldd));


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

    if(returnedAlgoCount == 0)  { std::cerr << "No valid solution found!" << std::endl; return HIPBLAS_STATUS_NOT_SUPPORTED;  }

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
   

#ifdef DEBUG
    std::cout << "\nTensor-A:\n" << A << "\nTensor-B:\n" << B << "\nTensor-C:\n" << C << "\nTensor-Bias:\n" << bias << std::endl;
    std::cout << "\nSizes: A[" << A.size(0) << "," <<  A.size(1) << "]" << std::endl;
    std::cout << "\nSizes: B[" << B.size(0) << "," <<  B.size(1) << "]" << std::endl;
    std::cout << "\nSizes: C[" << C.size(0) << "," <<  C.size(1) << "]" << std::endl;
    std::cout << "\nValues:: m:" << m << ", k:" <<  k << ", n:" << n << std::endl;
    std::cout <<"lda: " << lda << "\tldb: " << ldb << "\tldd: " << ldd << "\tm: " << m << "\tk: " << k << "\tn: " << n << std::endl;
#endif


    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matA));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matB));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matC));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescDestroy(matmulDesc));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceDestroy(pref));

    return HIPBLAS_STATUS_SUCCESS;
}

/****************************************************************************
  *
  *
  **************************************************************************/
at::Tensor linear_bias_forward( at::Tensor input, at::Tensor weight, at::Tensor bias) 
{
    int         status = HIPBLAS_STATUS_NOT_INITIALIZED;
    const float alpha  = 1.0, beta = 0.0;

    at::Tensor dummy_gelu = at::empty({0}, torch::device(torch::kCUDA).dtype(input.scalar_type()));
    
   if(input.size(1)==weight.size(1)) { weight = weight.transpose(1, 0).contiguous(); } 

    if (input.size(1)!=weight.size(0)) { 
	   // at::Tensor output = at::empty({0}, torch::device(torch::kCUDA).dtype(input.scalar_type()));
	   std::cout << "Matrix AxB for the size below is not possible" << std::endl; 
           std::cout << "\nSizes: A[" << input.size(0) << "," << input.size(1) << "]" << std::endl;
           std::cout << "\nSizes: B[" << weight.size(0) << "," <<  weight.size(1) << "]" << std::endl;
           return {at::empty({0})};
    }

    auto output = at::zeros({input.size(0), weight.size(1)}, torch::device(torch::kCUDA).dtype(input.scalar_type()));

    if (bias.size(0) != weight.size(1)){
           // at::Tensor output = at::empty({0}, torch::device(torch::kCUDA).dtype(input.scalar_type()));
           std::cout << "Bias size required to " <<  weight.size(1) << " but received " << bias.size(0) << std::endl;
           return {at::empty({0})};
    }
    
    CHECK_HIPBLASLT_ERROR(gemm_lt(HIPBLAS_OP_N, HIPBLAS_OP_N, &alpha, &beta, weight, input, output, bias, dummy_gelu, true, false, false));
    return {output};

}

/****************************************************************************
 * In the backward pass, we compute the gradients of the loss with respect to input, weight, and bias.
 * The key matrix operations are:
 *  1. Gradient of Input   (dX): dX  = dY ⋅ WT: Pass `dY`  as matrix `A`, `W`  as matrix `B`, and compute the result into `dX`.
 *  2. Gradient of Weights (dW): dW  = XT ⋅ dY: Pass `X^T` as matrix `A`  `dY` as matrix `B`, and compute the result into `dW`.
 *  3. Gradient of Bias    (db): db=sum(dY)
 *  
 **************************************************************************/
std::vector<at::Tensor>  linear_bias_backward(at::Tensor input, at::Tensor weight, at::Tensor  d_output)
{
    int status = HIPBLAS_STATUS_NOT_INITIALIZED;
    const float alpha = 1.0, beta = 0.0;

    auto dummy_gelu  = at::empty({0},             torch::device(torch::kCUDA).dtype(input.scalar_type()));
    auto d_bias      = at::zeros(weight.size(0),  torch::device(torch::kCUDA).dtype(input.scalar_type()));
    auto d_weight    = at::zeros_like(weight,     torch::device(torch::kCUDA).dtype(input.scalar_type()));
    auto d_input     = at::zeros_like(input,      torch::device(torch::kCUDA).dtype(input.scalar_type()));
    // **********************************************************************************
    // dX  = dY ⋅ WT: (Weight is transposed in Python layer before sending here) 
    // **********************************************************************************
    if (d_output.size(1)!=weight.size(0)) {
           // at::Tensor output = at::empty({0}, torch::device(torch::kCUDA).dtype(input.scalar_type()));
           std::cout << "Matrix AxB for the size below is not possible" << std::endl;
           std::cout << "\nSizes: A[" << d_output.size(0) << "," << d_output.size(1) << "]" << std::endl;
           std::cout << "\nSizes: B[" << weight.size(0) << "," << weight.size(1) << "]" << std::endl;
           return {at::empty({0})};
    }


    CHECK_HIPBLASLT_ERROR(gemm_lt(CUBLAS_OP_N, CUBLAS_OP_N, &alpha, &beta, weight, d_output, d_input, d_bias, dummy_gelu, false, false, false));

    // **********************************************************************************
    // dW  = XT ⋅ dY and db=sum(dY)
    // **********************************************************************************
    input = input.transpose(1, 0).contiguous();
    if(input.size(1)==d_output.size(1)) { d_output = d_output.transpose(1, 0).contiguous(); }

    if (input.size(1)!=d_output.size(0)) {
           // at::Tensor output = at::empty({0}, torch::device(torch::kCUDA).dtype(input.scalar_type()));
           std::cout << "Matrix AxB for the size below is not possible" << std::endl;
           std::cout << "\nSizes: A[" << input.size(0) << "," << input.size(1) << "]" << std::endl;
           std::cout << "\nSizes: B[" << d_output.size(0) << "," <<  d_output.size(1) << "]" << std::endl;
           return {at::empty({0})};
    }

    CHECK_HIPBLASLT_ERROR(gemm_lt( CUBLAS_OP_N, CUBLAS_OP_N, &alpha, &beta, d_output, input, d_weight, d_bias, dummy_gelu, true, true, false));

    return {d_input, d_weight, d_bias};
}

/****************************************************************************
  *
  *
  **************************************************************************/
std::vector<at::Tensor>  linear_gelu_linear_forward(at::Tensor input,	at::Tensor weight1,  
		                                    at::Tensor bias1, 	at::Tensor weight2,  
						    at::Tensor bias2)
{
    const float alpha      = 1.0, beta_zero  = 0.0;
    int status  = HIPBLAS_STATUS_NOT_INITIALIZED;
    auto batch_size      = input.size(0),   in_features      = input.size(1);
    int  hidden_features = weight1.size(0),  out_features    = weight2.size(0);


    auto output1      = at::zeros({batch_size, hidden_features}, torch::device(torch::kCUDA).dtype(input.scalar_type()));
    auto gelu_in      = at::zeros({batch_size, hidden_features}, torch::device(torch::kCUDA).dtype(input.scalar_type()));
    auto output2      = at::zeros({batch_size, out_features},    torch::device(torch::kCUDA).dtype(input.scalar_type()));

    at::Tensor dummy_gelu      = at::empty({0}, torch::device(torch::kCUDA).dtype(input.scalar_type()));

    status = gemm_lt(CUBLAS_OP_T, CUBLAS_OP_N, &alpha, &beta_zero, weight1, input,   output1, bias1,   gelu_in,    true, false, true);
    status = gemm_lt(CUBLAS_OP_T, CUBLAS_OP_N, &alpha, &beta_zero, weight2, output1, bias2,   output2, dummy_gelu, true, false, false);

    return {output1, output2, gelu_in};
}

/****************************************************************************
  *
  *
  **************************************************************************/
std::vector<at::Tensor>  linear_gelu_linear_backward(at::Tensor input,     at::Tensor gelu_in, 
	                                                  at::Tensor output1,   at::Tensor weight1, 
		                                          at::Tensor weight2,   at::Tensor d_output2)
{
    const float alpha      = 1.0, beta_zero  = 0.0;
    auto batch_size     = input.size(0),    in_features     = input.size(1);
    int hidden_features = weight1.size(0),  out_features    = weight2.size(0);

    int status  = HIPBLAS_STATUS_NOT_INITIALIZED;
   
    auto d_weight1    = at::empty({hidden_features,  in_features},     torch::device(torch::kCUDA).dtype(input.scalar_type()));
    auto d_weight2    = at::empty({out_features,     hidden_features}, torch::device(torch::kCUDA).dtype(input.scalar_type()));
    auto d_bias1      = at::empty({hidden_features},                   torch::device(torch::kCUDA).dtype(input.scalar_type()));
    auto d_bias2      = at::empty({out_features},                      torch::device(torch::kCUDA).dtype(input.scalar_type()));
    auto d_input      = at::empty({batch_size,       in_features},     torch::device(torch::kCUDA).dtype(input.scalar_type()));
    auto d_output1    = at::empty({batch_size,       hidden_features}, torch::device(torch::kCUDA).dtype(input.scalar_type()));
     
    at::Tensor dummy_gelu      = at::empty({0}, torch::device(torch::kCUDA).dtype(input.scalar_type()));

    //wgrad for first gemm
    status = gemm_lt(CUBLAS_OP_N, CUBLAS_OP_T,  &alpha, &beta_zero,  output1,  d_output2, d_weight2, d_bias2, dummy_gelu, true, true, false);

    // hidden_features, batch_size, out_features,
    //dgrad for second GEMM
    status = gemm_lt( CUBLAS_OP_N, CUBLAS_OP_N, &alpha,  &beta_zero,  weight2, d_output2, d_output1, d_bias1, gelu_in, true, true, false);

    return {d_input, d_weight1, d_bias1, d_weight2, d_bias2};
}


