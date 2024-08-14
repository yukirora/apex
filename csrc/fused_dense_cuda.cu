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

inline void checkCublasStatus(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS API failed with status %d\n", status);
        throw std::logic_error("cuBLAS API failed");
    }
}

/***********************************************************
* FP64 Wrapper around cublas GEMMEx
***********************************************************/
cublasStatus_t gemm_bias(
		cublasHandle_t handle, cublasOperation_t transa,  cublasOperation_t transb,    
		int m,    int n,    int k,    
		const float* alpha,  
		double* A,	int lda,    
		double* B,    	int ldb,    
		const float* beta,    
		double* C,   	int ldc) 
{
  return cublasGemmEx( 
	     	  handle, transa, transb, m, n, k, 
	     	  alpha,  A, CUDA_R_64F, lda,  
	      	          B, CUDA_R_64F, ldb, 
	     	  beta,   C, CUDA_R_64F, ldc, CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT);
}


/***********************************************************
* FP32 Wrapper around cublas GEMMEx
***********************************************************/
cublasStatus_t gemm_bias( 
	     	cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, 
	     	int m,  int n, 	int k,  
	      	const float* alpha, 
	     	float* A,  int lda,    
		float* B,  int ldb,  
	      	const float* beta, 
	     	float* C,  int ldc) 
{
  return cublasGemmEx( 
	     	  handle,  transa,  transb,  m,  n,  k,  
	      	  alpha,   A, CUDA_R_32F, lda,  
	      	           B, CUDA_R_32F, ldb,  
	      	  beta,    C, CUDA_R_32F, ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

/***********************************************************
* FP16 Wrapper around cublas GEMMEx
***********************************************************/
cublasStatus_t gemm_bias(
	    	cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, 
	     	int m,  int n, 	int k,  
	      	const float* alpha,
	    	at::Half* A, int lda,   
	       	at::Half* B, int ldb,    
		const float* beta,   
	       	at::Half* C, int ldc) 
{
  return cublasGemmEx(  
		  handle,  transa,  transb,  m,  n,  k,  
		  alpha,   A,  CUDA_R_16F,  lda,  
		           B,  CUDA_R_16F,  ldb,  
	      	  beta,    C,  CUDA_R_16F,  ldc,  CUBLAS_COMPUTE_16F,  CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}


#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 11600 || defined(USE_ROCM)
/**************************************************************************
  *  
  ************************************************************************/
template <typename T>
int gemm_bias_lt(    
		cublasLtHandle_t ltHandle, cublasOperation_t transa, cublasOperation_t transb,    
		int m,  int n,  int k,	const float *alpha, 
		T A,   int lda,    
		T B,   int ldb, const float *beta, 
		T C,   int ldc,    
		void *workspace,  size_t workspaceSize, cudaStream_t stream,    
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
  
  if ((std::is_same<T, at::Half>::value) || (std::is_same<T, float>::value)) 
  {  
	  checkCublasStatus(cublasLtMatmulDescCreate( &operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F)); 
  }

  if (std::is_same<T, double>::value)
  {  
	  checkCublasStatus(cublasLtMatmulDescCreate( &operationDesc, CUBLAS_COMPUTE_64F, CUDA_R_64F)); 
  }

  checkCublasStatus(cublasLtMatmulDescSetAttribute( operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa))); 
  checkCublasStatus(cublasLtMatmulDescSetAttribute( operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transa))); 
  
  if (use_bias)  {
    checkCublasStatus(cublasLtMatmulDescSetAttribute( operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias))); 
    epilogue = CUBLASLT_EPILOGUE_BIAS;
  } 

  checkCublasStatus(cublasLtMatmulDescSetAttribute( operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue))); 
 
  // create matrix descriptors, 
  if (std::is_same<T, at::Half>::value) 
  {
  	checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_16F, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
  	checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_16F, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
  	checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16F, m, n, ldc));
  }

  if (std::is_same<T, float>::value)
  {
  	checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
  	checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
  	checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, m, n, ldc));
  }

  if (std::is_same<T, double>::value)
  {
  	checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_64F, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
  	checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_64F, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
  	checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_64F, m, n, ldc));
  }

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

/****************************************************************************
  *
  **************************************************************************/
template <typename T>
int gemm_bias_gelu_lt(
	    	cublasLtHandle_t ltHandle, cublasOperation_t transa, cublasOperation_t transb, 
	     	int m, int n, int k, const float *alpha,   
	      	T A,  int lda, 
	     	T B,  int ldb, 	const float *beta,
	     	T C,  int64_t ldc, 
	      	void *workspace,  size_t workspaceSize, cudaStream_t stream, 
	     	bool use_bias,  const void* gelu_in, const void* bias) 
{
  cublasLtMatmulDesc_t               operationDesc = NULL;
  cublasLtMatrixLayout_t             Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
  cublasLtMatmulPreference_t         preference = NULL;
  cublasLtMatmulHeuristicResult_t    heuristicResult = {};
  cublasLtEpilogue_t                 epilogue = HIPBLASLT_EPILOGUE_GELU_AUX;

  int returnedResults = 0;

  // Create operation descriptor; see cublasLtMatmulDescAttributes_t for details about defaults; here we just set the 
  // transforms for A and B.
  if ((std::is_same<T, at::Half>::value) || (std::is_same<T, float>::value))
  {
          checkCublasStatus(cublasLtMatmulDescCreate( &operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  }

  if (std::is_same<T, double>::value)
  {
          checkCublasStatus(cublasLtMatmulDescCreate( &operationDesc, CUBLAS_COMPUTE_64F, CUDA_R_64F));
  }

  checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa))); 
  checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transa))); 

  checkCublasStatus(cublasLtMatmulDescSetAttribute( operationDesc, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &gelu_in, sizeof(gelu_in)));
  checkCublasStatus(cublasLtMatmulDescSetAttribute( operationDesc, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &ldc, sizeof(ldc)));

  if (use_bias) {
      checkCublasStatus(cublasLtMatmulDescSetAttribute( operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias))); 
      epilogue = HIPBLASLT_EPILOGUE_GELU_AUX_BIAS;
  } 

  checkCublasStatus(cublasLtMatmulDescSetAttribute( operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue))); 

  // Create matrix descriptors. 
  if (std::is_same<T, at::Half>::value)
  {
        checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_16F, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
        checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_16F, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
        checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16F, m, n, ldc));
  }

  if (std::is_same<T, float>::value)
  {
        checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
        checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
        checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, m, n, ldc));
  }

  if (std::is_same<T, double>::value)
  {
        checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_64F, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
        checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_64F, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
        checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_64F, m, n, ldc));
  }

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
  *
  *
  **************************************************************************/
template <typename T>
int gemm_bgradb_lt(    
	       	cublasLtHandle_t ltHandle, cublasOperation_t transa, cublasOperation_t transb,   
	       	int m, int n, int k, const float *alpha, 
	       	T A, int lda,    
		T B, int ldb, const float *beta,
	    	T C, int ldc,
	    	void *workspace, size_t workspaceSize, 	cudaStream_t stream, 
	     	bool use_bias, 	const void* bgrad) 
{
  cublasLtMatmulDesc_t             operationDesc = NULL;
  cublasLtMatrixLayout_t           Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
  cublasLtMatmulPreference_t       preference = NULL;
  cublasLtMatmulHeuristicResult_t  heuristicResult = {};
  cublasLtEpilogue_t               epilogue = CUBLASLT_EPILOGUE_DEFAULT;

  int returnedResults = 0;

  /*
   hipblasStatus_t hipblasLtMatmulDescCreate(
               hipblasLtMatmulDesc_t *matmulDesc,    //[out] Pointer to the structure holding the matrix multiply descriptor created by this function. 
	       hipblasComputeType_t   computeType,   //[in] Enumerant that specifies the data precision for the matrix multiply descriptor this function creates.
	       hipDataType            scaleType)     //[in] Enumerant that specifies the data precision for the matrix transform descriptor this function creates.

   This function creates a matrix multiply descriptor by allocating the memory needed to hold its opaque structure.

   Return values:
          HIPBLAS_STATUS_SUCCESS      – If the descriptor was created successfully.
          HIPBLAS_STATUS_ALLOC_FAILED – If the memory could not be allocated.

   ******************************************************************************************************************************
   enum hipblasComputeType_t:   The compute type to be used. Currently only used with GemmEx with the HIPBLAS_V2 interface. 
                                 Note that support for compute types is largely dependent on backend.

   Values:
       	HIPBLAS_COMPUTE_16F:            compute will be at least 16-bit precision
	HIPBLAS_COMPUTE_16F_PEDANTIC:   compute will be exactly 16-bit precision
	HIPBLAS_COMPUTE_32F:            compute will be at least 32-bit precision
	HIPBLAS_COMPUTE_32F_PEDANTIC:   compute will be exactly 32-bit precision
	HIPBLAS_COMPUTE_32F_FAST_16F:   32-bit input can use 16-bit compute
	HIPBLAS_COMPUTE_32F_FAST_16BF:  32-bit input can is bf16 compute
	HIPBLAS_COMPUTE_32F_FAST_TF32:  32-bit input can use tensor cores w/ TF32 compute. Only supported with cuBLAS backend currently
	HIPBLAS_COMPUTE_64F:            compute will be at least 64-bit precision
	HIPBLAS_COMPUTE_64F_PEDANTIC:   compute will be exactly 64-bit precision
	HIPBLAS_COMPUTE_32I:            compute will be at least 32-bit integer precision
	HIPBLAS_COMPUTE_32I_PEDANTIC:   compute will be exactly 32-bit integer precision
   */

  // Create operation descriptor.
  if ((std::is_same<T, at::Half>::value) || (std::is_same<T, float>::value))   {
          checkCublasStatus(cublasLtMatmulDescCreate( &operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  }

  if (std::is_same<T, double>::value)   {
          checkCublasStatus(cublasLtMatmulDescCreate( &operationDesc, CUBLAS_COMPUTE_64F, CUDA_R_64F));
  }
  /*
   hipblasLtMatmulDescSetAttribute()
   hipblasStatus_t hipblasLtMatmulDescSetAttribute(
               hipblasLtMatmulDesc_t              matmulDesc,   //[in] Pointer to the previously created structure holding the matrix multiply descriptor queried by this function.
	       hipblasLtMatmulDescAttributes_t    attr,         //[in] The attribute that will be set by this function.
	       const void                         *buf,         //[in] The value to which the specified attribute should be set.
	       size_t                             sizeInBytes)  //[in] Size of buf buffer (in bytes) for verification.

   This function sets the value of the specified attribute belonging to a previously created matrix multiply descriptor.

   Return values:
          HIPBLAS_STATUS_SUCCESS       – If the attribute was set successfully..
          HIPBLAS_STATUS_INVALID_VALUE – If buf is NULL or sizeInBytes doesn’t match the size of the internal storage for the selected attribute.

   ******************************************************************************************************************
   enum hipblasLtMatmulDescAttributes_t: Specify the attributes that define the specifics of the matrix multiply operation.

   Values:
       HIPBLASLT_MATMUL_DESC_TRANSA           Specifies the type of transformation operation that should be performed on matrix A. 
                                              Default value is HIPBLAS_OP_N (for example, non-transpose operation). See hipblasOperation_t. Data Type:int32_t
       HIPBLASLT_MATMUL_DESC_TRANSB           Specifies the type of transformation operation that should be performed on matrix B. 
                                              Default value is HIPBLAS_OP_N (for example, non-transpose operation). See hipblasOperation_t. Data Type:int32_t
       HIPBLASLT_MATMUL_DESC_EPILOGUE         Epilogue function. See hipblasLtEpilogue_t. Default value is: HIPBLASLT_EPILOGUE_DEFAULT. Data Type: uint32_t
       HIPBLASLT_MATMUL_DESC_BIAS_POINTER     Bias or Bias gradient vector pointer in the device memory. Data Type:void* /const void*
       HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE   Type of the bias vector in the device memory. Can be set same as D matrix type or Scale type. 
                                              Bias case: see HIPBLASLT_EPILOGUE_BIAS. Data Type:int32_t based on hipDataType
       HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER  Device pointer to the scale factor value that converts data in matrix A to the compute data type range. 
                                              The scaling factor must have the same type as the compute type. If not specified, or set to NULL, 
					      the scaling factor is assumed to be 1. If set for an unsupported matrix data, scale, and compute type combination, 
					      calling hipblasLtMatmul() will return HIPBLAS_INVALID_VALUE. Default value: NULL Data Type: void* /const void*
       HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER  Equivalent to HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER for matrix B. Default value: NULL Type: void* /const void*
       HIPBLASLT_MATMUL_DESC_C_SCALE_POINTER  Equivalent to HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER for matrix C. Default value: NULL Type: void* /const void*
       HIPBLASLT_MATMUL_DESC_D_SCALE_POINTER  Equivalent to HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER for matrix D. Default value: NULL Type: void* /const void*

       HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER  Equivalent to HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER for matrix AUX. Default value: NULL Type: void* /const void*
       HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER        Epilogue auxiliary buffer pointer in the device memory. Data Type:void* /const void*
       HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD             The leading dimension of the epilogue auxiliary buffer pointer in the device memory. Data Type:int64_t
       HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE   The batch stride of the epilogue auxiliary buffer pointer in the device memory. Data Type:int64_t
       HIPBLASLT_MATMUL_DESC_POINTER_MODE                Specifies alpha and beta are passed by reference, whether they are scalars on the host or on the device, 
                                                         or device vectors. Default value is: HIPBLASLT_POINTER_MODE_HOST (i.e., on the host). 
							 Data Type: int32_t based on hipblasLtPointerMode_t
       HIPBLASLT_MATMUL_DESC_AMAX_D_POINTER              Device pointer to the memory location that on completion will be set to the maximum of absolute values 
                                                         in the output matrix. Data Type:void* /const void*
       HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_A_EXT    Compute input A types. Defines the data type used for the input A of matrix multiply.
       HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_B_EXT    Compute input B types. Defines the data type used for the input B of matrix multiply.
       HIPBLASLT_MATMUL_DESC_MAX

   ******************************************************************************************************************

   enum hipblasLtEpilogue_t: Specify the enum type to set the postprocessing options for the epilogue.

   Values:
       HIPBLASLT_EPILOGUE_DEFAULT         No special postprocessing, just scale and quantize the results if necessary.
       HIPBLASLT_EPILOGUE_RELU            Apply ReLU point-wise transform to the results:(x:=max(x, 0))
       HIPBLASLT_EPILOGUE_BIAS            Apply (broadcast) bias from the bias vector. Bias vector length must match matrix D rows, 
                                          and it must be packed (such as stride between vector elements is 1). Bias vector is broadcast to 
                                          all columns and added before applying the final postprocessing.

       HIPBLASLT_EPILOGUE_RELU_BIAS       Apply bias and then ReLU transform.
       HIPBLASLT_EPILOGUE_GELU            Apply GELU point-wise transform to the results (x:=GELU(x)).
       HIPBLASLT_EPILOGUE_GELU_BIAS       Apply Bias and then GELU transform.
       HIPBLASLT_EPILOGUE_GELU_AUX        Output GEMM results before applying GELU transform.
       HIPBLASLT_EPILOGUE_GELU_AUX_BIAS   Output GEMM results after applying bias but before applying GELU transform.
       HIPBLASLT_EPILOGUE_DGELU           Apply gradient GELU transform. Requires additional aux input.
       HIPBLASLT_EPILOGUE_DGELU_BGRAD     Apply gradient GELU transform and bias gradient to the results. Requires additional aux input.
       HIPBLASLT_EPILOGUE_BGRADA          Apply bias gradient to A and output gemm result.
       HIPBLASLT_EPILOGUE_BGRADB          Apply bias gradient to B and output gemm result.
   */

  checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
  checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transa))); 

  if (use_bias) {
    // Bias or Bias gradient vector pointer in the device memory.
    checkCublasStatus(cublasLtMatmulDescSetAttribute( operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bgrad, sizeof(bgrad))); 

    // Apply bias gradient to B and output gemm result.
    epilogue = HIPBLASLT_EPILOGUE_BGRADB; 
  } 

  checkCublasStatus(cublasLtMatmulDescSetAttribute( operationDesc,  CUBLASLT_MATMUL_DESC_EPILOGUE,  &epilogue,  sizeof(epilogue))); 

  // Create matrix descriptors. Not setting any extra attributes.
  /*
   hipblasLtMatrixLayoutCreate()
   hipblasStatus_t hipblasLtMatrixLayoutCreate(
         hipblasLtMatrixLayout_t *matLayout,   //[out] Pointer to structure holding matrix layout descriptor created by this function.
         hipDataType              type,        //[in]  Enumerant that specifies the data precision for the matrix layout descriptor this function creates.
         uint64_t                 rows,        //[in]  Number of rows of the matrix.
	 uint64_t                 cols,        //[in]  Number of columns of the matrix.
	 int64_t                  ld)          //[in]  The leading dimension of the matrix. In column major layout, this is the number of elements 
		                                                    to jump to reach the next column. Thus ld >= m (number of rows).

   This function creates a matrix layout descriptor by allocating the memory needed to hold its opaque structure.

   Return values:
       HIPBLAS_STATUS_SUCCESS      – If the descriptor was created successfully.
       HIPBLAS_STATUS_ALLOC_FAILED – If the memory could not be allocated.
  */
  if (std::is_same<T, at::Half>::value)
  {
        checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_16F, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
        checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_16F, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
        checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16F, m, n, ldc));
  }

  if (std::is_same<T, float>::value)
  {
        checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
        checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
        checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, m, n, ldc));
  }

  if (std::is_same<T, double>::value)
  {
        checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_64F, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
        checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_64F, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
        checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_64F, m, n, ldc));
  }

  /*
   hipblasLtMatmulPreferenceCreate()
   hipblasStatus_t hipblasLtMatmulPreferenceCreate(
                hipblasLtMatmulPreference_t *pref)        //[out] Pointer to the structure holding the matrix multiply preferences descriptor created by this function. 

   This function creates a matrix multiply heuristic search preferences descriptor by allocating the memory needed to hold its opaque structure.
  
   Return values:
       HIPBLAS_STATUS_SUCCESS      – If the descriptor was created successfully.
       HIPBLAS_STATUS_ALLOC_FAILED – If memory could not be allocated.
  */

  checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference));


  /*
   hipblasLtMatmulPreferenceSetAttribute()
   hipblasStatus_t hipblasLtMatmulPreferenceSetAttribute(hipblasLtMatmulPreference_t pref, hipblasLtMatmulPreferenceAttributes_t attr, const void *buf, size_t sizeInBytes)

   Set attribute to a preference descriptor. This function sets the value of the specified attribute belonging to a 
   previously created matrix multiply preferences descriptor.

   Parameters:
	pref – [in] Pointer to the previously created structure holding the matrix multiply preferences descriptor queried by this function. See hipblasLtMatmulPreference_t
	attr – [in] The attribute that will be set by this function. See hipblasLtMatmulPreferenceAttributes_t.
	buf – [in] The value to which the specified attribute should be set.
	sizeInBytes – [in] Size of buf buffer (in bytes) for verification.

   Return values:
	HIPBLAS_STATUS_SUCCESS – If the attribute was set successfully..
	HIPBLAS_STATUS_INVALID_VALUE – If buf is NULL or sizeInBytes doesn’t match the size of the internal storage for the selected attribute.
   */
  checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize))); 



  /*
   hipblasLtMatmulAlgoGetHeuristic()
   
   hipblasStatus_t hipblasLtMatmulAlgoGetHeuristic(
               hipblasLtHandle_t                 handle,                  //[in]  Pointer to allocated hipBLASLt handle for the hipBLASLt context.
	       hipblasLtMatmulDesc_t             matmulDesc,              //[in]  Handle to a previously created matrix multiplication descriptor.
	       hipblasLtMatrixLayout_t           Adesc,                   //[in]  Handles to the previously created matrix layout descriptors
	       hipblasLtMatrixLayout_t           Bdesc,                   //[in]  Handles to the previously created matrix layout descriptors
	       hipblasLtMatrixLayout_t           Cdesc,                   //[in]  Handles to the previously created matrix layout descriptors
	       hipblasLtMatrixLayout_t           Ddesc,                   //[in]  Handles to the previously created matrix layout descriptors
	       hipblasLtMatmulPreference_t       pref,                    //[in]  Pointer to the structure holding the heuristic search preferences descriptor. 
	       int                               requestedAlgoCount,      //[in]  Size of the heuristicResultsArray (in elements) requested maximum number of algorithms to return.
	       hipblasLtMatmulHeuristicResult_t  heuristicResultsArray[], //[out] Array containing the algorithm heuristics and associated runtime characteristics, 
	                                                                  //      returned by this function, in the order of increasing estimated compute time.
	       int                               *returnAlgoCount)        //[out] Number of algorithms returned by this function. This is the number of heuristicResultsArray 
	                                                                  //      elements written.
	       
    Retrieve the possible algorithms. This function retrieves the possible algorithms for the matrix multiply operation hipblasLtMatmul() 
    function with the given input matrices A, B and C, and the output matrix D. The output is placed in heuristicResultsArray[] 
    in the order of increasing estimated compute time.

    Return values:
	HIPBLAS_STATUS_SUCCESS       – If query was successful. Inspect heuristicResultsArray[0 to (returnAlgoCount -1)].state for the status of the results.
	HIPBLAS_STATUS_NOT_SUPPORTED – If no heuristic function available for current configuration.
	HIPBLAS_STATUS_INVALID_VALUE – If requestedAlgoCount is less or equal to zero.
   */

  checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults));

  if (returnedResults == 0) { checkCublasStatus(CUBLAS_STATUS_NOT_SUPPORTED); }

  /*
   hipblasLtMatmul()
   hipblasStatus_t hipblasLtMatmul(
   		hipblasLtHandle_t            handle,                 //[in]  Pointer to the allocated hipBLASLt handle for the hipBLASLt context. 
		hipblasLtMatmulDesc_t        matmulDesc,             //[in]  Handle to a previously created matrix multiplication descriptor of type hipblasLtMatmulDesc_t
		const void                   *alpha,                 //[in]  Pointers to the scalars used in the multiplication.
		const void                   *A,                     //[in]  Pointers to the GPU memory associated with the corresponding descriptors Adesc. 
		hipblasLtMatrixLayout_t      Adesc,                  //[in]  Handles to the previously created matrix layout descriptors of the type hipblasLtMatrixLayout_t
		const void                   *B,                     //[in]  Pointers to the GPU memory associated with the corresponding descriptors Bdesc
		hipblasLtMatrixLayout_t      Bdesc,                  //[in]  Handles to the previously created matrix layout descriptors of the type hipblasLtMatrixLayout_t
		const void                   *beta,                  //[in]  Pointers to the scalars used in the multiplication.
		const void                   *C,                     //[in]  Pointers to the GPU memory associated with the corresponding descriptors Cdesc .
		hipblasLtMatrixLayout_t      Cdesc,                  //[in]  Handles to the previously created matrix layout descriptors of the type hipblasLtMatrixLayout_t  
		void                         *D,                     //[out] Pointer to the GPU memory associated with the descriptor Ddesc .
		hipblasLtMatrixLayout_t      Ddesc,                  //[in]  Handles to the previously created matrix layout descriptors of the type hipblasLtMatrixLayout_t
		const hipblasLtMatmulAlgo_t  *algo,                  //[in]  Handle for matrix multiplication algorithm to be used. See hipblasLtMatmulAlgo_t . 
		                                                             When NULL, an implicit heuristics query with default search preferences will be performed to 
									     determine actual algorithm to use.
		void                         *workspace,             //[in]  Pointer to the workspace buffer allocated in the GPU memory. Pointer must be 16B aligned 
		                                                             (that is, lowest 4 bits of address must be 0).
		size_t                       workspaceSizeInBytes,   //[in]  Size of the workspace. 
		hipStream_t                  stream)                 //[in]  The HIP stream where all the GPU work will be submitted.

    This function computes the matrix multiplication of matrices A and B to produce the output matrix D, according to the following operation: 
       
                           D = alpha*(A * B) + beta*( C )      
			   
		   Where    A, B, and C    : are input matrices, and 
			    alpha and beta : are input scalars. 

    Note: This function supports both in-place matrix multiplication (C == D and Cdesc == Ddesc) and 
          out-of-place matrix multiplication (C != D, both matrices must have the same data type, number of rows, number of columns, batch size, and memory order). 
	  In the out-of-place case, the leading dimension of C can be different from the leading dimension of D. 
	  Specifically the leading dimension of C can be 0 to achieve row or column broadcast. If Cdesc is omitted, this function assumes it to be equal to Ddesc.


    Return values:
         HIPBLAS_STATUS_SUCCESS          – If the operation completed successfully.
         HIPBLAS_STATUS_EXECUTION_FAILED – If HIP reported an execution error from the device.
         HIPBLAS_STATUS_ARCH_MISMATCH    – If the configured operation cannot be run using the selected device.
         HIPBLAS_STATUS_NOT_SUPPORTED    – If the current implementation on the selected device doesn’t support the configured operation.
         HIPBLAS_STATUS_INVALID_VALUE    – If the parameters are unexpectedly NULL, in conflict or in an impossible configuration. 
	                                   For example, when workspaceSizeInBytes is less than workspace required by the configured algo.
         HIBLAS_STATUS_NOT_INITIALIZED – If hipBLASLt handle has not been initialized.
   */


  checkCublasStatus(cublasLtMatmul(ltHandle, operationDesc, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, C, Cdesc, NULL, workspace, workspaceSize, stream));

  // descriptors are no longer needed as all GPU work was already enqueued
  if (preference) checkCublasStatus(cublasLtMatmulPreferenceDestroy(preference));
  if (Cdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Cdesc));
  if (Bdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));
  if (Adesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));
  if (operationDesc) checkCublasStatus(cublasLtMatmulDescDestroy(operationDesc));
}


/****************************************************************************
  *
  *
  **************************************************************************/
template <typename T>
int gemm_dgelu_bgradb_lt(    
		cublasLtHandle_t ltHandle, cublasOperation_t transa, cublasOperation_t transb,    
		int m,  int n,  int k,   const float *alpha, 
		T A,   int lda,   
	       	T B,   int ldb,  const float *beta,     
		T C,   int64_t ldc,    
		void *workspace, size_t workspaceSize, 	cudaStream_t stream,   
	       	const void *gelu_in, const void *bgrad) 
{

  cublasLtMatmulDesc_t             operationDesc = NULL;
  cublasLtMatrixLayout_t           Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
  cublasLtMatmulPreference_t       preference = NULL;
  cublasLtMatmulHeuristicResult_t  heuristicResult  = {};
  cublasLtEpilogue_t               epilogue = HIPBLASLT_EPILOGUE_DGELU_BGRAD;

  int                              returnedResults  = 0;

  // Create operation descriptor; see cublasLtMatmulDescAttributes_t for details about defaults; here we just set the transforms for
  // A and B.
  checkCublasStatus(cublasLtMatmulDescCreate( &operationDesc,       CUBLAS_COMPUTE_32F, CUDA_R_32F));
  checkCublasStatus(cublasLtMatmulDescSetAttribute( operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa,       sizeof(transa)));
  checkCublasStatus(cublasLtMatmulDescSetAttribute( operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb,       sizeof(transa)));

  checkCublasStatus(cublasLtMatmulDescSetAttribute( operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bgrad,  sizeof(bgrad))); 
 
  checkCublasStatus(cublasLtMatmulDescSetAttribute( operationDesc, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &gelu_in,  sizeof(gelu_in))); 
  checkCublasStatus(cublasLtMatmulDescSetAttribute( operationDesc, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD,      &ldc,      sizeof(ldc)));
  checkCublasStatus(cublasLtMatmulDescSetAttribute( operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE,             &epilogue, sizeof(epilogue))); 

  // Create matrix descriptors. Not setting any extra attributes.
  checkCublasStatus(cublasLtMatrixLayoutCreate( &Adesc, CUDA_R_16F, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda)); 
  checkCublasStatus(cublasLtMatrixLayoutCreate( &Bdesc, CUDA_R_16F, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb)); 
  checkCublasStatus(cublasLtMatrixLayoutCreate( &Cdesc, CUDA_R_16F,  m,  n,  ldc));

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
int linear_bias_forward_cuda(
		at::Tensor input, T *weight, 	      at::Tensor bias, 	 int in_features, 
		int batch_size,   int out_features,   at::Tensor output, void *lt_workspace) 
{
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

    // Get the stream from cublas handle to reuse for biasReLU kernel.
    cudaStream_t stream;
    cublasGetStream(handle, &stream);
    int status = HIPBLAS_STATUS_NOT_INITIALIZED;

    const float alpha     = 1.0, beta_zero = 0.0, beta_one  = 1.0;

#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 11600 || defined(USE_ROCM)
     status = gemm_bias_lt(
		  //cublasLtHandle_t ltHandle     cublasOperation_t transa,   cublasOperation_t transb,
		    (cublasLtHandle_t)handle,     CUBLAS_OP_T,                CUBLAS_OP_N,

		  //int m           int n,        int k,            const float *alpha,   T A,
		    out_features,   batch_size,	  in_features,     &alpha,               weight,

		  //int lda,        T B,                              int ldb,         const float *beta
                    in_features,    input.data_ptr<T>(),              in_features,     &beta_zero,

		  //T C                              int ldc          void *workspace  size_t workspaceSize
                    output.data_ptr<T>(),            out_features,    lt_workspace,    1 << 22,

		  //cudaStream_t stream    bool use_bias,   const void* bias
                    stream,                true,            static_cast<const void*>(bias.data_ptr<T>()));
#endif  
    output.copy_(bias);
    if(status!=HIPBLAS_STATUS_SUCCESS)
    {
	    status = gemm_bias(
			    handle,                CUBLAS_OP_T,	          CUBLAS_OP_N,	
		            out_features,	   batch_size,	          in_features,	 &alpha,	  weight,
		            in_features,	   input.data_ptr<T>(),   in_features,	 &beta_one,
		            output.data_ptr<T>(),  out_features);
    }
    return status;
}

  
/****************************************************************************
  *
  *
  **************************************************************************/
template <typename T>
int linear_bias_backward_cuda(
		T *input, T *weight, T *d_output,  int in_features,  int batch_size, 
		int out_features,    T *d_weight,  T *d_bias,        T *d_input,  
		void *lt_workspace) 
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
		    (cublasLtHandle_t)handle,	   CUBLAS_OP_N,    CUBLAS_OP_T,
		    // m                           n               k                alpha
		    in_features,                   out_features,   batch_size,	    &alpha,
		    // A                            lda            B                ldb  
		    input,        	 	   in_features,    d_output,	    out_features,
		    // beta                        C               ldc
		    &beta_zero,            	   d_weight,       in_features,	    lt_workspace,
		    1 << 22,     		   stream,         true,	    static_cast<const void*>(d_bias));
#endif

    if (status != HIPBLAS_STATUS_SUCCESS)
    { 
	    status = gemm_bias(  
		    	    handle,    	    CUBLAS_OP_N,    CUBLAS_OP_T,
			    in_features,    out_features,   batch_size,     &alpha,     input,  
		    	    in_features,    d_output,  	    out_features,   &beta_zero, d_weight,   
		     	    in_features);
    }
    
    status = gemm_bias(   
		    handle,         CUBLAS_OP_N,   CUBLAS_OP_N,  
		    in_features,    batch_size,    out_features,    &alpha,       weight,  
		    in_features,    d_output,      out_features,    &beta_zero,   d_input,  
		    in_features);
    return status;

}

/****************************************************************************
  *
  *
  **************************************************************************/
template <typename T>
int linear_gelu_linear_forward_cuda(
		T *input, 	T *weight1, 		T *bias1, 		T *weight2, 
		T *bias2, 	int in_features, 	int hidden_features, 	int batch_size, 
		int out_features, 	T *output1, 	T *output2, 		T *gelu_in, 
		void *lt_workspace) 
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
		    1 << 22,                   stream,          true,   	    static_cast<const void*>(gelu_in), 
		    static_cast<const void*>(bias1));
    status = gemm_bias_lt(
		    (cublasLtHandle_t)handle,   CUBLAS_OP_T,     CUBLAS_OP_N,	    out_features, 
		    batch_size,                 hidden_features, &alpha, 	    weight2,   
		    hidden_features,            output1,         hidden_features,   &beta_zero, 
		    output2,                    out_features,    lt_workspace,      1 << 22,
		    stream,                     true,            static_cast<const void*>(bias2));
#endif
    return status;
}


/****************************************************************************
  *
  *
  **************************************************************************/
template <typename T>
int linear_gelu_linear_backward_cuda(
		T *input, 		T *gelu_in, 		T *output1, 
		T *weight1, 		T *weight2, 		T *d_output1, 
		T *d_output2, 		int in_features, 	int batch_size, 
		int hidden_features, 	int out_features, 	T *d_weight1, 
		T *d_weight2, 		T *d_bias1, 		T *d_bias2, 
		T *d_input, 		void *lt_workspace) 
{
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

    // Get the stream from cublas handle to reuse for biasReLU kernel.
    cudaStream_t stream;
    cublasGetStream(handle, &stream);

    const float alpha      = 1.0;
    const float beta_zero  = 0.0;
    const float beta_one   = 1.0;
    int         status     = HIPBLAS_STATUS_NOT_INITIALIZED;

#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 11000 || defined(USE_ROCM)
    //wgrad for first gemm
    status = gemm_bgradb_lt(
		    (cublasLtHandle_t)handle,     CUBLAS_OP_N,		    CUBLAS_OP_T,
		    hidden_features,              out_features,		    batch_size,
		    &alpha,                       output1,  		    hidden_features,
		    d_output2,                    out_features,		    &beta_zero, /* host pointer */
		    d_weight2,                    hidden_features,  	    lt_workspace, 
		    1 << 22,                      stream,   		    true,
		    static_cast<const void*>(d_bias2));

    //dgrad for second GEMM
    status = gemm_dgelu_bgradb_lt(    
		    (cublasLtHandle_t)handle,     CUBLAS_OP_N,    	    CUBLAS_OP_N,    
		    hidden_features,              batch_size,    	    out_features,    
		    &alpha,                       weight2,    		    hidden_features,    
		    d_output2,                    out_features,    	    &beta_zero, /* host pointer */    
		    d_output1,                    hidden_features,    	    lt_workspace,    
		    1 << 22,    stream,           static_cast<const void*>(gelu_in),    
		    static_cast<const void*>(d_bias1));

    //wgrad for the first GEMM    
    status = gemm_bias(  
		    handle, 		    CUBLAS_OP_N,      		    CUBLAS_OP_T,
	      	    in_features,      	    hidden_features,	      	    batch_size,
	      	    &alpha,  		    input,      	      	    in_features,
	      	    d_output1,	      	    hidden_features, 	       	    &beta_zero,  
		    d_weight1, 	       	    in_features);

    //dgrad for the first GEMM    
    status = gemm_bias(      
		    handle, 		    CUBLAS_OP_N,      		    CUBLAS_OP_N,      
		    in_features,      	    batch_size,         	    hidden_features,      
		    &alpha,      	    weight1,      		    in_features,      
		    d_output1,      	    hidden_features,      	    &beta_zero,      
		    d_input,      	    in_features);
#endif
    return status;

}
    
template int linear_bias_forward_cuda<float>(
		at::Tensor input, 	float *weight, 		at::Tensor bias, 
		int in_features, 	int batch_size, 	int out_features, 
		at::Tensor output, 	void *lt_workspace);

template int linear_bias_forward_cuda<double>(
		at::Tensor input, 	double *weight, 	at::Tensor bias, 
		int in_features, 	int batch_size, 	int out_features, 
		at::Tensor output, 	void *lt_workspace);

template int linear_bias_backward_cuda<at::Half>(
		at::Half *input, 	at::Half *weight, 	at::Half *d_output, 
		int in_features, 	int batch_size, 	int out_features, 
		at::Half *d_weight, 	at::Half *d_bias, 	at::Half *d_input,  
		void *lt_workspace) ;

template int linear_bias_backward_cuda<float>(
		float *input,    	float *weight, 		float *d_output, 
		int in_features, 	int batch_size, 	int out_features, 
		float *d_weight, 	float *d_bias, 		float *d_input,  
		void *lt_workspace) ;

template int linear_bias_backward_cuda<double>(
		double *input, 		double *weight, 	double *d_output, 
		int in_features, 	int batch_size, 	int out_features, 
		double *d_weight, 	double *d_bias, 	double *d_input,  
		void *lt_workspace) ;

template int linear_gelu_linear_forward_cuda<at::Half>(
		at::Half *input, 	at::Half *weight1, 	at::Half *bias1, 
		at::Half *weight2, 	at::Half *bias2, 	int in_features, 
		int hidden_features, 	int batch_size, 	int out_features, 
		at::Half *output1, 	at::Half *output2, 	at::Half *gelu_in, 
		void *lt_workspace) ;

template int linear_gelu_linear_forward_cuda<float>(
		float *input,     	float *weight1, 	float *bias1, 
		float *weight2, 	float *bias2,           int in_features, 
		int hidden_features, 	int batch_size, 	int out_features, 
		float *output1, 	float *output2, 	float *gelu_in, 
		void *lt_workspace);
    
template int linear_gelu_linear_forward_cuda<double>(
		double *input, 		double *weight1, 	double *bias1, 
		double *weight2, 	double *bias2, 		int in_features, 
		int hidden_features, 	int batch_size, 	int out_features, 
		double *output1, 	double *output2, 	double *gelu_in, 
		void *lt_workspace) ;

template int linear_gelu_linear_backward_cuda<at::Half>(
		at::Half *input, 	at::Half *gelu_in, 	at::Half *output1, 
		at::Half *weight1, 	at::Half *weight2, 	at::Half *d_output1, 		
		at::Half *d_output2, 	int in_features, 	int batch_size, 
		int hidden_features, 	int out_features, 	at::Half *d_weight1, 
		at::Half *d_weight2, 	at::Half *d_bias1, 	at::Half *d_bias2, 
		at::Half *d_input, 	void *lt_workspace);

template int linear_gelu_linear_backward_cuda<float>(
		float *input, 		float *gelu_in, 	float *output1, 
		float *weight1, 	float *weight2, 	float *d_output1, 
		float *d_output2, 	int in_features, 	int batch_size, 
		int hidden_features, 	int out_features, 	float *d_weight1, 
		float *d_weight2, 	float *d_bias1, 	float *d_bias2, 
		float *d_input, 	void *lt_workspace);


template int linear_gelu_linear_backward_cuda<double>(
		double *input, 		double *gelu_in, 	double *output1, 
		double *weight1, 	double *weight2, 	double *d_output1, 
		double *d_output2, 	int in_features, 	int batch_size, 
		int hidden_features, 	int out_features, 	double *d_weight1, 
		double *d_weight2, 	double *d_bias1, 	double *d_bias2, 
		double *d_input, 	void *lt_workspace);

