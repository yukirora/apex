#pragma once
#include <iostream>
#include <vector>

#include <cuda.h>
#include <cuda_fp16.h>
//#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

//#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>

//#include "cutlass/cutlass.h"
//#include "cutlass/gemm/gemm.h"
//#include "cutlass/gemm/wmma_gemm_traits.h"

// symbol to be automatically resolved by PyTorch libs

rocblas_datatype a_type       = rocblas_datatype_f16_r;
rocblas_datatype b_type       = rocblas_datatype_f16_r;
rocblas_datatype c_type       = rocblas_datatype_f16_r;
rocblas_datatype d_type       = rocblas_datatype_f16_r;
rocblas_datatype compute_type       = rocblas_datatype_f32_r;

rocblas_gemm_algo algo           = rocblas_gemm_algo_standard;
int32_t           solution_index = 0;
rocblas_int       flags          = 0;


namespace {
cublasOperation_t convertTransToCublasOperation(char trans) {
  if (trans == 't')
    return CUBLAS_OP_T;
  else if (trans == 'n')
    return CUBLAS_OP_N;
  else if (trans == 'c')
    return CUBLAS_OP_C;
  else {
    AT_ERROR("trans must be one of: t, n, c");
    return CUBLAS_OP_T;
  }
}

void RocblasStridedBatchedGemm(char transa, char transb, long m, long n, long k,
                    float alpha, const half *a, long lda, long strideA, const half *b, long ldb, long strideB,
                    float beta, half *c, long ldc, long strideC, half *d, long ldd, long strideD, long batchCount, rocblas_gemm_algo algo) {
    cublasOperation_t opa = convertTransToCublasOperation(transa);
    cublasOperation_t opb = convertTransToCublasOperation(transb);

    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cudaStream_t   stream = at::cuda::getCurrentCUDAStream().stream();
    cublasSetStream(handle, stream);
    float fAlpha = alpha;
    float fBeta = beta;
    //THCublasCheck(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
    TORCH_CUDABLAS_CHECK(rocblas_gemm_strided_batched_ex(handle,
                                     opa, opb, (int)m, (int)n, (int)k,
                                     (void*)&fAlpha, a, a_type, (int)lda, strideA,
                                     b, b_type, (int)ldb, strideB,
                                     (void*)&fBeta, c, c_type, (int)ldc, strideC,
				     d, d_type, int(ldd), strideD,
                                     (int)batchCount, compute_type, algo, solution_index, flags));
}
} // namespace

template <cutlass::MatrixLayout::Kind A_LAYOUT,
          cutlass::MatrixLayout::Kind B_LAYOUT, int SRC_A, int SRC_B, int DST_C>
void CutlassGemm_FP32Accum(cudaStream_t stream, long m, long n, long k,
                           float alpha, const half *a, long lda, long strideA,
                           const half *b, long ldb, long strideB, float beta,
                           half *c, long ldc, long strideC, long batchCount) {
  // printf("CUTLASS-> %c%c M: %ld N: %ld K: %ld %d%d%d LDA: %ld LDB: %ld LDC:
  // %ld strideA: %ld strideB: %ld strideC: %ld Alpha: %f Beta: %f\n",
  // ((int)A_LAYOUT == 0 ? 'T' : 'N'), ((int)B_LAYOUT ==0 ? 'T' : 'N'), m, n, k,
  // SRC_A,SRC_B,DST_C, lda, ldb, ldc, strideA, strideB, strideC, alpha, beta);
  typedef cutlass::gemm::WmmaGemmTraits<
      A_LAYOUT, B_LAYOUT, cutlass::Shape<32, 16, 16>, half, half, half,
      cutlass::gemm::LinearScaling<float>, float,
      typename cutlass::gemm::WmmaGemmAccumulatorsPerWarp<
          typename cutlass::Shape<32, 16, 16>>::Shape,
      typename cutlass::Shape<16, 16, 16>,
      SRC_A,     // kScalarsPerLdgA_
      SRC_B,     // kScalarsPerLdgB_
      SRC_A,     // KScalarsPerLdsA_
      SRC_B,     // KScalarsPerLdsB_
      DST_C,     // kScalarsPerLdgCAndStgD_
      DST_C / 2, // kScalarsPerStsD_
      DST_C / 2  // kScalarsPerLdsD_
      >
      WmmaGemmTraits;

  typedef cutlass::gemm::Gemm<WmmaGemmTraits> Gemm;
  typename Gemm::Params params;

  int result = params.initialize(
      m,     // M dimension for each batch
      n,     // N dimension for each batch
      k,     // K dimension for each batch
      alpha, // scalar alpha
      a, lda,
      strideA, // distance in memory between the first element of neighboring
               // batch
      b, ldb,
      strideB, // distance in memory between the first element of neighboring
               // batch
      beta,    // scalar beta
      c,       // source matrix C
      ldc,
      strideC, // distance in memory between the first element of neighboring
               // batch
      c, // destination matrix C (may be different memory than source C matrix)
      ldc,
      strideC, // distance in memory between the first element of neighboring
               // batch
      batchCount);

  AT_ASSERTM(result == 0, "Failed to initialize CUTLASS Gemm::Params object.");

  // batchCount in cutlass batched GEMM kernels maps to gridDim.z, which is
  // limited to 16 bits. To implement batched GEMM with larger batch size, we
  // fragment it into smaller batched GEMMs of gridDim.z <= 64k
  long batchesLeft = batchCount;
  long iterBatchCount = std::min(batchesLeft, static_cast<long>((1 << 16) - 1));

  do {
    // printf("CUTLASS-> %c%c M: %ld N: %ld K: %ld %d%d%d LDA: %ld LDB: %ld LDC:
    // %ld strideA: %ld strideB: %ld strideC: %ld Alpha: %f Beta: %f
    // TotalBatches: %ld iterBatchCount %ld\n", ((int)A_LAYOUT == 0 ? 'T' : 'N'),
    // ((int)B_LAYOUT ==0 ? 'T' : 'N'), m, n, k, SRC_A,SRC_B,DST_C, lda, ldb,
    // ldc, strideA, strideB, strideC, alpha, beta, batchesLeft, iterBatchCount);
    int result =
        params.initialize(m,     // M dimension for each batch
                          n,     // N dimension for each batch
                          k,     // K dimension for each batch
                          alpha, // scalar alpha
                          a, lda,
                          strideA, // distance in memory between the first
                                   // element of neighboring batch
                          b, ldb,
                          strideB, // distance in memory between the first
                                   // element of neighboring batch
                          beta,    // scalar beta
                          c,       // source matrix C
                          ldc,
                          strideC, // distance in memory between the first
                                   // element of neighboring batch
                          c, // destination matrix C (may be different memory
                             // than source C matrix)
                          ldc,
                          strideC, // distance in memory between the first
                                   // element of neighboring batch
                          iterBatchCount);

    AT_ASSERTM(result == 0,
               "Failed to initialize CUTLASS Gemm::Params object.");
    // Launch the CUTLASS GEMM kernel.
    C10_CUDA_CHECK(Gemm::launch(params, stream));

    // Update batched GEMM params based on completed work
    batchesLeft = batchesLeft - iterBatchCount;
    a += iterBatchCount * strideA;
    b += iterBatchCount * strideB;
    c += iterBatchCount * strideC;
    ;

    iterBatchCount = std::min(batchesLeft, static_cast<long>((1 << 16) - 1));

  } while (batchesLeft > 0);
}

namespace {
void gemm_switch_fp32accum(char transa, char transb, long m,
                           long n, long k, float alpha, const half *a, long lda,
                           long strideA, const half *b, long ldb, long strideB,
                           float beta, half *c, long ldc, long strideC,
                           long batchCount) {
  auto stream = c10::cuda::getCurrentCUDAStream();
  if        ( (transa == 't') && (transb == 'n') ) { 
    if      (!(lda & 0x7) && !(ldb & 0x7) && !(ldc & 0x7)) { RocblasStridedBatchedGemm(transa, transb, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, d, ldd, strideD, batchCount, algo); }
    else                                                   { RocblasStridedBatchedGemm(transa, transb, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, d, ldd, strideD, batchCount, algo); }
  } else if ( (transa == 'n') && (transb == 'n') ) {
    if      (!(lda & 0x7) && !(ldb & 0x7) && !(ldc & 0x7)) { RocblasStridedBatchedGemm(transa, transb, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, d, ldd, strideD, batchCount, algo); }
    else                                                   { RocblasStridedBatchedGemm(transa, transb, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, d, ldd, strideD, batchCount, algo); }
  } else if ( (transa == 'n') && (transb == 't') ) {
    if      (!(lda & 0x7) && !(ldb & 0x7) && !(ldc & 0x7)) {RocblasStridedBatchedGemm(transa, transb, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, d, ldd, strideD, batchCount, algo); }
    else                                                   { RocblasStridedBatchedGemm(transa, transb, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, d, ldd, strideD, batchCount, algo); }
  } else {
    AT_ASSERTM(false, "TransA and TransB are invalid");
  }
}

void adjustLdLevel3(char transa, char transb, int64_t m, int64_t n, int64_t k,
                    int64_t *lda, int64_t *ldb, int64_t *ldc) {
  int transa_ = ((transa == 't') || (transa == 'T'));
  int transb_ = ((transb == 't') || (transb == 'T'));

  // Note: leading dimensions generally are checked that they are > 0 and at
  // least as big the result requires (even if the value won't be used).
  if (n <= 1)
    *ldc = std::max<int64_t>(m, 1);

  if (transa_) {
    if (m <= 1)
      *lda = std::max<int64_t>(k, 1);
  } else {
    if (k <= 1)
      *lda = std::max<int64_t>(m, 1);
  }

  if (transb_) {
    if (k <= 1)
      *ldb = std::max<int64_t>(n, 1);
  } else {
    if (n <= 1)
      *ldb = std::max<int64_t>(k, 1);
  }
}

void HgemmStridedBatched(char transa, char transb, long m,
                         long n, long k, float alpha, const half *a, long lda,
                         long strideA, const half *b, long ldb, long strideB,
                         float beta, half *c, long ldc, long strideC,
                         half *d, long ldd, long strideD, long batchCount) {

  if ((m >= INT_MAX) || (n >= INT_MAX) || (k >= INT_MAX) || (lda >= INT_MAX) ||
      (ldb >= INT_MAX) || (ldc >= INT_MAX) || (batchCount >= INT_MAX))

  {
    AT_ERROR("Cublas_SgemmStridedBatched only supports m, n, k, lda, ldb, ldc, "
             "batchCount"
             "with the bound [val] <= %d",
             INT_MAX);
  }

  adjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);

  // gemm_switch_fp32accum(transa, transb, m, n, k, alpha, a, lda, strideA,
  //                       b, ldb, strideB, beta, c, ldc, strideC, batchCount);
  gemm_switch_fp32accum(transa, transb, m, n, k, alpha, a, lda, strideA, 
                        b, ldb, strideB, beta, c, ldc, strideC, d, ldd, strideD, batchCount);
}

} // namespace
