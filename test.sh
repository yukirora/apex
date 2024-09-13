#!/bin/bash -x


export HIPBLASLT_LOG_LEVEL=5
export HIPBLASLT_LOG_MASK=0xff
export TENSILE_DB=0x40
export HIPBLASLT_LOG_FILE=hipblaslt_bgrad.log

python apex/contrib/test/fused_dense/test_fused_dense_1.py

exit 0

#export HIPBLASLT_LOG_FILE=hipblaslt_float32.log
PYTORCH_TEST_WITH_ROCM=1 python apex/contrib/test/fused_dense/test_fused_dense_1.py -k test_fused_dense_float32_cuda_float32

#export HIPBLASLT_LOG_FILE=hipblaslt_float16.log
PYTORCH_TEST_WITH_ROCM=1 python apex/contrib/test/fused_dense/test_fused_dense_1.py -k test_fused_dense_float16_cuda_float16

# export HIPBLASLT_LOG_FILE=hipblaslt_bfloat16.log
PYTORCH_TEST_WITH_ROCM=1 python apex/contrib/test/fused_dense/test_fused_dense_1.py -k test_fused_dense_bfloat16_cuda_bfloat16

