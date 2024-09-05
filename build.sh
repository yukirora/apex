#!/bin/bash -x
export PYTORCH_ROCM_ARCH=gfx90a
python setup.py develop --cuda_ext --cpp_ext
cp build/lib.linux-x86_64-cpython-310/fused_dense_cuda.cpython-310-x86_64-linux-gnu.so /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fused_dense_cuda.cpython-310-x86_64-linux-gnu.so
python apex/contrib/test/fused_dense/test_half.py
