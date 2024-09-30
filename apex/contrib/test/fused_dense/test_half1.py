from apex import fused_dense
import torch

batch_size   = 4
in_features  = 3
out_features = 2

#tst_dtype = torch.float8_e4m3
# tst_dtype = torch.float8_e5m2
tst_dtype = torch.float16

# I = torch.randn(batch_size, in_features, dtype=tst_dtype, device='cuda')
I = torch.tensor([[1., 2. , 3., 4.],
                  [1., 2. , 3., 4.],
                  [1., 2. , 3., 4.],
                  [1., 2. , 3., 4.],
                  [1., 2. , 3., 4.]],dtype=tst_dtype, device='cuda')

# W = torch.randn(out_features, in_features, dtype=tst_dtype, device='cuda')
W = torch.tensor([[1., 1. , 1. , 1. ],
                  [2., 2. , 2. , 2. ],
                  [3., 3. , 3. , 3. ]],dtype=tst_dtype, device='cuda')

# b = torch.randn(in_features, dtype=tst_dtype, device='cuda')
b = torch.tensor([1, 1, 1], dtype=tst_dtype, device='cuda')

print("Torch-A:\n", I)
print("Torch-B:\n", W)
print("Torch-b:\n", b)

C  = torch.matmul(I, W.t())+b
print("Torch-C:\n", C)

aC = fused_dense.fused_dense_function(I, W, b)
print("Torch-aC:\n", aC)
