from apex import fused_dense
import torch

in_features  = 3
out_features = 2

# I = torch.randn(in_features, out_features, dtype=torch.float, device='cuda')
I = torch.tensor([[1., 2. , 3., 4.], 
                  [1., 2. , 3., 4.],
                  [1., 2. , 3., 4.],
                  [1., 2. , 3., 4.],
                  [1., 2. , 3., 4.]],dtype=torch.float, device='cuda')

# W = torch.randn(out_features, in_features, dtype=torch.float, device='cuda')
W = torch.tensor([[1., 2. , 3.],
                  [1., 2. , 3.],
                  [1., 2. , 3.],
                  [1., 2. , 3.]],dtype=torch.float, device='cuda')

b = torch.tensor([1.8597, 1.4086, 0.1826], dtype=torch.float, device='cuda')

print("Torch-A:\n", I)
print("Torch-B:\n", W)
print("Torch-b:\n", b)

C  = torch.matmul(I, W)+b
print("Torch-C:\n", C)

aC = fused_dense.fused_dense_function(I, W, b)
print("Torch-aC:\n", aC)
# torch.testing.assert_close(C,  aC,  atol=1e-3, rtol=1e-3, equal_nan=True)
