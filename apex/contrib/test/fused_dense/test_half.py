from apex import fused_dense
import torch
h = torch.randn(3, 4, dtype=torch.float16, device='cuda')
w = torch.rand(3, 4, dtype=torch.float16, device='cuda')
b = torch.randn(4, dtype=torch.float16, device='cuda')
c = fused_dense.fused_dense_function(h, w, b)

