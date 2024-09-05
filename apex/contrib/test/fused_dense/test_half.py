from apex import fused_dense
import torch
h = torch.randn(3, 4, dtype=torch.float16, device='cuda')
w = torch.rand(4, 3, dtype=torch.float16, device='cuda')
b = torch.randn(4, dtype=torch.float16, device='cuda')
c = fused_dense.fused_dense_function(h, w, b)

h = torch.randn(3, 4, dtype=torch.float, device='cuda')
w = torch.rand(4, 3, dtype=torch.float, device='cuda')
b = torch.randn(4, dtype=torch.float, device='cuda')
c = fused_dense.fused_dense_function(h, w, b)

h = torch.randn(3, 4, dtype=torch.bfloat16, device='cuda')
w = torch.rand(4, 3, dtype=torch.bfloat16, device='cuda')
b = torch.randn(4, dtype=torch.bfloat16, device='cuda')
c = fused_dense.fused_dense_function(h, w, b)

