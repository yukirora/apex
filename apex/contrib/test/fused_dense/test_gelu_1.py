import torch
import torch.nn as nn
from apex import fused_dense

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TwoLayerFC(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(TwoLayerFC, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x

tst_dtype = torch.float16
batch_size   = 4
in_features  = 3
hidden_features = 3
out_features = 2

# Example usage:
model        = TwoLayerFC(in_features, hidden_features, out_features).to(device).half()


input_tensor = torch.tensor(
                 [[1., 2. , 3.],
                  [1., 2. , 3.],
                  [1., 2. , 3.],
                  [1., 2. , 3.]],dtype=tst_dtype, device='cuda').requires_grad_(True) 
# input_tensor = torch.randn(batch_size, in_features, dtype=tst_dtype, device=torch.device(device)).requires_grad_(True) 
output  = model(input_tensor) # Forward pass
loss    = output.mean()       # Compute loss (example)
loss.backward()               # Backward pass

print(output.shape)  # Should print torch.Size([32, 5])
print(output)
print(model.fc1.weight.grad) # Print gradients of the first layer's weights
print(model.fc2.weight.grad) # Print gradients of the first layer's weights

denseGlue = fused_dense.FusedDenseGeluDense(in_features, hidden_features, out_features)
denseGlue.to(dtype=tst_dtype)
denseGlue.to(device)

dense_output = denseGlue(input_tensor)

grad_dense_output  = torch.randn_like(dense_output).to(dtype=tst_dtype)

dense_output.backward(grad_dense_output)

print(dense_output.shape)
print(dense_output)
print(denseGlue.weight.grad)
print(denseGlue.weight2.grad)