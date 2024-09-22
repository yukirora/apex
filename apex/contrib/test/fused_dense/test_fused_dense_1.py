import unittest
import os

import torch
from torch.testing._internal import common_utils
from torch.testing._internal.common_device_type import instantiate_device_type_tests

SKIP_TEST = None
try:
    from apex import fused_dense
except ImportError as e:
    SKIP_TEST = e


@unittest.skipIf(SKIP_TEST, f"{SKIP_TEST}")
class FusedDenseTest(common_utils.TestCase):

    def _test_fused_dense(self, dtype, seed=0):

        os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
        torch.manual_seed(seed)

        # seq_length = 4 # 512
        # sequences  = 3 # 3
        # hidden_dim = 8 # 1024
        batch_size   = 4       
        in_features  = 3 
        out_features = 2

        # --------------------------------------------------------------------------------------------------
        #  Setup
        # --------------------------------------------------------------------------------------------------

        ref_inputs = torch.randn(batch_size,in_features, dtype=dtype, device=torch.device("cuda")).requires_grad_(True) 

        tst_inputs = ref_inputs.clone().detach().requires_grad_(True) 

        # Create dense
        # self.weight = nn.Parameter(torch.randn(in_features, out_features))
        # self.bias   = nn.Parameter(torch.randn(out_features))
        dense = fused_dense.FusedDense(in_features, out_features) 
        dense.to(dtype=dtype)
        dense.cuda()

        # --------------------------------------------------------------------------------------------------
        #  Farward pass
        # --------------------------------------------------------------------------------------------------

        y_ref = torch.matmul(ref_inputs, dense.weight.t())+dense.bias
        y_tst = dense(tst_inputs)
        torch.testing.assert_close(y_ref,  y_tst,  atol=1e-3, rtol=1e-3, equal_nan=True)

        # --------------------------------------------------------------------------------------------------
        #  Backward pass
        #    dX  = dY ⋅ WT
        #    dW  = XT ⋅ dY and db=sum(dY)
        # --------------------------------------------------------------------------------------------------

        dy  = torch.randn_like(y_tst).to(dtype=dtype)
        dx_ref = torch.matmul(dy, dense.weight.clone())
        # fused_dense_cuda.linear_bias_backward(input, weight.t(), grad_output)
        y_tst.backward(dy)

        print("dx_ref Tensor:\n",   dx_ref)
        print("tst_inputs.grad Tensor:\n", tst_inputs.grad)
        torch.testing.assert_close(dx_ref, tst_inputs.grad, atol=1e-3, rtol=1e-3, equal_nan=True)

        dw_ref = torch.matmul(ref_inputs.t(), dy)
        print("dw_ref Tensor:\n",   dw_ref)
        print("dense.weight.grad Tensor:\n", dense.weight.grad)
        # torch.testing.assert_close(dw_ref, dense.weight.grad., atol=1e-3, rtol=1e-3, equal_nan=True)

        db_ref = dy.sum(0, False)
        print("db_ref Tensor:\n",   db_ref)
        print("dense.bias.grad Tensor:\n",   dense.bias.grad)
        # torch.testing.assert_close(db_ref, dense.bias.grad, atol=1e-3, rtol=1e-3, equal_nan=True)

        print("********************************************************************")

    # @common_utils.parametrize("dtype", [torch.half, torch.float, torch.bfloat16, torch.float8_e4m3fn])
    @common_utils.parametrize("dtype", [torch.half])
    def test_fused_dense(self, dtype):
        self._test_fused_dense(dtype)


instantiate_device_type_tests(FusedDenseTest, globals(), only_for=("cuda",))

if __name__ == "__main__":
    common_utils.run_tests()
