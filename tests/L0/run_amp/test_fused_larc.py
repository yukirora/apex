import unittest
import os

import torch
import apex
from apex.parallel import fused_larc, LARC

class TestFusedLARC(unittest.TestCase):
    def setUp(self, max_abs_diff=1e-3, max_rel_diff=1e-6):
        self.max_abs_diff = max_abs_diff
        self.max_rel_diff = max_rel_diff
        torch.cuda.manual_seed(9876)

    def tearDown(self):
        pass

    def gen_param_optim(self, tensors, lr, weight_decay):
        ref_param = []
        tst_param = []
        for tensor in tensors:
            ref_param.append(torch.nn.Parameter(tensor.clone()))
            tst_param.append(torch.nn.Parameter(tensor.clone()))

        options = {'lr': lr, 'weight_decay': weight_decay}
        ref_optim = torch.optim.SGD(ref_param, **options)
        ref_optim = LARC.LARC(ref_optim)
        tst_optim = torch.optim.SGD(tst_param, **options)
        tst_optim = fused_larc.FusedLARC(tst_optim)

        return (ref_param, tst_param, ref_optim, tst_optim)

    def gen_grad(self, ref_param, tst_param):
        for p_ref, p_tst in zip(ref_param, tst_param):
            tensor = torch.rand_like(p_ref)
            p_ref.grad = tensor.clone()
            p_tst.grad = tensor.clone()

    def check_tensors(self, ref_param, tst_param):
        for p_ref, p_tst in zip(ref_param, tst_param):
            if torch.equal(p_ref, p_tst):
                continue

            self.assertTrue(torch.allclose(
                p_ref, p_tst, rtol=self.max_rel_diff, atol=self.max_abs_diff))

    def gen_single_type_test(self, param_type=torch.float):
        nelem = [1024, 32768, 278011]
        weight_decay = [0, 0.001]
        lr = [1e-4, 1]

        for n in nelem:
            tensor = torch.rand(n, dtype=param_type, device='cuda')
            for wd in weight_decay:
                for l in lr:
                    ref_param, tst_param, ref_optim, tst_optim = \
                        self.gen_param_optim([tensor], l, wd)

                    self.gen_grad(ref_param, tst_param)
                    ref_optim.step()
                    tst_optim.step()
                    self.check_tensors(ref_param, tst_param)

    def test_float(self):
        self.gen_single_type_test(param_type=torch.float)

    def test_half(self):
        self.gen_single_type_test(param_type=torch.float16)

    def test_multi_params(self):
        sizes = [[4096, 1024], [4096], [4096, 2048], [32320, 1024], [1]]
        weight_decay = [0, 0.001]
        lr = [1e-4, 1]

        for wd in weight_decay:
            for l in lr:
                tensors = []
                for size in sizes:
                    tensors.append(torch.rand(size, dtype=torch.float, device='cuda'))
                ref_param, tst_param, ref_optim, tst_optim = \
                    self.gen_param_optim(tensors, l, wd)

                self.gen_grad(ref_param, tst_param)
                ref_optim.step()
                tst_optim.step()
                self.check_tensors(ref_param, tst_param)

if __name__ == '__main__':
    script_path = os.path.dirname(os.path.realpath(__file__))
    unittest.main()
