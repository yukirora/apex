import torch
import re
import argparse
from itertools import product
from torch.utils.benchmark import Timer, Compare
try:
    from apex.normalization import FusedLayerNorm
except ImportError:
    raise AssertionError("Please build ROCm Apex first: pip install -v --install-option=\"--cpp_ext\"--install-option=\"--cuda_ext\"\'git+https://github.com/ROCmSoftwarePlatform/apex.git\'")

def process_shape_list(shape_list):
    temp = re.findall(r'\d+', shape_list)
    res = list(map(int, temp))
    return res

def ln_fwd(ln, X):
    ln(X)

def ln_fwdbwd(ln, X, gO):
    X.grad=None
    ln.zero_grad(set_to_none=True)
    out = ln(X)
    out.backward(gO)

def layer_norm_benchmark():
    results = []
    m_list = process_shape_list(args.m_list)
    n_list = process_shape_list(args.n_list)

    for mi, ni in product(m_list, n_list):
        for dtype in (torch.half, torch.float, ):
            if args.torch_layernorm:
                if dtype == torch.half:
                    ln = torch.nn.LayerNorm((ni,)).half().to("cuda")
                else:
                    ln = torch.nn.LayerNorm((ni,)).to("cuda")
            else:
                if dtype == torch.half:
                    ln = FusedLayerNorm((ni,)).half().to("cuda")
                else:
                    ln = FusedLayerNorm((ni,)).to("cuda")

            X = torch.randn(mi, ni, device="cuda", dtype=dtype, requires_grad=True)
            gO = torch.rand_like(X)

            tfwd = Timer(
                    stmt='ln_fwd(ln, X)',
                    label="ln",
                    sub_label=f"{mi:5}, {ni:5}",
                    description=f"fwd, {dtype}",
                    globals={'ln': ln, 'X': X},
                    setup='from __main__ import ln_fwd',
                    )
            tfwdbwd = Timer(
                    stmt='ln_fwdbwd(ln, X, gO)',
                    label="ln",
                    sub_label=f"{mi:5}, {ni:5}",
                    description=f"fwdbwd, {dtype}",
                    globals={'ln': ln, 'X': X, 'gO': gO},
                    setup='from __main__ import ln_fwdbwd',
                    )
            for t in (tfwd, tfwdbwd):
                results.append(t.blocked_autorange())
            print(ni, end='\r')

    c = Compare(results)
    c.print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--torch-layernorm', type=bool, default=False, help="Default option is apex.normalization.FusedLayerNorm")
    parser.add_argument("--m-list", type=str, required=True, default="", help="List of the fisrt dimension (m) of input tensor for LayerNorm using comma to seperate different shapes, e.g. 32,64,128")
    parser.add_argument("--n-list", type=str, required=True, default="", help="List of the fisrt dimension (m) of input tensor for LayerNorm to seperate different shapes, e.g. 4,8,16")
    args = parser.parse_args()
    layer_norm_benchmark()
