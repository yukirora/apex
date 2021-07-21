import numpy as np
import random
import torch
import torch.nn as nn
import apex
from argparse import ArgumentParser
from apex.contrib.groupbn.batch_norm import BatchNorm2d_NHWC


def parse_args():
    parser = ArgumentParser(description='Pytorch batchnorm microbenchmark')
    parser.add_argument('--batch-size', type=int, default=120, help='Batch size')
    parser.add_argument('--rtol', type=float, default=1e-3, help='The relative tolerance parameter')
    parser.add_argument('--atol', type=float, default=1e-3, help='The absolute tolerance parameter')
    parser.add_argument('--get_time', type=bool, default=True, help='Whether to enable the time taken by the op')
    parser.add_argument('--mode', type=str, default='bn', choices=['bn', 'bn_relu', 'bn_add_relu'], help='Execution mode')
    parser.add_argument('--ref_device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to run the golden model')
    parser.add_argument('--seed', type=int, default=5, help='Random seed')

    return parser.parse_args()

def validate(tensors, output_ref, output_test, is_torch=True):
    if is_torch:
        output_ref = output_ref.detach().cpu().numpy()
        output_test = output_test.detach().cpu().numpy()
    print('>>> tensor_size\t{}'.format(tensors))
    print("sum_output_ref {}, isnan {}".format(np.sum(output_ref, dtype=float), np.isnan(output_ref).any()))
    print("sum_output_test {}, isnan {}".format(np.sum(output_test, dtype=float), np.isnan(output_test).any()))
    ret = np.array_equal(output_ref, output_test)
    if not ret:
        ret_allclose = np.allclose(output_ref, output_test, rtol=args.rtol, atol=args.atol, equal_nan = True)
        print('{}\tshape {}\tidentical {}\tclose {}'.format('cpu/gpu', tensors, ret, ret_allclose))
        output_ref = output_ref.flatten()
        output_test = output_test.flatten()
        if not ret:
            sub = np.absolute(output_ref - output_test)
            rel = np.divide(sub, np.absolute(output_ref), where=(sub!=0) & (output_ref!=0))
            print('max_diff {}, max_rel_diff {}, norm_diff {}'.format(np.max(sub), np.max(rel), np.average(sub)))
            max_abs_idx = np.argmax(sub)
            max_rel_idx = np.argmax(rel)
            print('max_abs pair {} {}'.format(output_ref[max_abs_idx], output_test[max_abs_idx]))
            print('max_rel pair {} {}'.format(output_ref[max_rel_idx], output_test[max_rel_idx]))

    if ret or ret_allclose:
        print("Result= PASS")
    else:
        print("Result= FAIL")

def generate_uniform_tensor(size, np_dtype, device):
    array = None
    while array is None or np.isnan(array).any():
        array = np.random.uniform(low=-1.0, high=1.0, size=size).astype(np_dtype)
    return torch.from_numpy(array).to(device)

class ResidualAdd(nn.Module):
    def __init__(self, y):
        super(ResidualAdd, self).__init__()
        self.y = y

    def forward(self, x):
        return torch.add(x, self.y) if self.y is not None else x

def to_channels_last(tensor):
    return tensor.permute(0, 2, 3, 1).contiguous()

def to_channels_first(tensor):
    return tensor.permute(0, 3, 1, 2).contiguous()

def run_pyt(fn, input, device):
    if device == 'cpu':
        out = fn(input.cpu().float())
        if out is not None:
            return out.half()
    else:
        return fn(input)

if __name__ == '__main__':
    args = parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    tensor_sizes = [
            (args.batch_size, 64, 150, 150),
            (args.batch_size, 64, 75, 75),
            (args.batch_size, 128, 38, 38),
            (args.batch_size, 256, 38, 38)
            ]

    for i in range(len(tensor_sizes)):
        print(args.mode)
        tensor_size = tensor_sizes[i]
        num_channels = tensor_size[1]

        input_data = generate_uniform_tensor(tensor_size, np.float16, 'cuda')
        np.save('input.npy', input_data.detach().cpu().numpy())
        input_data.requires_grad = True

        residual_data = None
        gbn_residual_data = None
        if args.mode == 'bn':
            fuse_relu = False
        else:
            fuse_relu = True
            if args.mode == 'bn_add_relu':
                residual_data = generate_uniform_tensor(tensor_size, np.float16, 'cuda')
                gbn_residual_data = to_channels_last(residual_data)

        ###################### Run torch.nn.BatchNorm2d ######################
        layers = []
        layers.append(nn.BatchNorm2d(num_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        if args.mode == 'bn_add_relu':
            layers.append(ResidualAdd(residual_data))
        if args.mode != 'bn':
            layers.append(nn.ReLU(inplace=False))
        batchnorm_model = nn.Sequential(*layers).to(args.ref_device)

        # Forward
        bn_output = run_pyt(batchnorm_model, input_data, args.ref_device)

        # Backward
        bn_grad = generate_uniform_tensor(input_data.shape, np.float16, 'cuda')
        run_pyt(bn_output.backward, bn_grad, args.ref_device)
        bn_output_grad = input_data.grad.detach().clone().cpu().half()

        ######## Run apex.contrib.groupbn.batch_norm.BatchNorm2d_NHWC ########
        group_batchnorm = BatchNorm2d_NHWC(num_channels, fuse_relu=fuse_relu, bn_group=1).cuda()

        # Forward
        gbn_input = torch.from_numpy(np.load('input.npy')).cuda().half()
        gbn_input.requires_grad = True
        gbn_input_data = to_channels_last(gbn_input)
        gbn_output = group_batchnorm(gbn_input_data, gbn_residual_data)

        # Bacward
        gbn_grad = to_channels_last(bn_grad)
        gbn_output.backward(gbn_grad)
        torch.cuda.synchronize()
        gbn_output_grad = None

        gbn_output = to_channels_first(gbn_output)
        gbn_output_grad = gbn_input.grad.detach().clone().cpu()

        ########################## Validate results ##########################
        print('Validate activation tensor')
        validate(tensor_size, bn_output, gbn_output)
        print('Validate gradient tensor')
        validate(bn_output_grad.shape, bn_output_grad, gbn_output_grad)

        with torch.no_grad():
            batchnorm_model.eval()
            group_batchnorm.training = False

            bn_output = run_pyt(batchnorm_model, input_data, args.ref_device)
            gbn_output = group_batchnorm(gbn_input_data, gbn_residual_data)
            gbn_output = to_channels_first(gbn_output)
            print('Validate inference')
            validate(tensor_size, bn_output, gbn_output)

        print("===========================================================================")
