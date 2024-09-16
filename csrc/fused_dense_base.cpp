#include <torch/extension.h>
#include <torch/torch.h>
#include <vector>
#include <cublasLt.h>
#include <stdio.h>

template <typename T>
int linear_bias_forward_cuda(at::Tensor input, at::Tensor weight, at::Tensor bias, at::Tensor output);

template <typename T>
int linear_bias_backward_cuda( at::Tensor    input,    at::Tensor    weight,  at::Tensor    d_output,
                               at::Tensor    d_weight, at::Tensor    d_bias,  at::Tensor    d_input);

template <typename T>
int linear_gelu_linear_forward_cuda(
                at::Tensor input,  at::Tensor weight1,  at::Tensor bias1,    at::Tensor weight2,
                at::Tensor bias2,  at::Tensor output1,  at::Tensor output2,  at::Tensor gelu_in);

template <typename T>
int linear_gelu_linear_backward_cuda(
                at::Tensor input,     at::Tensor gelu_in,   at::Tensor output1,    at::Tensor weight1,
                at::Tensor weight2,   at::Tensor d_output1, at::Tensor d_output2,  at::Tensor d_weight1,
                at::Tensor d_weight2, at::Tensor d_bias1,   at::Tensor d_bias2,    at::Tensor d_input);



/****************************************************************
 *
 *
 * ************************************************/
at::Tensor linear_bias_forward( at::Tensor input, at::Tensor weight, at::Tensor bias) 
{
  auto out = at::zeros({input.size(0), weight.size(1)}, torch::device(torch::kCUDA).dtype(input.scalar_type()));

  // std::cout << "Input:\n" << input << "\nweight:\n" << weight << "\nbias:\n" << bias <<  std::endl;

  AT_DISPATCH_FLOATING_TYPES_AND2 ( at::ScalarType::Half,  at::ScalarType::BFloat16, input.scalar_type(), "linear_bias_forward", 
	   [&]{ linear_bias_forward_cuda<scalar_t> ( input,  weight, bias, out); }
  );

  // linear_bias_forward_cuda<c10::Float8_e5m2fnuz>(input, weight, bias, out);
  // linear_bias_forward_cuda<c10::Float8_e4m3fnuz>(input, weight, bias, out);

  return {out};
}


/****************************************************************
 * In the backward pass, we compute the gradients of the loss with respect to input, weight, and bias. 
 * The key matrix operations are:
 *  1. Gradient of Input   (dX): dX  = dY ⋅ WT: Pass `dY`  as matrix `A`, `W`  as matrix `B`, and compute the result into `dX`.
 *  2. Gradient of Weights (dW): dWi = XT ⋅ dY: Pass `X^T` as matrix `A`  `dY` as matrix `B`, and compute the result into `dW`.
 *  3. Gradient of Bias    (db): db=sum(dY)
 *
 * ************************************************/
std::vector<at::Tensor> linear_bias_backward( at::Tensor input, at::Tensor weight,  at::Tensor d_output) 
{

  auto d_input   = at::zeros_like(input, torch::device(torch::kCUDA).dtype(input.scalar_type()));
  auto d_weight  = at::zeros_like(weight, torch::device(torch::kCUDA).dtype(input.scalar_type()));
  auto d_bias    = at::zeros(weight.size(1), torch::device(torch::kCUDA).dtype(input.scalar_type()));

  AT_DISPATCH_FLOATING_TYPES_AND2 ( at::ScalarType::Half,  at::ScalarType::BFloat16,  input.scalar_type(),  "linear_bias_backward",
        [&] { linear_bias_backward_cuda<scalar_t> (input, weight, d_output,  d_weight,  d_bias,  d_input); }
  );

 /*
 linear_bias_backward_cuda<c10::Float8_e5m2fnuz> (input, weight, d_output,  d_weight,  d_bias,  d_input);
 linear_bias_backward_cuda<c10::Float8_e4m3fnuz> (input, weight, d_output,  d_weight,  d_bias,  d_input);
 */
  return {d_input, d_weight, d_bias};
}

/****************************************************************
 *
 *
 * ************************************************/
std::vector<at::Tensor> linear_gelu_linear_forward(at::Tensor input, at::Tensor weight1, at::Tensor bias1, at::Tensor weight2, at::Tensor bias2) 
{
  auto batch_size      = input.size(0);
  auto in_features     = input.size(1);
  int  hidden_features = weight1.size(0);
  int  out_features    = weight2.size(0);

  //auto reserved_size = get_mlp_reserved_space(batch_size, num_layers, output_features.data());

  // create output/workspace tensor
  auto output1      = at::zeros({batch_size, hidden_features}, torch::device(torch::kCUDA).dtype(input.scalar_type())); 
  auto gelu_in      = at::zeros({batch_size, hidden_features}, torch::device(torch::kCUDA).dtype(input.scalar_type()));
  auto output2      = at::zeros({batch_size, out_features},    torch::device(torch::kCUDA).dtype(input.scalar_type()));
  auto lt_workspace = at::zeros({1 << 22}, input.scalar_type()); // allocate fixed 4MB workspace for cublaslt for now, and this gets at least 4 MB

  //auto reserved_space = at::empty({reserved_size}, inputs[0].type());

  AT_DISPATCH_FLOATING_TYPES_AND2 ( at::ScalarType::Half,  at::ScalarType::BFloat16,  input.scalar_type(), "linear_gelu_linear_forward", 
	[&] { linear_gelu_linear_forward_cuda<scalar_t>(input, weight1, bias1, weight2, bias2, output1, output2, gelu_in); }
  );
  return {output1, output2, gelu_in};
}

/****************************************************************
 *
 *
 * ************************************************/
std::vector<at::Tensor> linear_gelu_linear_backward(
		at::Tensor input, 
		at::Tensor gelu_in, 
		at::Tensor output1, 
		at::Tensor weight1, 
		at::Tensor weight2, 
		at::Tensor d_output2) {

  auto batch_size  = input.size(0);
  auto in_features = input.size(1);
  int hidden_features = weight1.size(0);
  int out_features    = weight2.size(0);

  //auto reserved_size = get_mlp_reserved_space(batch_size, num_layers, output_features.data());

  // create output/workspace tensor
  auto d_weight1    = at::empty({hidden_features,  in_features},     torch::device(torch::kCUDA).dtype(input.scalar_type())); 
  auto d_weight2    = at::empty({out_features,     hidden_features}, torch::device(torch::kCUDA).dtype(input.scalar_type()));
  auto d_bias1      = at::empty({hidden_features},                   torch::device(torch::kCUDA).dtype(input.scalar_type()));
  auto d_bias2      = at::empty({out_features},                      torch::device(torch::kCUDA).dtype(input.scalar_type()));
  auto d_input      = at::empty({batch_size,       in_features},     torch::device(torch::kCUDA).dtype(input.scalar_type()));
  auto d_output1    = at::empty({batch_size,       hidden_features}, input.scalar_type());
  auto lt_workspace = at::empty({1 << 22}, input.scalar_type()); // allocate fixed 4MB workspace for cublaslt for now, and this gets at least 4 MB

  //auto reserved_space = at::empty({reserved_size}, inputs[0].type());
  AT_DISPATCH_FLOATING_TYPES_AND2( at::ScalarType::Half,  at::ScalarType::BFloat16,  input.scalar_type(),  "linear_gelu_linear_backward", 
	[&] { linear_gelu_linear_backward_cuda<scalar_t>(input, gelu_in, output1, weight1, weight2, d_output1, d_output2,
        			d_weight1, d_weight2, d_bias1,	d_bias2, d_input);}
  );
  return {d_input, d_weight1, d_bias1, d_weight2, d_bias2};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("linear_bias_forward",         &linear_bias_forward,         "linear bias forward");
  m.def("linear_bias_backward",        &linear_bias_backward,        "linear bias backward");
  m.def("linear_gelu_linear_forward",  &linear_gelu_linear_forward,  "linear gelu linear forward");
  m.def("linear_gelu_linear_backward", &linear_gelu_linear_backward, "linear gelu linear backward");
}

