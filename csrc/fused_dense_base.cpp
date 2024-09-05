#include <torch/extension.h>
#include <torch/torch.h>
#include <vector>
#include <cublasLt.h>
#include <stdio.h>

template <typename T>
int linear_bias_forward_cuda(
                at::Tensor  input,
                at::Tensor  weight,
                at::Tensor  bias,
                at::Tensor  output,
                int in_features, int batch_size, int out_features,
                void *lt_workspace);

template <typename T>
int linear_bias_backward_cuda(
                at::Tensor    input,
                at::Tensor    weight,
                at::Tensor    d_output,
                int           in_features,  int batch_size, int out_features,
                at::Tensor    d_weight,
                at::Tensor    d_bias,
                at::Tensor    d_input,
                void          *lt_workspace);

template <typename T>
int linear_gelu_linear_forward_cuda(
                at::Tensor input,
                at::Tensor weight1,
                at::Tensor bias1,
                at::Tensor weight2,
                at::Tensor bias2,
                int in_features, int hidden_features, int batch_size, int out_features,
                at::Tensor output1,
                at::Tensor output2,
                at::Tensor gelu_in,
                void *lt_workspace);

template <typename T>
int linear_gelu_linear_backward_cuda(
                at::Tensor input,
                at::Tensor gelu_in,
                at::Tensor output1,
                at::Tensor weight1,
                at::Tensor weight2,
                at::Tensor d_output1,
                at::Tensor d_output2,
                int in_features, int batch_size, int hidden_features, int out_features,
                at::Tensor d_weight1,
                at::Tensor d_weight2,
                at::Tensor d_bias1,
                at::Tensor d_bias2,
                at::Tensor d_input,
                void *lt_workspace);



/****************************************************************
 *
 *
 * ************************************************/
at::Tensor linear_bias_forward(
		at::Tensor input, 
		at::Tensor weight, 
		at::Tensor bias) 
{

  auto batch_size  = input.size(0);
  auto in_features = input.size(1);
  int out_features = weight.size(0);

  auto out = at::empty({batch_size, out_features}, torch::device(torch::kCUDA).dtype(input.scalar_type()));
  auto lt_workspace = at::empty({1 << 22}, input.scalar_type()); // allocate fixed 4MB workspace for cublaslt for now, and this gets at least 4 MB

  AT_DISPATCH_FLOATING_TYPES_AND2 (
	   at::ScalarType::Half, 
	   at::ScalarType::BFloat16,
	   input.scalar_type(), 
	   "linear_bias_forward", 
	   [&]{
	      linear_bias_forward_cuda<scalar_t> (
	             input,  
		     weight, 
		     bias,
		     out, 
		     in_features, batch_size, out_features,
                     (void*) (lt_workspace.data_ptr<scalar_t>())
              );
           }
  );
  return {out};
}


/****************************************************************
 *
 *
 * ************************************************/
std::vector<at::Tensor> linear_bias_backward( 
		at::Tensor input, 
		at::Tensor weight,  
		at::Tensor d_output) 
{
  auto batch_size   = input.size(0);
  auto in_features  = input.size(1);
  int  out_features = weight.size(0);

  auto d_weight     = at::empty({out_features, in_features}, torch::device(torch::kCUDA).dtype(input.scalar_type()));
  auto d_input      = at::empty({batch_size,   in_features}, torch::device(torch::kCUDA).dtype(input.scalar_type()));
  auto lt_workspace = at::empty({1 << 22},                   input.scalar_type()); //allocate 4MB 
  auto d_bias       = at::empty({out_features}, torch::device(torch::kCUDA).dtype(input.scalar_type()));

  AT_DISPATCH_FLOATING_TYPES_AND2 (
	 at::ScalarType::Half, 
	 at::ScalarType::BFloat16, 
	 input.scalar_type(), 
	 "linear_bias_backward", 
	 [&] {
	     linear_bias_backward_cuda<scalar_t> (
	 	 input, 
		 weight, 
		 d_output, 
		 in_features, batch_size, out_features, 
		 d_weight, 
		 d_bias, 
		 d_input,
		 (void*) (lt_workspace.data_ptr<scalar_t>())
		);
       	 }
   );
  return {d_input, d_weight, d_bias};
}

/****************************************************************
 *
 *
 * ************************************************/
std::vector<at::Tensor> linear_gelu_linear_forward(
		at::Tensor input, 	at::Tensor weight1, 	at::Tensor bias1, 
		at::Tensor weight2, 	at::Tensor bias2) 
{

  auto batch_size      = input.size(0);
  auto in_features     = input.size(1);
  int  hidden_features = weight1.size(0);
  int  out_features    = weight2.size(0);

  //auto reserved_size = get_mlp_reserved_space(batch_size, num_layers, output_features.data());

  // create output/workspace tensor
  auto output1      = at::empty({batch_size, hidden_features}, torch::device(torch::kCUDA).dtype(input.scalar_type())); 
  auto gelu_in      = at::empty({batch_size, hidden_features}, torch::device(torch::kCUDA).dtype(input.scalar_type()));
  auto output2      = at::empty({batch_size, out_features},    torch::device(torch::kCUDA).dtype(input.scalar_type()));
  auto lt_workspace = at::empty({1 << 22}, input.scalar_type()); // allocate fixed 4MB workspace for cublaslt for now, and this gets at least 4 MB

  //auto reserved_space = at::empty({reserved_size}, inputs[0].type());

  AT_DISPATCH_FLOATING_TYPES_AND2 (
		  at::ScalarType::Half, 
		  at::ScalarType::BFloat16, 
		  input.scalar_type(), 
		  "linear_gelu_linear_forward", 
		  [&] {
		       linear_gelu_linear_forward_cuda<scalar_t>(
			       	       input,
			       	       weight1,
			       	       bias1,
			       	       weight2,
			       	       bias2,
			       	       in_features,
			       	       hidden_features,
			       	       batch_size,
			       	       out_features,
			       	       output1,
			       	       output2,
			       	       gelu_in,
				       (void*) (lt_workspace.data_ptr<scalar_t>()));
                 }
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
  AT_DISPATCH_FLOATING_TYPES_AND2(
		  at::ScalarType::Half, 
		  at::ScalarType::BFloat16, 
		  input.scalar_type(), 
		  "linear_gelu_linear_backward", 
		  [&] {
    		      linear_gelu_linear_backward_cuda<scalar_t>(
        			input,
        			gelu_in,
        			output1,
        			weight1,
        			weight2,
        			d_output1,
        			d_output2,
        			in_features,
        			batch_size,
        			hidden_features,
        			out_features,
        			d_weight1,
        			d_weight2,
        			d_bias1,
        			d_bias2,
        			d_input,
        			(void*) (lt_workspace.data_ptr<scalar_t>())
			);
  		}
  );
  return {d_input, d_weight1, d_bias1, d_weight2, d_bias2};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("linear_bias_forward",         &linear_bias_forward,         "linear bias forward");
  m.def("linear_bias_backward",        &linear_bias_backward,        "linear bias backward");
  m.def("linear_gelu_linear_forward",  &linear_gelu_linear_forward,  "linear gelu linear forward");
  m.def("linear_gelu_linear_backward", &linear_gelu_linear_backward, "linear gelu linear backward");
}

