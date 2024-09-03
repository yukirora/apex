#include <torch/extension.h>
#include <torch/torch.h>
#include <vector>
#include <cublasLt.h>
#include <stdio.h>

template <typename T>
int linear_bias_forward_cuda(
		at::Tensor input,    T *weight,	           at::Tensor bias, 
		int in_features,     int batch_size,       int out_features, 
		at::Tensor output,   void *lt_workspace);

template <typename T>
int linear_bias_backward_cuda(
		T *input,            T *weight,            T *d_output, 
		int in_features,     int batch_size,       int out_features, 
		T *d_weight,         T *d_bias,            T *d_input,  
		void *lt_workspace);

template <typename T>
int linear_gelu_linear_forward_cuda(
		T *input,            T *weight1,           T *bias1, 
		T *weight2,          T *bias2,             int in_features, 
		int hidden_features, int batch_size,       int out_features, 
		T *output1,          T *output2,           T *gelu_in, 
		void *lt_workspace) ;

template <typename T>
int linear_gelu_linear_backward_cuda(
		T *input,            T *gelu_in,           T *output1, 
		T *weight1,          T *weight2,           T *d_output1, 
		T *d_output2,        int in_features,      int batch_size, 
		int hidden_features, int out_features,     T *d_weight1, 
		T *d_weight2,        T *d_bias1,           T *d_bias2, 
		T *d_input,          void *lt_workspace);



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

  //auto reserved_size = get_mlp_reserved_space(batch_size, num_layers, output_features.data());
  // create output/workspace tensor
  auto out = at::empty({batch_size, out_features}, input.scalar_type());
  auto lt_workspace = at::empty({1 << 22}, input.scalar_type()); // allocate fixed 4MB workspace for cublaslt for now, and this gets at least 4 MB
  //auto reserved_space = at::empty({reserved_size}, inputs[0].scalar_type());


  printf("linear_bias_forward:1\n");

  std::cout << "Input:\n"        << input        << std::endl;
  std::cout << "Intput Element Size: " << input.element_size() << std::endl;
  std::cout << "In Features: "  << in_features  << std::endl; 
  std::cout << "Input dtype: "  << input.type() << std::endl;

  std::cout << "Weight:\n"       << weight       << std::endl;
  std::cout << "Bias:\n"         << bias         << std::endl;

  std::cout << "Batch Size: "   << batch_size   << std::endl;
  std::cout << "In Features: "  << in_features  << std::endl;
  std::cout << "Out Features: " << out_features << std::endl;

  AT_DISPATCH_FLOATING_TYPES_AND2 (
	   at::ScalarType::Half, 
	   at::ScalarType::BFloat16,
	   input.scalar_type(), 
	   "linear_bias_forward", 
	   [&]{
	      linear_bias_forward_cuda<scalar_t> (
	             input,  weight.data_ptr<scalar_t>(), bias, in_features, batch_size, out_features, out,
                     (void*) (lt_workspace.data_ptr<scalar_t>())
              );
           }
  );
  printf("linear_bias_forward:2\n");

  std::cout << "Out:\n"        << out        << std::endl;
  std::cout << "Out Element Size: " << out.element_size() << std::endl;
  std::cout << "Intput Element Size: " << input.element_size() << std::endl;
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

  //auto reserved_size = get_mlp_reserved_space(batch_size, num_layers, output_features.data());
  // create output/workspace tensor
  auto d_weight     = at::empty({out_features, in_features}, input.scalar_type());
  auto d_input      = at::empty({batch_size,   in_features}, input.scalar_type());
  auto lt_workspace = at::empty({1 << 22},                   input.scalar_type()); //allocate 4MB 
  // auto reserved_space = at::empty({reserved_size}, inputs[0].type());
  
#if (defined(CUBLAS_VERSION) && CUBLAS_VERSION < 11600) || USE_ROCM
  auto d_bias = d_output.view({-1, out_features}).sum(0, false);
#else                                                                              
  auto d_bias = at::empty({out_features}, input.scalar_type());
#endif              

  AT_DISPATCH_FLOATING_TYPES_AND2 (
	 at::ScalarType::Half, 
	 at::ScalarType::BFloat16, 
	 input.scalar_type(), 
	 "linear_bias_backward", 
	 [&] {
	     linear_bias_backward_cuda<scalar_t> (
	 	 input.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(), d_output.data_ptr<scalar_t>(), in_features,
        	 batch_size, out_features, d_weight.data_ptr<scalar_t>(), d_bias.data_ptr<scalar_t>(), d_input.data_ptr<scalar_t>(),
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
  auto output1      = at::empty({batch_size, hidden_features}, input.scalar_type());
  auto gelu_in      = at::empty({batch_size, hidden_features}, input.scalar_type());
  auto output2      = at::empty({batch_size, out_features},    input.scalar_type());
  auto lt_workspace = at::empty({1 << 22}, input.scalar_type()); // allocate fixed 4MB workspace for cublaslt for now, and this gets at least 4 MB

  //auto reserved_space = at::empty({reserved_size}, inputs[0].type());

  AT_DISPATCH_FLOATING_TYPES_AND2 (
		  at::ScalarType::Half, 
		  at::ScalarType::BFloat16, 
		  input.scalar_type(), 
		  "linear_gelu_linear_forward", 
		  [&] {
		       linear_gelu_linear_forward_cuda<scalar_t>(
			       	       input.data_ptr<scalar_t>(),
			       	       weight1.data_ptr<scalar_t>(),
			       	       bias1.data_ptr<scalar_t>(),
			       	       weight2.data_ptr<scalar_t>(),
			       	       bias2.data_ptr<scalar_t>(),
			       	       in_features,
			       	       hidden_features,
			       	       batch_size,
			       	       out_features,
			       	       output1.data_ptr<scalar_t>(),
			       	       output2.data_ptr<scalar_t>(),
			       	       gelu_in.data_ptr<scalar_t>(),
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
  auto d_weight1    = at::empty({hidden_features,  in_features},     input.scalar_type());
  auto d_weight2    = at::empty({out_features,     hidden_features}, input.scalar_type());
  auto d_bias1      = at::empty({hidden_features},                   input.scalar_type());
  auto d_bias2      = at::empty({out_features},                      input.scalar_type());
  auto d_input      = at::empty({batch_size,       in_features},     input.scalar_type());
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
        			input.data_ptr<scalar_t>(),
        			gelu_in.data_ptr<scalar_t>(),
        			output1.data_ptr<scalar_t>(),
        			weight1.data_ptr<scalar_t>(),
        			weight2.data_ptr<scalar_t>(),
        			d_output1.data_ptr<scalar_t>(),
        			d_output2.data_ptr<scalar_t>(),
        			in_features,
        			batch_size,
        			hidden_features,
        			out_features,
        			d_weight1.data_ptr<scalar_t>(),
        			d_weight2.data_ptr<scalar_t>(),
        			d_bias1.data_ptr<scalar_t>(),
        			d_bias2.data_ptr<scalar_t>(),
        			d_input.data_ptr<scalar_t>(),
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

