/*
 * Orginially From Baidu Research DeepBench.
 * Modifed to test Dawn
 * Author: Yanzhao Wu
 */

#include <iomanip>
#include <memory>
#include <chrono>
#include <vector>
#include <tuple>
#include <chrono>

#include <cuda.h>
#include <cudnn.h>
#include <curand.h>

#include <thrust/device_ptr.h>
#include <thrust/fill.h>

#include <cudautility.hpp>

#include "tensor.h"
#include "cudnn_helper.h"
#include "conv_problems.h"

/*
Usage:

The default precision is set based on the architecture and mode.

By default, the program runs the benchmark in training mode.

./conv_bench

To run inference mode, use the following command:

./conv_bench inference


To change the precision for training/inference, use:

./conv_bench train <precision>
./conv_bench inference <precision>

Supported precision types:

For Maxwell GPUS: 
float for training and inference

For Pascal GPUS:
float, half for training
float, half, int8 for inference

*/

// T1 is used as the data type for inputs, weights and outputs. 
// T2 is used to describe the compute precision. This is used in inference mode in the INT8_CONFIG
template <typename T1, typename T2>
class cudnnCNN {
    TensorDescriptor4d<T1> x_desc_;
    TensorDescriptor4d<T1> h_desc_;

    FilterDescriptor4d<T1> w_desc_;

    std::vector<int> output_dims_;
    int num_repeats_;

    size_t fwd_workspace_size_;
    size_t bwd_inputs_workspace_size_;
    size_t bwd_params_workspace_size_;

    Tensor<float> fwd_workspace_;
    Tensor<float> bwd_inputs_workspace_;
    Tensor<float> bwd_params_workspace_;

    cudnnConvolutionFwdAlgo_t fwd_algo_;
    cudnnConvolutionBwdDataAlgo_t bwd_inputs_algo_;
    cudnnConvolutionBwdFilterAlgo_t bwd_params_algo_;

    const float alpha_ = 1.f;
    const float beta_  = 0.f;

    ConvolutionDescriptor<T2> conv_desc_;
    CudnnHandle cudnn_handle_;

public:

    cudnnCNN(int w, int h, int c, int n, int k, int r, int s,
             int pad_w, int pad_h, int wstride, int hstride)
             :
        cudnn_handle_(),
        conv_desc_(pad_h, pad_w, hstride, wstride)
    {
        int out_h, out_w, out_c, out_n;

        cudnnTensorFormat_t format;
        // For int8 inference, the supported format is NHWC
        if (std::is_same<T1, uint8_t>::value) {
            format = CUDNN_TENSOR_NHWC;
        } else {
            format = CUDNN_TENSOR_NCHW;
        }

        x_desc_ = TensorDescriptor4d<T1>(format, n, c, h, w);
        w_desc_ = FilterDescriptor4d<T1>(format, k, c, r, s);

        // Get output dimensions
        CHECK_CUDNN_ERROR(cudnnGetConvolution2dForwardOutputDim(conv_desc_.desc(),
                                                                x_desc_.desc(),
                                                                w_desc_.desc(),
                                                                &out_n,
                                                                &out_c,
                                                                &out_h,
                                                                &out_w));

        h_desc_ = TensorDescriptor4d<T1>(format, out_n, out_c, out_h, out_w);

        output_dims_ = {out_w, out_h, out_c, out_n};

       // Pick forward convolution algorithm
        cudnnConvolutionFwdAlgoPerf_t fwd_perf;
        int ret_count;

        if (std::is_same<T1, uint8_t>::value) {
            //Note: cuDNN only supports IMPLICIT_PRECOMP_GEMM for int8 data type.
            fwd_algo_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
        } else {
            CHECK_CUDNN_ERROR(cudnnFindConvolutionForwardAlgorithm(cudnn_handle_.handle(),
                                                                   x_desc_.desc(),
                                                                   w_desc_.desc(),
                                                                   conv_desc_.desc(),
                                                                   h_desc_.desc(),
                                                                   1,
                                                                   &ret_count,
                                                                   &fwd_perf));
            fwd_algo_ = fwd_perf.algo;
        }
        if (std::is_same<T1, uint8_t>::value) {
            //Note: cudnn workspace size function doesn't work for INT8_CONFIG
            fwd_workspace_size_= 1073741824;
        } else {
            // Set fwd workspace size
            CHECK_CUDNN_ERROR(cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle_.handle(),
                                                                      x_desc_.desc(),
                                                                      w_desc_.desc(),
                                                                      conv_desc_.desc(),
                                                                      h_desc_.desc(),
                                                                      fwd_algo_,
                                                                      &fwd_workspace_size_));
        }

        fwd_workspace_ = zeros<float>(std::vector<int>{static_cast<int>(fwd_workspace_size_ / sizeof(float)), 1});

		cudnnConvolutionBwdFilterAlgoPerf_t filter_perf;

		if (std::is_same<T1, uint8_t>::value) {

			fwd_algo_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;

		}
		CHECK_CUDNN_ERROR(cudnnFindConvolutionBackwardFilterAlgorithm(cudnn_handle_.handle(),
																	 x_desc_.desc(),
																	 h_desc_.desc(),
																	 conv_desc_.desc(),
																	 w_desc_.desc(),
																	 1,
																	 &ret_count,
																	 &filter_perf));
		bwd_params_algo_ = filter_perf.algo;

		// Backward params workspace
		CHECK_CUDNN_ERROR(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn_handle_.handle(),
																		 x_desc_.desc(),
																		 h_desc_.desc(),
																		 conv_desc_.desc(),
																		 w_desc_.desc(),
																		 bwd_params_algo_,
																		 &bwd_params_workspace_size_));



		bwd_params_workspace_ = zeros<float>(std::vector<int>{static_cast<int>(bwd_params_workspace_size_ / sizeof(float)), 1});

		cudnnConvolutionBwdDataAlgoPerf_t data_perf;
		CHECK_CUDNN_ERROR(cudnnFindConvolutionBackwardDataAlgorithm(cudnn_handle_.handle(),
																	w_desc_.desc(),
																	h_desc_.desc(),
																	conv_desc_.desc(),
																	x_desc_.desc(),
																	1,
																	&ret_count,
																	&data_perf));
		bwd_inputs_algo_ = data_perf.algo;

		CHECK_CUDNN_ERROR(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle_.handle(),
																	   w_desc_.desc(),
																	   h_desc_.desc(),
																	   conv_desc_.desc(),
																	   x_desc_.desc(),
																	   bwd_inputs_algo_,
																	   &bwd_inputs_workspace_size_));

		bwd_inputs_workspace_ = zeros<float>(std::vector<int>{static_cast<int>(bwd_inputs_workspace_size_ / sizeof(float)), 1});
    }

    std::vector<int> get_output_dims() { return output_dims_; }

    std::string get_fwd_algo_string() {
        if (fwd_algo_ == CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM)
            return "IMPLICIT_GEMM";
        else if (fwd_algo_ == CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM)
            return "IMPLICIT_PRECOMP_GEMM";
        else if (fwd_algo_ == CUDNN_CONVOLUTION_FWD_ALGO_GEMM) 
            return "GEMM";
        else if (fwd_algo_ == CUDNN_CONVOLUTION_FWD_ALGO_DIRECT)
            return "DIRECT";
        else if (fwd_algo_ == CUDNN_CONVOLUTION_FWD_ALGO_FFT)
            return "FFT";
        else if (fwd_algo_ == CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING)
            return "FFT_TILING";
        else if (fwd_algo_ == CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD)
            return "WINOGRAD";
        else if (fwd_algo_ == CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED)
            return "WINOGRAD_NONFUSED";
        else {
            std::stringstream ss;
            ss << "Illegal algorithm passed to get_fwd_algo_string. Algo: " << fwd_algo_ << std::endl;
            throw std::runtime_error(ss.str());
        }
    }


    void forward(Tensor<T1> x, Tensor<T1> filter, Tensor<T1> h) {

        // Convolution forward.
        CHECK_CUDNN_ERROR(cudnnConvolutionForward(cudnn_handle_.handle(),
                                                  &alpha_,
                                                  x_desc_.desc(),
                                                  x.begin(),
                                                  w_desc_.desc(),
                                                  filter.begin(),
                                                  conv_desc_.desc(),
                                                  fwd_algo_,
                                                  fwd_workspace_.begin(),
                                                  fwd_workspace_size_,
                                                  &beta_,
                                                  h_desc_.desc(),
                                                  h.begin()));

    }

    void backward_params(Tensor<T1> x, Tensor<T1> delta, Tensor<T1> dW) {

        CHECK_CUDNN_ERROR(cudnnConvolutionBackwardFilter(cudnn_handle_.handle(),
                                                         &alpha_,
                                                         x_desc_.desc(),
                                                         x.begin(),
                                                         h_desc_.desc(),
                                                         delta.begin(),
                                                         conv_desc_.desc(),
                                                         bwd_params_algo_,
                                                         bwd_params_workspace_.begin(),
                                                         bwd_params_workspace_size_,
                                                         &beta_,
                                                         w_desc_.desc(),
                                                         dW.begin()));


    }

    void backward_inputs(Tensor<T1> filter, Tensor<T1> delta, Tensor<T1> dX) {

        CHECK_CUDNN_ERROR(cudnnConvolutionBackwardData(cudnn_handle_.handle(),
                                                      &alpha_,
                                                      w_desc_.desc(),
                                                      filter.begin(),
                                                      h_desc_.desc(),
                                                      delta.begin(),
                                                      conv_desc_.desc(),
                                                      bwd_inputs_algo_,
                                                      bwd_inputs_workspace_.begin(),
                                                      bwd_inputs_workspace_size_,
                                                      &beta_,
                                                      x_desc_.desc(),
                                                      dX.begin()));

    }
};
template <typename T1, typename T2>
std::tuple<int, int, int, std::string> time_cnn(
         int k, int c, int r, int s,
         int n, int h, int w,
         int pad_h, int pad_w,
         int hstride, int wstride,
         int num_repeats,
         curandGenerator_t curand_gen) {

    cudnnCNN<T1, T2> cnn(w, h, c, n, k, r, s, pad_w, pad_h, wstride, hstride);

    // Allocate memory for filter
    auto filter = rand<T1>(std::vector<int>{s, r, c, k}, curand_gen);

    auto random_input = rand<T1>(std::vector<int>{w, h, c, n}, curand_gen);
        // Allocate memory for input
    auto input = zeros<T1>(std::vector<int>{w, h, c, n});
    // Allocate memory for output tensor
    auto output = zeros<T1>(cnn.get_output_dims());

    std::string fwd_algo_s = cnn.get_fwd_algo_string();

    //Warm up
	cnn.forward(input, filter, output);

	cudaDeviceSynchronize();
	auto start = std::chrono::steady_clock::now();

	for (int i = 0; i < num_repeats; ++i) {
		cnn.forward(input, filter, output);
	}

	cudaDeviceSynchronize();
	auto end = std::chrono::steady_clock::now();
	int fwd_time = static_cast<int>(std::chrono::duration<double, std::micro>(end - start).count() / num_repeats);

    int bwd_inputs_time = 0;
    int bwd_params_time = 0;

	// Allocate memory for backward pass wrt weights
	auto delta = rand<T1>(cnn.get_output_dims(), curand_gen);
	auto dW = zeros<T1>(std::vector<int>{s, r, c, k});

	// Warm up backward
	cnn.backward_params(input, delta, dW);

	cudaDeviceSynchronize();
	start = std::chrono::steady_clock::now();

	for (int i = 0; i < num_repeats; ++i) {
		// Backward pass wrt weights
		cnn.backward_params(input, delta, dW);
	}

	cudaDeviceSynchronize();
	end = std::chrono::steady_clock::now();

	bwd_params_time = static_cast<int>(std::chrono::duration<double, std::micro>(end - start).count() / num_repeats);

	//Allocate memory for backward pass wrt inputs
	auto dX = zeros<T1>(std::vector<int>{w, h, c, n});

	//Warm up backward inputs
	cnn.backward_inputs(filter, delta, dX);

	cudaDeviceSynchronize();
	start = std::chrono::steady_clock::now();

	for (int i = 0; i < num_repeats; ++i) {
		// Backward pass wrt weights
		cnn.backward_inputs(filter, delta, dX);

	}

	cudaDeviceSynchronize();
	end = std::chrono::steady_clock::now();

	bwd_inputs_time = static_cast<int>(std::chrono::duration<double, std::micro>(end - start).count() / num_repeats);

    return std::tuple<int, int, int, std::string>(fwd_time, bwd_inputs_time, bwd_params_time, fwd_algo_s);

}

int main(int argc, char **argv) {

    int num_repeats = 300;

    std::string precision = "half";


    // Handles to various cuda libraries, structures
    curandGenerator_t curand_gen;

    cudaFree(0);

    // Initialize curand_gen and set appropriate seed.
    curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(curand_gen, 123ULL);


    std::cout << std::setw(45) << "Running training benchmark " << std::endl;

    std::cout << std::setw(30) << "Times" << std::endl;
    std::cout << std::setfill('-') << std::setw(190) << "-" << std::endl;
    std::cout << std::setfill(' ');
    std::cout << "   w      h      c      n      k      f_w    f_h  pad_w  pad_h    stride_w  stride_h    precision  fwd_time (usec)  ";

    std::cout << "bwd_inputs_time (usec)  bwd_params_time (usec)  ";
    std::cout << "total_time (usec)";

    std::cout << "   fwd_algo " << std::endl;

    std::cout << std::setfill('-') << std::setw(200) << "-" << std::endl;
    std::cout << std::setfill(' ');

    for (const auto &problem : training_set) {

        // Filter parameters
        int k, c, r, s; // r - filter_h (f_h), s - filter_w (f_w)

        // Input parameters
        int n, w, h;

        // Padding
        int pad_w, pad_h;

        // Stride
        int wstride, hstride;

        std::tie(w, h, c, n, k, s, r, pad_w, pad_h, wstride, hstride) = problem;

        int padded_c, padded_w, padded_h;

        padded_c = c;
        padded_h = h;
        padded_w = w;


        int fwd_time, bwd_inputs_time, bwd_params_time;
        std::string fwd_algo_s;

        std::stringstream ss;

        std::tie(fwd_time, bwd_inputs_time, bwd_params_time, fwd_algo_s) =
             time_cnn<float, float>(k, padded_c, r, s, n, padded_h, padded_w, pad_h, pad_w, hstride, wstride, num_repeats, curand_gen);

        std::cout << std::setw(5) << w;
        std::cout << std::setw(7) << h;
        std::cout << std::setw(7) << c;
        std::cout << std::setw(7) << n;
        std::cout << std::setw(7) << k;
        std::cout << std::setw(7) << s;
        std::cout << std::setw(7) << r;
        std::cout << std::setw(7) << pad_w;
        std::cout << std::setw(8) << pad_h;
        std::cout << std::setw(10) << wstride;
        std::cout << std::setw(10) << hstride;
        std::cout << std::setw(10) << precision;
        std::cout << std::setw(15) << std::setprecision(7);



        std::cout << std::setw(24) << std::setprecision(7) << bwd_inputs_time;
		std::cout << std::setw(24) << std::setprecision(7) << bwd_params_time;
		std::cout << std::setw(19) << std::setprecision(8) << fwd_time + bwd_inputs_time + bwd_params_time;


        std::cout << std::setw(25) << fwd_algo_s;
        std::cout << std::endl;
    }
    // Destroy all the handles
    curandDestroyGenerator(curand_gen);

    return 0;
}
