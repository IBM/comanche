/*
 * Orginially From Baidu Research DeepBench.
 * Modifed to test Dawn
 * Author: Yanzhao Wu
 */
#include <boost/program_options.hpp>

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

#include <common/exceptions.h>
#include <api/components.h>
#include <common/dump_utils.h>
#include <common/logging.h>
#include <api/kvstore_itf.h>
#include <config_comanche.h>

#include <api/block_itf.h>
#include "core/physical_memory.h"
#include "core/xms.h"
#include "api/memory_itf.h"

#define USE_GET 0

#ifndef PAD_KERNELS
#define PAD_KERNELS 1
#endif

#ifndef USE_TENSOR_CORES
#if CUDNN_MAJOR >= 7
#define USE_TENSOR_CORES 1
#else
#define USE_TENSOR_CORES 0
#endif
#endif

using pool_t = uint64_t;
using namespace Component;
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
             int pad_w, int pad_h, int wstride, int hstride,
             int inference)
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

#if (CUDNN_MAJOR >= 7) && (USE_TENSOR_CORES)
        cudnnSetConvolutionMathType(conv_desc_.desc(), CUDNN_TENSOR_OP_MATH);
#endif
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

#if USE_GET
        if (std::is_same<T1, uint8_t>::value) {
            //Note: cuDNN only supports IMPLICIT_PRECOMP_GEMM for int8 data type.
            fwd_algo_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
        } else {
        // Pick forward convolution algorithm
        CHECK_CUDNN_ERROR(cudnnGetConvolutionForwardAlgorithm(cudnn_handle_.handle(),
                                                              x_desc_.desc(),
                                                              w_desc_.desc(),
                                                              conv_desc_.desc(),
                                                              h_desc_.desc(),
                                                              CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                              0,
                                                              &fwd_algo_));
        }
#else
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
#endif
#if (CUDNN_MAJOR >= 7) && (USE_TENSOR_CORES)
        // Tensor Op math only supports IMPLICIT_PRECOMP_GEMM algorithm
        fwd_algo_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
#endif
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

        if (!inference) {
#if USE_GET
            // Pick backward convolution algorithm
            CHECK_CUDNN_ERROR(cudnnGetConvolutionBackwardFilterAlgorithm(cudnn_handle_.handle(),
                                                                         x_desc_.desc(),
                                                                         h_desc_.desc(),
                                                                         conv_desc_.desc(),
                                                                         w_desc_.desc(),
                                                                         CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
                                                                         0,
                                                                         &bwd_params_algo_));
#else
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
#endif
#if (CUDNN_MAJOR >= 7) && (USE_TENSOR_CORES)
            // Tensor Op math only supports this algorithm.
            bwd_params_algo_ = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
#endif

            // Backward params workspace
            CHECK_CUDNN_ERROR(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn_handle_.handle(),
                                                                             x_desc_.desc(),
                                                                             h_desc_.desc(),
                                                                             conv_desc_.desc(),
                                                                             w_desc_.desc(),
                                                                             bwd_params_algo_,
                                                                             &bwd_params_workspace_size_));



            bwd_params_workspace_ = zeros<float>(std::vector<int>{static_cast<int>(bwd_params_workspace_size_ / sizeof(float)), 1});

#if USE_GET
            // Pick backward wrt inputs convolution algorithm
            CHECK_CUDNN_ERROR(cudnnGetConvolutionBackwardDataAlgorithm(cudnn_handle_.handle(),
                                                                       w_desc_.desc(),
                                                                       h_desc_.desc(),
                                                                       conv_desc_.desc(),
                                                                       x_desc_.desc(),
                                                                       CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
                                                                       0,
                                                                       &bwd_inputs_algo_));
#else
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
#endif
#if (CUDNN_MAJOR >= 7) && (USE_TENSOR_CORES)
            //Tensor Op math only supports this algorithm.
            bwd_inputs_algo_ = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
#endif

            CHECK_CUDNN_ERROR(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle_.handle(),
                                                                           w_desc_.desc(),
                                                                           h_desc_.desc(),
                                                                           conv_desc_.desc(),
                                                                           x_desc_.desc(),
                                                                           bwd_inputs_algo_,
                                                                           &bwd_inputs_workspace_size_));

            bwd_inputs_workspace_ = zeros<float>(std::vector<int>{static_cast<int>(bwd_inputs_workspace_size_ / sizeof(float)), 1});
        }

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
#if CUDNN_MAJOR >= 6
        else if (fwd_algo_ == CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED)
            return "WINOGRAD_NONFUSED";
#endif
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
         curandGenerator_t curand_gen,
         int inference,
	 Component::IKVStore * store,
	 pool_t pool
        ) {

    cudnnCNN<T1, T2> cnn(w, h, c, n, k, r, s, pad_w, pad_h, wstride, hstride, inference);

    // Allocate memory for filter
    auto filter = rand<T1>(std::vector<int>{s, r, c, k}, curand_gen);

    auto random_input = rand<T1>(std::vector<int>{w, h, c, n}, curand_gen);
    std::string key_name("Task");
    key_name = key_name+"_k"+std::to_string(k)\
			+"_c"+std::to_string(c)\
			+"_r"+std::to_string(r)\
			+"_s"+std::to_string(s)\
			+"_n"+std::to_string(n)\
			+"_h"+std::to_string(h)\
			+"_w"+std::to_string(w)\
			+"_padH"+std::to_string(pad_h)\
			+"_padW"+std::to_string(pad_w)\
			+"_hstride"+std::to_string(hstride)\
			+"_wstride"+std::to_string(wstride);


    // Allocate memory for input
    auto input = zeros<T1>(std::vector<int>{w, h, c, n});
    // Allocate memory for output tensor
    auto output = zeros<T1>(cnn.get_output_dims());

    //std::string fwd_algo_s = cnn.get_fwd_algo_string();
    //write_to_file(key_name, random_input);
    put_into_kvstore(key_name, random_input, store, pool);

    auto handle = store->register_direct_memory((void*) input.begin(), input.size());
    assert(handle);

    std::string fwd_algo_s = cnn.get_fwd_algo_string();

    //Warm up
    //read_from_file(key_name, input);
    size_t input_size = input.size();
    status_t rc = store->get_direct(pool, key_name, input.begin(), input_size, handle);
    if(rc != S_OK) {
    	PLOG("Error: Get_Direct %d", rc);
    	PLOG("Error Code: S_MORE %d", IKVStore::S_MORE);
    	PLOG("Error Code: E_POOL_NOT_FOUND %d", IKVStore::E_POOL_NOT_FOUND);
    	PLOG("Error Code: E_KEY_NOT_FOUND %d", IKVStore::E_KEY_NOT_FOUND);
    }

    std::chrono::system_clock::time_point read_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_repeats; ++i) {
        //read_from_file(key_name, input);
        rc=store->get_direct(pool, key_name, input.begin(), input_size, handle);
        assert(rc == S_OK || rc == IKVStore::S_MORE);
    }
    //read_from_file(key_name, input);
    //store->get_direct(file_name, gdr_ptr._h_ptr, input.size());
    std::chrono::system_clock::time_point read_end = std::chrono::high_resolution_clock::now();
    auto micro_sec = std::chrono::duration_cast<std::chrono::microseconds>(read_end - read_start).count();
    PLOG("Read from remote kvstore takes %lf microsecond", micro_sec/(double)num_repeats);
    //void * tmp_space = (void *) malloc(input.size());
    //memset(tmp_space, 0xb, input.size());
    //size_t tmp_space_len = 0;
    //assert(tmp_space);
    //PLOG("GPU_direct read to CPU memory");
    //kvstore->get_direct(file_name, tmp_space, input.size());
    //kvstore->get_direct(file_name, gdr_ptr._h_ptr, input.size());
    //assert(tmp_space_len == input.size());
    //hexdump(tmp_space, 32);
    //kvstore->get_direct(file_name, gdr_ptr._h_ptr, input.size());
    //PLOG("Dump input tensor");
    //hexdump_tensor(input);
    
    //cnn.forward(input, filter, output);
    store->unregister_direct_memory(handle);
    return std::tuple<int, int, int, std::string>(0, 0, 0, fwd_algo_s);
    cudaDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();
    /*
    for (int i = 0; i < num_repeats; ++i) {
    	//read_from_file(file_name, input);
    	kvstore->get_direct(file_name, gdr_ptr._h_ptr, input.size());
        cnn.forward(input, filter, output);
    }
    */

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    int fwd_time = static_cast<int>(std::chrono::duration<double, std::micro>(end - start).count() / num_repeats);

    int bwd_inputs_time = 0;
    int bwd_params_time = 0;

    if (!inference) {
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
    }

    return std::tuple<int, int, int, std::string>(fwd_time, bwd_inputs_time, bwd_params_time, fwd_algo_s);

}

Component::IKVStore * create_store(const std::string& addr,
                                   const std::string& device,
                                   const unsigned debug_level) {
  using namespace Component;

  std::string path = CONF_COMANCHE_INSTALL;
  path += "/lib/libcomanche-dawn-client.so";

  IBase * comp = load_component(path.c_str(), dawn_client_factory);
  assert(comp);
  IKVStore_factory * fact = (IKVStore_factory *) comp->query_interface(IKVStore_factory::iid());

  IKVStore * inst = nullptr;

  inst = fact->create(debug_level,
                      "convbench",
                      addr.c_str(),
                      device.c_str());

  fact->release_ref();
  return inst;
}

int main(int argc, char **argv) {

    int num_repeats = 300;

    int inference = 0;

    namespace po = boost::program_options;
    po::options_description desc("Options");
    desc.add_options()
    		("dawn-server", po::value<std::string>()->default_value("10.0.0.22:11911"))
			("debug", po::value<unsigned>()->default_value(0))
			("device", po::value<std::string>()->default_value("mlx5_0"))
			("inference", po::value<int>()->default_value(0))
			("precision", po::value<std::string>()->default_value("float")) 
			("help", "Show this help")
			;
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    if(vm.count("help")) {
    	std::cout << desc;
    	return -1;
    }
    inference = vm["inference"].as<int>();

#if CUDNN_MAJOR >= 6
    std::string precision;
    if (inference)
        precision = "int8";
    else
        precision = "half";
#else
    std::string precision = "float";
#endif
    std::cout << vm.count("precision") << vm["precision"].as<std::string>() << std::endl;
    if (vm.count("precision")) {
        precision = vm["precision"].as<std::string>();
    }

    // Create KVStore
    Component::IKVStore *  store = create_store(vm["dawn-server"].as<std::string>(),
    							vm["device"].as<std::string>(),
								vm["debug"].as<unsigned>());
    // Create pool
    auto pool = store->create_pool("/poolsgpu0", MB(128));


    // Handles to various cuda libraries, structures
    curandGenerator_t curand_gen;


    cudaFree(0);

    // Initialize curand_gen and set appropriate seed.
    curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(curand_gen, 123ULL);


    if (inference) {
        std::cout << std::setw(45) << "Running inference benchmark " << std::endl;
    } else {
        std::cout << std::setw(45) << "Running training benchmark " << std::endl;
    }

    std::cout << std::setw(30) << "Times" << std::endl;
    std::cout << std::setfill('-') << std::setw(190) << "-" << std::endl;
    std::cout << std::setfill(' ');
    std::cout << "   w      h      c      n      k      f_w    f_h  pad_w  pad_h    stride_w  stride_h    precision  fwd_time (usec)  ";

    if (!inference) {
        std::cout << "bwd_inputs_time (usec)  bwd_params_time (usec)  ";
        std::cout << "total_time (usec)";
    }

    if (PAD_KERNELS && ((precision == "int8" && inference) || (USE_TENSOR_CORES && !inference)))
        std::cout << " pad_kerenels  ";

    std::cout << "   fwd_algo " << std::endl;

    std::cout << std::setfill('-') << std::setw(200) << "-" << std::endl;
    std::cout << std::setfill(' ');

    int pad_kernels_count = 0;

    for (const auto &problem : (inference ? inference_server_set : training_set)) {

        // Filter parameters
        int k, c, r, s; // r - filter_h (f_h), s - filter_w (f_w)

        // Input parameters
        int n, w, h;

        // Padding
        int pad_w, pad_h;

        // Stride
        int wstride, hstride;

        std::tie(w, h, c, n, k, s, r, pad_w, pad_h, wstride, hstride) = problem;

        bool skip_kernel = false;
        bool need_padding = false;

#if CUDNN_MAJOR >= 6
        int padded_c, padded_w, padded_h;
        int pad_value;

        padded_c = c;
        padded_h = h;
        padded_w = w;

        if (precision == "int8") {
            pad_value = 4;
            if (c % pad_value || w % pad_value || h % pad_value) {
                pad_kernels_count++;
                if (PAD_KERNELS) {
                    pad_dim(padded_c, pad_value);
                    pad_dim(padded_h, pad_value);
                    pad_dim(padded_w, pad_value);
                    need_padding = true;
                } else {
                    skip_kernel = true;
                }
            }
        }
#if (USE_TENSOR_CORES)
        // Tensor cores need channels to be a multiple of 8. So, added padding for some kernels.
        if (!inference) {
            pad_value = 8;
            if (c % pad_value) {
                pad_kernels_count++;
                if (PAD_KERNELS) {
                    pad_dim(padded_c, pad_value);
                    need_padding = true;
                } else {
                    skip_kernel = true;
                }
            }
        }
#endif
#endif

        int fwd_time, bwd_inputs_time, bwd_params_time;
        std::string fwd_algo_s;

        std::stringstream ss;
        ss << "Unsupported precision requested. Precision: " << precision << " Inference: " << inference;
        //if(w*h*c*n >= MB(8) || padded_w*padded_h*padded_c*n >= MB(8)) {
        //   PLOG(" > 8MB, skip");
        //   continue;
        //} 
#if CUDNN_MAJOR >= 6
        if (precision == "float") {
            std::tie(fwd_time, bwd_inputs_time, bwd_params_time, fwd_algo_s) =
                time_cnn<float, float>(k, padded_c, r, s, n, padded_h, padded_w, pad_h, pad_w, hstride, wstride, num_repeats, curand_gen, inference, store, pool);
        } else if (precision == "half") {
            std::tie(fwd_time, bwd_inputs_time, bwd_params_time, fwd_algo_s) =
                time_cnn<uint16_t, uint16_t>(k, padded_c, r, s, n, padded_h, padded_w, pad_h, pad_w, hstride, wstride, num_repeats, curand_gen, inference, store, pool);
        } else if ((precision == "int8") && inference) {
            if (!skip_kernel) {
                std::tie(fwd_time, bwd_inputs_time, bwd_params_time, fwd_algo_s) =
                    time_cnn<uint8_t, int>(k, padded_c, r, s, n, padded_h, padded_w, pad_h, pad_w, hstride, wstride, num_repeats, curand_gen, inference, store, pool);
            }
        } else {
            throw std::runtime_error(ss.str());
        }
#else
        if (precision != "float")
            throw std::runtime_error(ss.str());
        std::tie(fwd_time, bwd_inputs_time, bwd_params_time, fwd_algo_s) =
            time_cnn<float, float>(k, c, r, s, n, h, w, pad_h, pad_w, hstride, wstride, num_repeats, curand_gen, inference, store, pool);
#endif

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

        if (skip_kernel) {
            std::cout << "Not Supported";
        } else {
            std::cout << fwd_time;
        }

        if (PAD_KERNELS && precision == "int8" && inference) {
            std::cout << std::setw(15) <<  need_padding;
        }



        if (!inference) {
            std::cout << std::setw(24) << std::setprecision(7) << bwd_inputs_time;
            std::cout << std::setw(24) << std::setprecision(7) << bwd_params_time;
            std::cout << std::setw(19) << std::setprecision(8) << fwd_time + bwd_inputs_time + bwd_params_time;
        }

        if (USE_TENSOR_CORES && PAD_KERNELS && !inference) {
            std::cout << std::setw(15) <<  need_padding;
        }


        std::cout << std::setw(25) << fwd_algo_s;
        std::cout << std::endl;
    }

    if (precision == "int8") {
        std::cout << " Total kernels ";
        if (PAD_KERNELS)
            std::cout << "padded: " << pad_kernels_count << std::endl;
        else
            std::cout << "skipped: " << pad_kernels_count << std::endl;

        std::cout << " Total kernels: " << inference_server_set.size() << std::endl;
    }
    // Close the pool
    store->close_pool(pool);
    store->delete_pool("/poolsgpu0");
    // Release the KVStore
    store->release_ref();

    // Destroy all the handles
    curandDestroyGenerator(curand_gen);

    return 0;
}
