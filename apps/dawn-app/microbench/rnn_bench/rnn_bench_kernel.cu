/*
 * Orginially From Baidu Research DeepBench.
 * Modifed to test Dawn
 * Author: Yanzhao Wu
 */
#include <boost/program_options.hpp>

#include <chrono>
#include <iomanip>
#include <memory>
#include <stdexcept>
#include <tuple>

#include <cuda.h>
#include <cudnn.h>
#include <curand.h>

#include <thrust/device_ptr.h>
#include <thrust/fill.h>

#include <cudautility.hpp>

#include "tensor.h"
#include "cudnn_helper.h"
#include "rnn_problems.h"

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

using pool_t = uint64_t;
using namespace Component;

/*
Usage:

The default precision is set based on the architecture and mode.

By default, the program runs the benchmark in training mode.

bin/rnn_bench

To run inference mode, use the following command:

bin/rnn_bench inference


To change the precision for training/inference, use:

bin/rnn_bench train <precision>
bin/rnn_bench inference <precision>

Supported precision types:

For Maxwell GPUS:
float for training and inference

For Pascal GPUS:
float, half for training
float, half, int8 for inference

*/

#ifndef USE_TENSOR_CORES
#if CUDNN_MAJOR >= 7
#define USE_TENSOR_CORES 1
#else
#define USE_TENSOR_CORES 0
#endif
#endif


cudnnHandle_t cudnn_handle;
curandGenerator_t curand_gen;


class cudnnDropout {
    std::shared_ptr<cudnnDropoutDescriptor_t> dropout_desc_;
    std::shared_ptr<Tensor<uint8_t>> dropout_state_;

    struct DropoutDeleter {
        void operator()(cudnnDropoutDescriptor_t * dropout_desc) {
            cudnnDestroyDropoutDescriptor(*dropout_desc);
            delete dropout_desc;
        }
    };

    public:

    cudnnDropout(float dropout_percentage) : dropout_desc_(new cudnnDropoutDescriptor_t,
                                                           DropoutDeleter()) {
        size_t dropoutStateSize;
        CHECK_CUDNN_ERROR(cudnnCreateDropoutDescriptor(dropout_desc_.get()));
        CHECK_CUDNN_ERROR(cudnnDropoutGetStatesSize(cudnn_handle, &dropoutStateSize));

        dropout_state_.reset(new Tensor<uint8_t>(std::vector<int>{static_cast<int>(dropoutStateSize), 1}));

        CHECK_CUDNN_ERROR(cudnnSetDropoutDescriptor(*dropout_desc_,
                                                    cudnn_handle,
                                                    dropout_percentage,
                                                    dropout_state_->begin(),
                                                    dropoutStateSize,
                                                    0ULL) );
    }

    cudnnDropoutDescriptor_t desc() const { return *dropout_desc_; }
};

template <typename T>
class cudnnRNN {
    RNNDescriptor<T> rnn_desc_;
    FilterDescriptorNd<T> wDesc_;
    cudnnDropout dropout_;

    int time_steps_;

    TensorDescriptorNdArray<T> xDescArray_;
    TensorDescriptorNdArray<T> yDescArray_;
    TensorDescriptorNdArray<T> dxDescArray_;
    TensorDescriptorNdArray<T> dyDescArray_;

    TensorDescriptorNd<T> hx_desc_;
    TensorDescriptorNd<T> hy_desc_;
    TensorDescriptorNd<T> dhx_desc_;
    TensorDescriptorNd<T> dhy_desc_;
    TensorDescriptorNd<T> cx_desc_;
    TensorDescriptorNd<T> cy_desc_;
    TensorDescriptorNd<T> dcx_desc_;
    TensorDescriptorNd<T> dcy_desc_;

    size_t weight_size_;
    size_t workspace_size_;
    size_t train_size_;

    Tensor<T> weights_;
    Tensor<T> dW_;

    Tensor<float> workspace_;
    Tensor<float> trainspace_;

    public:

    cudnnRNN(int hidden_size, int batch_size, int time_steps, const std::string& rnn_type) :
        dropout_(0.f), time_steps_(time_steps),
        xDescArray_({batch_size, hidden_size, 1}, {hidden_size, 1, 1}, time_steps),
        yDescArray_({batch_size, hidden_size, 1}, {hidden_size, 1, 1}, time_steps),
        dxDescArray_({batch_size, hidden_size, 1}, {hidden_size, 1, 1}, time_steps),
        dyDescArray_({batch_size, hidden_size, 1}, {hidden_size, 1, 1}, time_steps),
        hx_desc_({1, batch_size, hidden_size}, {hidden_size * batch_size, hidden_size, 1}),
        hy_desc_({1, batch_size, hidden_size}, {hidden_size * batch_size, hidden_size, 1}),
        dhx_desc_({1, batch_size, hidden_size}, {hidden_size * batch_size, hidden_size, 1}),
        dhy_desc_({1, batch_size, hidden_size}, {hidden_size * batch_size, hidden_size, 1}),
        cx_desc_({1, batch_size, hidden_size}, {hidden_size * batch_size, hidden_size, 1}),
        cy_desc_({1, batch_size, hidden_size}, {hidden_size * batch_size, hidden_size, 1}),
        dcx_desc_({1, batch_size, hidden_size}, {hidden_size * batch_size, hidden_size, 1}),
        dcy_desc_({1, batch_size, hidden_size}, {hidden_size * batch_size, hidden_size, 1})
        {


            rnn_desc_ = RNNDescriptor<T>(hidden_size,
                                             1,
                                             dropout_.desc(),
                                             CUDNN_SKIP_INPUT,
                                             CUDNN_UNIDIRECTIONAL,
                                             rnn_type,
                                             cudnn_handle);
            cudnnDataType_t type;
            if (std::is_same<T, float>::value)
                type = CUDNN_DATA_FLOAT;
#if CUDNN_MAJOR >= 6
            else if (std::is_same<T, uint8_t>::value)
                type = CUDNN_DATA_INT8;
#endif
            else if (std::is_same<T, uint16_t>::value)
                type= CUDNN_DATA_HALF;
            else 
                throw std::runtime_error("Unknown type in cudnnRNN constructor.");

            CHECK_CUDNN_ERROR( cudnnGetRNNParamsSize(cudnn_handle,
                                                     rnn_desc_.desc(),
                                                     xDescArray_.ptr()[0],
                                                     &weight_size_,
                                                     type) );

#if (CUDNN_MAJOR >= 7) && (USE_TENSOR_CORES)
            CHECK_CUDNN_ERROR( cudnnSetRNNMatrixMathType(rnn_desc_.desc(), CUDNN_TENSOR_OP_MATH) );
#endif

            weights_ = rand<T>(std::vector<int>{static_cast<int>(weight_size_ / sizeof(T)), 1}, curand_gen);

            std::vector<int> dim = {weights_.size(), 1, 1};
            wDesc_ = FilterDescriptorNd<T>(CUDNN_TENSOR_NCHW, dim);

            CHECK_CUDNN_ERROR( cudnnGetRNNWorkspaceSize(cudnn_handle,
                                                        rnn_desc_.desc(),
                                                        time_steps,
                                                        xDescArray_.ptr(),
                                                        &workspace_size_) );

            dW_ = zeros<T>(std::vector<int>{static_cast<int>(weight_size_ / sizeof(T)), 1});

            workspace_ = zeros<float>(std::vector<int>{static_cast<int>(workspace_size_ / sizeof(float)), 1});

            CHECK_CUDNN_ERROR( cudnnGetRNNTrainingReserveSize(cudnn_handle,
                                                              rnn_desc_.desc(),
                                                              time_steps,
                                                              xDescArray_.ptr(),
                                                              &train_size_) );
            trainspace_ = zeros<float>(std::vector<int>{static_cast<int>(train_size_ / sizeof(float)), 1});
        }
        void forward(Tensor<T> x, Tensor<T> hx, Tensor<T> cx,
                     Tensor<T> y, Tensor<T> hy, Tensor<T> cy) {
            CHECK_CUDNN_ERROR( cudnnRNNForwardTraining(cudnn_handle,
                                                       rnn_desc_.desc(),
                                                       time_steps_,
                                                       xDescArray_.ptr(),
                                                       (void *)x.begin(),
                                                       hx_desc_.desc(),
                                                       (void *)hx.begin(),
                                                       cx_desc_.desc(),
                                                       (void *)cx.begin(),
                                                       wDesc_.desc(),
                                                       (void *)weights_.begin(),
                                                       yDescArray_.ptr(),
                                                       (void *)y.begin(),
                                                       hy_desc_.desc(),
                                                       (void *)hy.begin(),
                                                       cy_desc_.desc(),
                                                       (void *)cy.begin(),
                                                       (void *)workspace_.begin(),
                                                       workspace_size_,
                                                       (void *)trainspace_.begin(),
                                                       train_size_) );
        }
        void backward_data(Tensor<T> y, Tensor<T> dy, Tensor<T> dhy,
                           Tensor<T> dcy, Tensor<T> hx, Tensor<T> cx,
                           Tensor<T> dx, Tensor<T> dhx, Tensor<T> dcx) {
            CHECK_CUDNN_ERROR( cudnnRNNBackwardData(cudnn_handle,
                                                    rnn_desc_.desc(),
                                                    time_steps_,
                                                    yDescArray_.ptr(),
                                                    (void *)y.begin(),
                                                    dyDescArray_.ptr(),
                                                    (void *)dy.begin(),
                                                    dhy_desc_.desc(),
                                                    (void *)dhy.begin(),
                                                    dcy_desc_.desc(),
                                                    (void *)dcy.begin(),
                                                    wDesc_.desc(),
                                                    (void *)weights_.begin(),
                                                    hx_desc_.desc(),
                                                    (void *)hx.begin(),
                                                    cx_desc_.desc(),
                                                    (void *)cx.begin(),
                                                    dxDescArray_.ptr(),
                                                    (void *)dx.begin(),
                                                    dhx_desc_.desc(),
                                                    (void *)dhx.begin(),
                                                    dcx_desc_.desc(),
                                                    (void *)dcx.begin(),
                                                    (void *)workspace_.begin(),
                                                    workspace_size_,
                                                    (void *)trainspace_.begin(),
                                                    train_size_) );
        }

        void backward_params(Tensor<T> x, Tensor<T> hx, Tensor<T> y) {
            CHECK_CUDNN_ERROR(cudnnRNNBackwardWeights(cudnn_handle,
                                                      rnn_desc_.desc(),
                                                      time_steps_,
                                                      xDescArray_.ptr(),
                                                      (void *)x.begin(),
                                                      hx_desc_.desc(),
                                                      (void *)hx.begin(),
                                                      yDescArray_.ptr(),
                                                      (void *)y.begin(),
                                                      (void *)workspace_.begin(),
                                                      workspace_size_,
                                                      wDesc_.desc(),
                                                      (void *)dW_.begin(),
                                                      (void *)trainspace_.begin(),
                                                      train_size_) );
        }

};

template <typename T>
std::tuple<int, int, int> time_rnn(int hidden_size,
                                   int batch_size,
                                   int time_steps,
                                   const std::string& type,
                                   int inference,
								   Component::IKVStore * store,
								   pool_t pool) {

    cudnnRNN<T> rnn(hidden_size, batch_size, time_steps, type);

    auto x  = rand<T>({hidden_size, batch_size * time_steps}, curand_gen);
    std::string key_name("Task");
    key_name = key_name+"_hs" + std::to_string(hidden_size)\
    		+ "_bs" + std::to_string(batch_size)\
			+ "_ts" + std::to_string(time_steps);
    auto input = zeros<T>({hidden_size, batch_size * time_steps});
    auto handle = store -> register_direct_memory((void*) input.begin(), input.size());
    assert(handle);

    //write_to_file(key_name, x);
    put_into_kvstore(key_name, x, store, pool);

    auto y  = rand<T>({hidden_size, batch_size * time_steps}, curand_gen);
    auto dx = rand<T>({hidden_size, batch_size * time_steps}, curand_gen);
    auto dy = rand<T>({hidden_size, batch_size * time_steps}, curand_gen);

    auto hx = rand<T>({hidden_size, batch_size}, curand_gen);
    auto hy = rand<T>({hidden_size, batch_size}, curand_gen);
    auto cx = rand<T>({hidden_size, batch_size}, curand_gen);
    auto cy = rand<T>({hidden_size, batch_size}, curand_gen);
    auto dhx = rand<T>({hidden_size, batch_size}, curand_gen);
    auto dhy = rand<T>({hidden_size, batch_size}, curand_gen);
    auto dcx = rand<T>({hidden_size, batch_size}, curand_gen);
    auto dcy = rand<T>({hidden_size, batch_size}, curand_gen);

    int numRepeats = 100;

    //Warm up
    size_t input_size = input.size();
    //read_from_file(key_name, input);
    status_t rc = store->get_direct(pool, key_name, input.begin(), input_size, handle);
    rnn.forward(input, hx, cx, y, hy, cy);

    cudaDeviceSynchronize();

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < numRepeats; ++i) {
        //read_from_file(key_name, input);
        rc = store->get_direct(pool, key_name, input.begin(), input_size, handle);
    	assert(rc == S_OK);
        //rnn.forward(input, hx, cx, y, hy, cy);
    }
    cudaDeviceSynchronize();

    auto end = std::chrono::steady_clock::now();
    auto micro_sec = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    PLOG("Read from remote kvstore takes %lf us", micro_sec/(double)numRepeats);


    auto forward_time = std::chrono::duration<double, std::micro>(end - start).count() / numRepeats;
    int bwd_data_time = 0;
    int bwd_params_time = 0;

    if (!inference) {
        //Warm up
        rnn.backward_data(y, dy, dhy, dcy,
                          hx, cx, dx, dhx, dcx);

        cudaDeviceSynchronize();

        start = std::chrono::steady_clock::now();

        for (int i = 0; i < numRepeats; ++i) {
            rnn.backward_data(y, dy, dhy, dcy,
                              hx, cx, dx, dhx, dcx);
        }
        cudaDeviceSynchronize();

        end = std::chrono::steady_clock::now();
        bwd_data_time = std::chrono::duration<double, std::micro>(end - start).count() / numRepeats;

        /* Backward wrt params */
        //Warm up
        rnn.backward_params(x, hx, y);

        cudaDeviceSynchronize();

        start = std::chrono::steady_clock::now();

        for (int i = 0; i < numRepeats; ++i) {
            rnn.backward_params(x, hx, y);
        }

        cudaDeviceSynchronize();

        end = std::chrono::steady_clock::now();
        bwd_params_time = std::chrono::duration<double, std::micro>(end - start).count() / numRepeats;


    }

    return std::make_tuple(static_cast<int>(forward_time),
                           static_cast<int>(bwd_data_time),
                           static_cast<int>(bwd_params_time));

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
                      "rnnbench",
                      addr.c_str(),
                      device.c_str());

  fact->release_ref();
  return inst;
}


int main(int argc, char **argv) {

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

	cudaFree(0);
	CHECK_CUDNN_ERROR( cudnnCreate(&cudnn_handle) );

	curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(curand_gen, 123ULL);

    if (inference) {
        std::cout << std::setw(45) << "Running inference benchmark " << std::endl;
    } else {
        std::cout << std::setw(45) << "Running training benchmark " << std::endl;
    }

    std::cout << std::setw(30) << "Times" << std::endl;
    std::cout << std::setfill('-') << std::setw(115) << "-" << std::endl;
    std::cout << std::setfill(' ');
    std::cout << "    type    hidden   N     timesteps   precision   fwd_time (usec)   ";
    if (!inference) {
        std::cout << "bwd_inputs_time (usec)";
        std::cout << "  bwd_params_time (usec)";
    }

    std::cout << std::endl;
    for (const auto &problem : (inference ? inference_server_set : training_set)) {
        int hidden_state, batch_size, time_steps;
        std::string type;
        std::tie(hidden_state, batch_size, time_steps, type) = problem;

        std::cout << std::setw(8) << type;
        std::cout << std::setw(8) << hidden_state;
        std::cout << std::setw(8) << batch_size;
        std::cout << std::setw(8) << time_steps;
        std::cout << std::setw(14) << precision;
        int fwd_time, bwd_data_time, bwd_params_time;

        std::stringstream ss;
        ss << "Unsupported precision requested. Precision: " << precision << " Inference: " << inference;

#if CUDNN_MAJOR >= 6
        if (inference) {
            if (precision == "float") {
                std::tie(fwd_time, bwd_data_time, bwd_params_time) =
                    time_rnn<float>(hidden_state,
                                    batch_size,
                                    time_steps,
                                    type,
                                    inference, store, pool);

            } else if (precision == "half") {
                std::tie(fwd_time, bwd_data_time, bwd_params_time) =
                    time_rnn<uint16_t>(hidden_state,
                                       batch_size,
                                       time_steps,
                                       type,
                                       inference, store, pool);
            } else if (precision == "int8") {
                std::tie(fwd_time, bwd_data_time, bwd_params_time) =
                    time_rnn<uint8_t>(hidden_state,
                                      batch_size,
                                      time_steps,
                                      type,
                                      inference, store, pool);
            } else {
                throw std::runtime_error(ss.str());
            }
        } else {
            if (precision == "float") {
                std::tie(fwd_time, bwd_data_time, bwd_params_time) =
                     time_rnn<float>(hidden_state,
                                     batch_size,
                                     time_steps,
                                     type,
                                     inference, store, pool);

            } else if (precision == "half") {
                std::tie(fwd_time, bwd_data_time, bwd_params_time) =
                     time_rnn<uint16_t>(hidden_state,
                                        batch_size,
                                        time_steps,
                                        type,
                                        inference, store, pool);
            } else {
                throw std::runtime_error(ss.str());
            }
        }
#else
        if (precision != "float")
            throw std::runtime_error(ss.str());

        std::tie(fwd_time, bwd_data_time, bwd_params_time) =
             time_rnn<float>(hidden_state,
                             batch_size,
                             time_steps,
                             type,
                             inference, store, pool);
#endif

        std::cout << std::setw(18) << fwd_time;
        if (!inference) {
            std::cout << std::setw(20) << bwd_data_time;
            std::cout << std::setw(20) << bwd_params_time;
        }
        std::cout << std::endl;
    }

    // Close the pool
	store->close_pool(pool);
	store->delete_pool("/poolsgpu0");
	// Release the KVStore
	store->release_ref();


    cudnnDestroy(cudnn_handle);
    curandDestroyGenerator(curand_gen);

    return 0;
}
