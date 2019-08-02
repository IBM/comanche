/*
 * Orginially From Baidu Research DeepBench.
 * Modifed to test Dawn
 * Author: Yanzhao Wu
 */
#include <boost/program_options.hpp>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <vector>
#include <cstdint>
#include <sstream>

#include <cuda.h>
#include <cublas_v2.h>
#include <curand.h>

#include <thrust/device_ptr.h>
#include <thrust/fill.h>

#include <cudautility.hpp>

#include "tensor.h"
#include "gemm_problems.h"

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

#ifndef PAD_KERNELS
#define PAD_KERNELS 1
#endif

#ifndef USE_TENSOR_CORES
#if __CUDACC_VER_MAJOR__ > 8
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

bin/gemm_bench

To run inference mode, use the following command:

bin/gemm_bench inference


To change the precision for training/inference, use:

bin/gemm_bench train <precision>
bin/gemm_bench inference <precision>

Supported precision types:

For Maxwell GPUS: 
float for training and inference

For Pascal GPUS:
float, half for training
float, half, int8 for inference

*/

template <typename T1, typename T2>
int time_gemm(Tensor<T1> A, Tensor<T1> B, Tensor<T2> C, bool a_t, bool b_t, cublasHandle_t cublas_handle, Component::IKVStore * store, pool_t pool) {

#if (__CUDACC_VER_MAJOR__ >= 8)
    const int alpha = 1.f;
    const int beta  = 1.f;
#else
    const float alpha = 1.f / static_cast<float>(A.dims()[1]);
    const float beta  = 1.f;
#endif

    int m = C.dims()[0];
    int k = a_t ? A.dims()[0] : A.dims()[1];
    int n = C.dims()[1];

    std::string key_name("Task");
    key_name = key_name+"_A"+std::to_string(A.dims()[0])\
    			+"x"+std::to_string(A.dims()[1]);

    write_to_file(key_name, A);
    //put_into_kvstore(key_name, A, store, pool);

    auto handle = store->register_direct_memory((void*) A.begin(), A.size());
    assert(handle);

    int numRepeats = 100;
    cublasStatus_t stat;

#if (__CUDACC_VER_MAJOR__ >= 8)
    cudaDataType_t A_type = CUDA_R_32F;
    cudaDataType_t B_type = CUDA_R_32F;
    cudaDataType_t C_type = CUDA_R_32F;
    cudaDataType_t compute_type = CUDA_R_32F;
    cublasGemmAlgo_t algo;

    if (std::is_same<T1, uint16_t>::value) {
        A_type = CUDA_R_16F;
        B_type = CUDA_R_16F;
        C_type = CUDA_R_16F;
        compute_type = CUDA_R_16F;
    }

    if (std::is_same<T1, uint8_t>::value) {
        A_type = CUDA_R_8I;
        B_type = CUDA_R_8I;
        C_type = CUDA_R_32I;
        compute_type = CUDA_R_32I;
    }

#if (USE_TENSOR_CORES)
        algo = CUBLAS_GEMM_DFALT_TENSOR_OP;
#else
        algo = CUBLAS_GEMM_DFALT;
#endif

#endif

    //Storage Warm-up
    read_from_file(key_name, A);
    size_t A_size = 0;
    //store->get_direct(pool, key_name, A.begin(), A_size, handle);

#if (__CUDACC_VER_MAJOR__ < 8)
    // Warm up
    stat = cublasSgemm(cublas_handle,
                a_t ? CUBLAS_OP_T : CUBLAS_OP_N,
                b_t ? CUBLAS_OP_T : CUBLAS_OP_N,
                m,
                n,
                k,
                &alpha,
                A.begin(), A.dims()[0],
                B.begin(), B.dims()[0],
                &beta,
                C.begin(), C.dims()[0]);
#else
    stat = cublasGemmEx(cublas_handle,
                a_t ? CUBLAS_OP_T : CUBLAS_OP_N,
                b_t ? CUBLAS_OP_T : CUBLAS_OP_N,
                m,
                n,
                k,
                &alpha,
                A.begin(), A_type, A.dims()[0],
                B.begin(), B_type, B.dims()[0],
                &beta,
                C.begin(), C_type, C.dims()[0],
                compute_type,
                algo);
#endif

    if (stat != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("sgemm failed");
    }

    cudaDeviceSynchronize();

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < numRepeats; ++i) {
    	read_from_file(key_name, A);
    	//store->get_direct(pool, key_name, A.begin(), A_size);
/*
#if (__CUDACC_VER_MAJOR__ < 8)
        stat = cublasSgemm(cublas_handle,
                    a_t ? CUBLAS_OP_T : CUBLAS_OP_N,
                    b_t ? CUBLAS_OP_T : CUBLAS_OP_N,
                    m,
                    n,
                    k,
                    &alpha,
                    A.begin(), A.dims()[0],
                    B.begin(), B.dims()[0],
                    &beta,
                    C.begin(), C.dims()[0]);
#else
        stat = cublasGemmEx(cublas_handle,
                    a_t ? CUBLAS_OP_T : CUBLAS_OP_N,
                    b_t ? CUBLAS_OP_T : CUBLAS_OP_N,
                    m,
                    n,
                    k,
                    &alpha,
                    A.begin(), A_type, A.dims()[0],
                    B.begin(), B_type, B.dims()[0],
                    &beta,
                    C.begin(), C_type, C.dims()[0],
                    compute_type,
                    algo);
#endif
        if (stat != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("sgemm failed");
        }
*/
    }
    cudaDeviceSynchronize();

    auto end = std::chrono::steady_clock::now();
    auto micro_sec = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    PLOG("Read from remote kvstore takes %lf ms", micro_sec/(double)numRepeats);
    store->unregister_direct_memory(handle);
    return static_cast<int>(std::chrono::duration<double, std::micro>(end - start).count() / numRepeats);

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
    cudaFree(0);

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

#if (__CUDACC_VER_MAJOR__ >= 8)
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

    cublasHandle_t cublas_handle;
    cublasStatus_t status = cublasCreate(&cublas_handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "CUBLAS init failed" << std::endl;
    }

#if (USE_TENSOR_CORES) && (__CUDACC_VER_MAJOR__ > 8)
    status = cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH);
#endif

    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "CUBLAS math mode failed" << std::endl;
    }



    curandGenerator_t curand_gen;

    curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(curand_gen, 123ULL);

    if (inference) {
        std::cout << std::setw(45) << "Running inference benchmark " << std::endl;
    } else {
        std::cout << std::setw(45) << "Running training benchmark " << std::endl;
    }

    std::cout << std::setw(30) << "Times" << std::endl;
    std::cout << std::setfill('-') << std::setw(88) << "-" << std::endl;
    std::cout << std::setfill(' ');
    std::cout << "    m       n      k      a_t     b_t      precision        time (usec) ";

    if (PAD_KERNELS && precision == "int8" && inference)
        std::cout << " pad_kerenels  ";


    std::cout << std::endl;

    int pad_kernels_count = 0;

    for (const auto &problem : (inference ? inference_server_set : training_set)) {
        int m, n, k;
        bool a_t, b_t;
        std::tie(m, n, k, a_t, b_t) = problem;
        int time_ms;
        bool skip_kernel = false;
        bool need_padding = false;


#if (__CUDACC_VER_MAJOR__ >= 8)
        int pad_m;
        pad_m = m;
        if (precision == "int8") {
            if (pad_m%4) {
                pad_kernels_count++;
                if (PAD_KERNELS) {
                    pad_dim(pad_m, 4);
                    need_padding = true;
                } else {
                    skip_kernel = true;
                }
            }
        }
#endif

        std::cout << std::setw(7) << m;
        std::cout << std::setw(7) << n;
        std::cout << std::setw(7) << k;
        std::cout << std::setw(7) << a_t ? "t" : "n";
        std::cout << std::setw(7) << b_t ? "t" : "n";

        std::stringstream ss;
        ss << "Unsupported precision requested. Precision: " << precision << " Inference: " << inference;

#if (__CUDACC_VER_MAJOR__ >= 8)
        if (precision == "int8" & inference) {
            auto a = rand<uint8_t>({a_t ? k : pad_m, a_t ? pad_m : k}, curand_gen);
            auto b = rand<uint8_t>({b_t ? n : k, b_t ? k : n}, curand_gen);
            auto c = zeros<int>({pad_m, n});
            std::cout << std::setw(14) << precision;
            if (!skip_kernel)
                time_ms = time_gemm<uint8_t, int>(a, b, c, a_t, b_t, cublas_handle, store, pool);
        } else if (precision == "half") {
            auto a = rand<uint16_t>({a_t ? k : m, a_t ? m : k}, curand_gen);
            auto b = rand<uint16_t>({b_t ? n : k, b_t ? k : n}, curand_gen);
            auto c = zeros<uint16_t>({m, n});
            std::cout << std::setw(13) << precision;
            time_ms = time_gemm<uint16_t, uint16_t>(a, b, c, a_t, b_t, cublas_handle, store, pool);
        } else if (precision == "float") {
            auto a = rand<float>({a_t ? k : m, a_t ? m : k}, curand_gen);
            auto b = rand<float>({b_t ? n : k, b_t ? k : n}, curand_gen);
            auto c = zeros<float>({m, n});
            std::cout << std::setw(13) << precision;
            time_ms = time_gemm<float, float>(a, b, c, a_t, b_t, cublas_handle, store, pool);
        } else {
            throw std::runtime_error(ss.str());
        }
#else

        if (precision != "float") {
            throw std::runtime_error(ss.str());
        }

        auto a = rand<float>({a_t ? k : m, a_t ? m : k}, curand_gen);
        auto b = rand<float>({b_t ? n : k, b_t ? k : n}, curand_gen);
        auto c = zeros<float>({m, n});
        std::cout << std::setw(13) << precision;
        time_ms = time_gemm<float, float>(a, b, c, a_t, b_t, cublas_handle, store, pool);
#endif
        std::cout << std::setw(20) << std::setprecision(6);

        if (skip_kernel) {
            std::cout << "Not Supported";
        } else {
            std::cout << time_ms;
        }

        if (PAD_KERNELS && precision == "int8" && inference) {
            std::cout << std::setw(10) <<  need_padding;
        }

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


    cublasDestroy(cublas_handle);
    curandDestroyGenerator(curand_gen);

    return 0;
}
