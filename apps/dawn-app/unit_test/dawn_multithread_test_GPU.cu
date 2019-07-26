/*
 * dawn_multithread_test.cpp
 *
 *  Created on: Jul 15, 2019
 *      Author: Yanzhao Wu
 */

#include <api/components.h>
#include <api/kvstore_itf.h>
#include <common/cpu.h>
#include <common/str_utils.h>
#include <core/dpdk.h>
#include <core/task.h>
#include <gtest/gtest.h>
#include <sys/mman.h>
#include <boost/program_options.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <iostream>

#include <chrono> /* milliseconds */
#include <cuda.h>



#define TEST_MULTIPLE_SHARD_CLIENT
#define TEST_SINGLE_SHARD_MULTIPLE_CLIENT

#define TEST_BARRIER

#ifdef TEST_BARRIER
#include <boost/thread/barrier.hpp>
boost::barrier * g_barrier;
#endif

static constexpr unsigned long ITERATIONS = 10;
static constexpr unsigned long VALUE_SIZE = MB(16);
static constexpr unsigned long KEY_SIZE   = 8;

struct {
  std::string addr;
  std::string pool_prefix;
  std::string device;
  unsigned    debug_level;
  unsigned    base_core;
  unsigned    port_base;
  unsigned	  num_thread;
} Options;

struct record_t {
  std::string key;
  char        value[VALUE_SIZE];
};

using namespace Component;

namespace {

	void initialize(unsigned _id, CUdevice & cuDevice, CUcontext & cuContext)
	{
	  	CUresult error = cuInit(0);
		if (error != CUDA_SUCCESS) {
			PINF("cuInit(0) returned %d", error);
			exit(1);
		}
	
		int deviceCount = 0;
		error = cuDeviceGetCount(&deviceCount);
		if (error != CUDA_SUCCESS) {
			PINF("cuDeviceGetCount() returned %d", error);
			exit(1);
		}
		/* This function call returns 0 if there are no CUDA capable devices. */
		if (deviceCount == 0) {
			throw General_exception("There are no available device(s) that support CUDA");
		} else if (deviceCount == 1)
			PINF("There is 1 device supporting CUDA");
		else
			PINF("There are %d devices supporting CUDA, picking %d", deviceCount, _id);
		ASSERT_TRUE(_id < deviceCount);	
	 	 int devID = _id;
	  
		/* pick up device with zero ordinal (default, or devID) */
		ASSERT_FALSE(cuDeviceGet(&cuDevice, devID)!=CUDA_SUCCESS);
	
	  	char name[128];
		ASSERT_FALSE(cuDeviceGetName(name, sizeof(name), devID)!=CUDA_SUCCESS);
		PINF("[pid = %d, dev = %d] device name = [%s]", getpid(), cuDevice, name);
		PINF("creating CUDA Ctx");
	
		/* Create context */
		error = cuCtxCreate(&cuContext, CU_CTX_MAP_HOST, cuDevice);
		if (error != CUDA_SUCCESS) {
			throw General_exception("cuCtxCreate() error=%d", error);
		}
	
		PINF("making it the current CUDA Ctx");
		error = cuCtxSetCurrent(cuContext);
		if (error != CUDA_SUCCESS) {
			throw General_exception("cuCtxSetCurrent() error=%d", error);
		}
	}


	// mutex for display
	boost::mutex g_display_mutex;
	// mutex for generating data
	boost::mutex gen_mutex;

	// Generate Test Data
	// Generate a single shard
	void generateTestData(IKVStore * _dawn, std::string _pool_name,
			std::vector<std::string> keys, unsigned _id) {
		IKVStore::pool_t pool = _dawn->create_pool(_pool_name, MB(256));
		ASSERT_FALSE(pool == IKVStore::POOL_ERROR);
		size_t data_size = sizeof(record_t) * ITERATIONS;
		record_t * data = (record_t *) aligned_alloc(MiB(2), data_size);
		ASSERT_FALSE(data == nullptr);
		auto handle = _dawn->register_direct_memory(data, data_size);
		// generate data
		for (unsigned long i = 0; i < ITERATIONS; i++) {
			gen_mutex.lock();
			std::string val = Common::random_string(VALUE_SIZE);
			gen_mutex.unlock();
			data[i].key = keys[i];
			memcpy(data[i].value, val.c_str(), VALUE_SIZE);
		}
		int rc;
		auto start = std::chrono::high_resolution_clock::now();
		for (unsigned long i = 0; i < ITERATIONS; i++) {
			rc = _dawn->put_direct(pool, data[i].key, data[i].value, VALUE_SIZE, handle);
			ASSERT_TRUE(rc == S_OK || rc == -6);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto micro_secs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

		//output results
		g_display_mutex.lock();
		PMAJOR("Thread ID: %d PutDirect Latency: %lf us", _id, (micro_secs/(double)ITERATIONS));
		g_display_mutex.unlock();

		_dawn->unregister_direct_memory(handle);
		_dawn->close_pool(pool);
		free(data);
		//_dawn->delete_pool(_pool_name); // TODO: After test
	}

	// Get_direct Test
	void getDirectTest(unsigned _debug_level, std::string _serverPort, std::string _device,
			std::string _pool_name, std::vector<std::string> keys, unsigned _id) {
		CUdevice cuDevice;
		CUcontext cuContext;
		initialize(_id, cuDevice, cuContext);
		Component::IBase * comp = Component::load_component("libcomanche-dawn-client.so", dawn_client_factory);
		ASSERT_TRUE(comp);
		IKVStore_factory * fact = (IKVStore_factory *) comp->query_interface(IKVStore_factory::iid());
		IKVStore * _dawn = fact ->create(_debug_level, "dawnTest", _serverPort, _device);
		ASSERT_TRUE(_dawn);
		fact->release_ref();

		// Test get direct
		IKVStore::pool_t pool = _dawn->open_pool(_pool_name);
		ASSERT_FALSE(pool == IKVStore::POOL_ERROR);

		PLOG("Allocating buffer for test data ...");

		size_t data_size = VALUE_SIZE * ITERATIONS;
		CUdeviceptr d_ptr;
		CUresult error = cuMemAlloc(&d_ptr, data_size);
		ASSERT_FALSE(error != CUDA_SUCCESS);
		auto handle = _dawn->register_direct_memory((void*)d_ptr, data_size);
		char * data = (char *) d_ptr;
		int rc;

		auto start = std::chrono::high_resolution_clock::now();
		for (unsigned long i = 0; i < ITERATIONS; i++) {
			size_t value_len = VALUE_SIZE;
			rc = _dawn->get_direct(pool, keys[i], &data[i*VALUE_SIZE], value_len, handle);
			ASSERT_TRUE(rc == S_OK || rc == -6);
#ifdef TEST_BARRIER
			g_barrier -> wait();
#endif
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto micro_secs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

		g_display_mutex.lock();
		PMAJOR("Thread ID: %d GetDirect Latency: %lf us", _id, (micro_secs/(double)ITERATIONS));
		g_display_mutex.unlock();

		_dawn->unregister_direct_memory(handle);
		_dawn->close_pool(pool);
		_dawn->delete_pool(_pool_name);
	}
#ifdef TEST_MULTIPLE_SHARD_CLIENT
	TEST(Dawn_client_multithread_test, MultipleShardsAndClients) {
		std::vector<std::string> keys;
		for(unsigned long i = 0; i < ITERATIONS; i++) {
			std::string key = Common::random_string(KEY_SIZE);
			keys.push_back(key);
		}

		Component::IBase * comp = Component::load_component("libcomanche-dawn-client.so", dawn_client_factory);
                ASSERT_TRUE(comp);
		std::vector<boost::thread> threads;
		IKVStore_factory * fact = (IKVStore_factory *) comp->query_interface(IKVStore_factory::iid());
		// Perform generateTestData
		for(unsigned i = 0; i < Options.num_thread; i++) {
			IKVStore * _dawn = fact ->create(Options.debug_level, "dawnTest",
					Options.addr+":"+std::to_string(Options.port_base+i), Options.device);
			threads.push_back(boost::thread(generateTestData,_dawn, Options.pool_prefix+std::to_string(i), keys, i));
		}
		for(unsigned i = 0; i < Options.num_thread; i++) {
			threads[i].join();
		}
		fact->release_ref();
		// Perform getDirectTest
		threads.clear();
		for(unsigned i = 0; i < Options.num_thread; i++) {
			threads.push_back(boost::thread(getDirectTest, Options.debug_level, Options.addr+":"+std::to_string(Options.port_base+i),
					Options.device, Options.pool_prefix+std::to_string(i), keys, i));
		}
		for(unsigned i = 0; i < Options.num_thread; i++) {
			threads[i].join();
		}
	}
#endif

#ifdef TEST_SINGLE_SHARD_MULTIPLE_CLIENT
	TEST(Dawn_client_multithread_test, SingleShardMultipleClients) {
		std::vector<std::string> keys;
		for(unsigned long i = 0; i < ITERATIONS; i++) {
			std::string key = Common::random_string(KEY_SIZE);
			keys.push_back(key);
		}

		std::vector<boost::thread> threads;
		Component::IBase * comp = Component::load_component("libcomanche-dawn-client.so", dawn_client_factory);
                ASSERT_TRUE(comp);
		// Perform generateTestData
		IKVStore_factory * fact = (IKVStore_factory *) comp->query_interface(IKVStore_factory::iid());
		IKVStore * _dawn = fact ->create(Options.debug_level, "dawnTest",
					Options.addr+":"+std::to_string(Options.port_base), Options.device);
		generateTestData(_dawn, Options.pool_prefix, keys, 0);
		fact->release_ref();
		// Perform getDirectTest
		threads.clear();
		for(unsigned i = 0; i < Options.num_thread; i++) {
			threads.push_back(boost::thread(getDirectTest, Options.debug_level, Options.addr+":"+std::to_string(Options.port_base),
					Options.device, Options.pool_prefix, keys, i));
		}
		for(unsigned i = 0; i < Options.num_thread; i++) {
			threads[i].join();
		}
	}
#endif

}

// Main Function
int main(int argc, char **argv)
{
  try {
    namespace po = boost::program_options;
    po::options_description desc("Options");
    desc.add_options()("help", "Show help")
      ("debug", po::value<unsigned>()->default_value(0), "Debug level 0-3")
      ("server-addr", po::value<std::string>()->default_value("10.0.0.91"), "Server address IP")
	  ("port_base", po::value<unsigned>()->default_value(11900), "Port base")
      ("device", po::value<std::string>()->default_value("mlx5_0"), "Network device (e.g., mlx5_0)")
      ("pool", po::value<std::string>()->default_value("myPool"), "Pool name prefix")
      ("base", po::value<unsigned>()->default_value(0), "Base core")
	  ("num_thread", po::value<unsigned>()->default_value(1), "Number of threads.");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    if (vm.count("help") > 0) {
      std::cout << desc;
      return -1;
    }

    Options.addr        = vm["server-addr"].as<std::string>();
    Options.port_base   = vm["port_base"].as<unsigned>();
    Options.debug_level = vm["debug"].as<unsigned>();
    Options.pool_prefix = vm["pool"].as<std::string>();
    Options.device      = vm["device"].as<std::string>();
    Options.base_core   = vm["base"].as<unsigned>();
    Options.num_thread  = vm["num_thread"].as<unsigned>();

#ifdef TEST_BARRIER
    g_barrier = new boost::barrier(Options.num_thread);
#endif
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
  }
  catch (...) {
    PLOG("bad command line option configuration");
    return -1;
  }

  return 0;
}
