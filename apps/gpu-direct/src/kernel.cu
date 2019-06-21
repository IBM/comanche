/** 
 * Must have OFED compiled with CUDA and nv_peer_memory module loaded.
 * 
 */
#include <chrono>
#include <stdio.h>
#include <gdrapi.h>
#include <common/logging.h>
#include <common/dump_utils.h>
#include <common/utils.h>
#include <api/kvstore_itf.h>
#include <api/rdma_itf.h>

#include <core/xms.h>

#include <infiniband/verbs.h>
#include <infiniband/verbs_exp.h>
#include <cuda.h>

#define ASSERT(x)                                                       \
	do {                                                                  \
    if (!(x)) {                                                         \
      fprintf(stdout, "Assertion \"%s\" failed at %s:%d\n", #x, __FILE__, __LINE__); \
    }                                                                   \
  } while (0)

#define CUDA_CHECK(x)  if(x != cudaSuccess)                             \
    PERR("error: cuda err=%s", cudaGetErrorString (cudaGetLastError()));

#define CUCHECK(stmt)                           \
	do {                                          \
    CUresult result = (stmt);                   \
    ASSERT(CUDA_SUCCESS == result);             \
  } while (0)

static CUdevice cuDevice;
static CUcontext cuContext;

__global__ void verify_memory(void * ptr)
{
  char * p = (char *) ptr;
  printf("Viewed from GPU: %02x %02x %02x ...\n", p[0], p[1], p[2]);
}

void initialize()
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
		PINF("There are %d devices supporting CUDA, picking first...", deviceCount);

  int devID = 0;
  
	/* pick up device with zero ordinal (default, or devID) */
	CUCHECK(cuDeviceGet(&cuDevice, devID));

  char name[128];
	CUCHECK(cuDeviceGetName(name, sizeof(name), devID));
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
  

extern "C" void run_cuda_basic_test(Component::IKVStore * store)
{ 
  PINF("** run_test_basic_test (cuda app lib)");

  initialize();

  CUresult error;
  const size_t obj_size = MB(128);
  CUdeviceptr d_A;

  /* allocate GPU side memory */
	error = cuMemAlloc(&d_A, obj_size);
	if (error != CUDA_SUCCESS)
		throw General_exception("cuMemAlloc error=%d", error);
  
	PINF("allocated GPU buffer address at %p", d_A);

  cuMemsetD8(d_A, 0xBB, obj_size);
  verify_memory<<<1,1>>>((char*)d_A);

  /* register memory with RDMA engine */
  auto handle = store->register_direct_memory((void*)d_A, obj_size);
  assert(handle);
  PLOG("registered memory with storage/RDMA.. OK (handle=%p)", handle);
  
  /* create pool */
  auto pool = store->create_pool("/pools","gpu0", MiB(2));
  PLOG("Pool created OK.");
  
  /* put into dawn storage */
  status_t rc;
  auto start = std::chrono::high_resolution_clock::now();

  constexpr unsigned ITERATIONS = 10;
  
  for(unsigned i=0;i<ITERATIONS;i++) {
    char tmp[255];
    sprintf(tmp,"key%u", i);
    rc = store->put_direct(pool, tmp, (void*)d_A, obj_size, handle);
    if(rc != S_OK)
      PLOG("put_direct returned: %d", rc);
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto secs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;
  PMAJOR("GPU-to-Dawn Throughput: %f MB/s", ((float) REDUCE_MB(obj_size * ITERATIONS)) / secs);

  /* zero memory on GPU */
  cuMemsetD8(d_A, 0x0, obj_size);
  cudaDeviceSynchronize();
  verify_memory<<<1,1>>>((char*)d_A);

  /* reload memory from store */
  size_t rsize = obj_size;
  assert(rsize > 0);
  
  start = std::chrono::high_resolution_clock::now();

  for(unsigned i=0;i<ITERATIONS;i++) {
    char tmp[255];
    sprintf(tmp,"key%u", i);
    
    rc = store->get_direct(pool, tmp, (void*)d_A, rsize, handle);
    if(rc != S_OK) 
      throw General_exception("get_direct: returned %d", rc);

    if(rsize != obj_size) PWRN("rsize=%lu obj_size=%lu", rsize, obj_size);
  }
  
  end = std::chrono::high_resolution_clock::now();
  secs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;
  PMAJOR("Dawn-to-GPU Throughput: %f MB/s", ((float) REDUCE_MB(obj_size * ITERATIONS)) / secs);

  /* re-verify from GPU side and voila!! */
  cudaDeviceSynchronize();
  verify_memory<<<1,1>>>((char*)d_A);

  cudaDeviceSynchronize();
    
  /* clean up */
  cuMemFree(d_A);
  store->unregister_direct_memory(handle);
  store->close_pool(pool);
}


void run_cuda_perf_direct(Component::IKVStore * store)
{
  PINF("** run_cuda_perf_direct (cuda app lib)");

  initialize();
  const size_t obj_size = MB(128);
  CUdeviceptr d_A;
  CUresult error;

  /* allocate GPU side memory */
	error = cuMemAlloc(&d_A, obj_size);
	if (error != CUDA_SUCCESS)
		throw General_exception("cuMemAlloc error=%d", error);  
	PINF("allocated GPU buffer address at %p", d_A);

  /* register memory with RDMA engine */
  auto handle = store->register_direct_memory((void*)d_A, obj_size);
  assert(handle);
  PLOG("registered memory with storage/RDMA.. OK (handle=%p)", handle);
  
  /* create pool */
  auto pool = store->create_pool("/poolsgpu0", obj_size);
  if (pool == Component::IKVStore::POOL_ERROR) {
	  throw General_exception("Create pool error");
  }
  PLOG("Pool created OK.");
  
  /* put into dawn storage */
  status_t rc;
  auto start = std::chrono::high_resolution_clock::now();

  constexpr unsigned ITERATIONS = 100;
  
  for(unsigned i=0;i<ITERATIONS;i++) {
    rc = store->put_direct(pool, "key0", (void*)d_A, obj_size, handle);
    if(rc != S_OK)
      PLOG("put_direct returned: %d", rc);
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto secs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;
  PMAJOR("GPU-to-Dawn GPU-Direct Throughput: %f MB/s", ((float) REDUCE_MB(obj_size * ITERATIONS)) / secs);

  /* reload memory from store */
  size_t rsize = obj_size;
  assert(rsize > 0);
  
  start = std::chrono::high_resolution_clock::now();

  for(unsigned i=0;i<ITERATIONS;i++) {
    rc = store->get_direct(pool, "key0", (void*)d_A, rsize, handle);
    if(rc != S_OK) 
      throw General_exception("get_direct: returned %d", rc);

    assert(rsize == obj_size);
  }
  
  end = std::chrono::high_resolution_clock::now();
  secs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;
  PMAJOR("Dawn-to-GPU GPU-Direct Throughput: %f MB/s", ((float) REDUCE_MB(obj_size * ITERATIONS)) / secs);

  cudaDeviceSynchronize();
  
  /* clean up */
  cuMemFree(d_A);
  store->unregister_direct_memory(handle);
  store->close_pool(pool);
  store->delete_pool("/poolsgpu0");
}

void run_cuda_perf_bounce(Component::IKVStore * store)
{
  PINF("** run_cuda_perf_bounce (cuda app lib)");

  initialize();
  const size_t obj_size = MB(128);
  void * d_A;
  CUdeviceptr d_B;
  CUresult error;

  /* allocate DRAM memory */  
  error = cuMemAllocHost(&d_A, obj_size);
  if (error != CUDA_SUCCESS)
		throw General_exception("cuMemAllocHost error=%d", error);  
	assert(d_A);  
	PINF("allocated DRAM buffer address at %p", d_A);

  /* allocate GPU memory */
	error = cuMemAlloc(&d_B, obj_size);
	if (error != CUDA_SUCCESS)
		throw General_exception("cuMemAlloc error=%d", error);  
	PINF("allocated GPU buffer address at %p", d_B);
  
  /* register memory with RDMA engine */
  auto handle = store->register_direct_memory((void*)d_A, obj_size);
  assert(handle);
  PLOG("registered memory with storage/RDMA.. OK (handle=%p)", handle);
  
  /* create pool */
  auto pool = store->create_pool("/poolsgpu0", obj_size);
  if (pool == Component::IKVStore::POOL_ERROR) {
  	  throw General_exception("Create pool error");
  }
  PLOG("Pool created OK.");
  
  /* put into dawn storage */
  status_t rc;
  auto start = std::chrono::high_resolution_clock::now();

  constexpr unsigned ITERATIONS = 100;
  
  for(unsigned i=0;i<ITERATIONS;i++) {
    error = cuMemcpyDtoH(d_A, d_B, obj_size);
    assert(error == CUDA_SUCCESS);
    
    rc = store->put_direct(pool, "key0", (void*)d_A, obj_size, handle);
    if(rc != S_OK)
      PLOG("put_direct returned: %d", rc);
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto secs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;
  PMAJOR("GPU-to-Dawn Bounce Throughput: %f MB/s", ((float) REDUCE_MB(obj_size * ITERATIONS)) / secs);

  /* reload memory from store */
  size_t rsize = obj_size;
  assert(rsize > 0);
  
  start = std::chrono::high_resolution_clock::now();

  for(unsigned i=0;i<ITERATIONS;i++) {

    rc = store->get_direct(pool, "key0", (void*)d_A, rsize, handle);
    if(rc != S_OK) 
      throw General_exception("get_direct: returned %d", rc);    
    assert(rsize == obj_size);

    error = cuMemcpyHtoD(d_B, d_A, obj_size);
    assert(error == CUDA_SUCCESS);
  }
  
  end = std::chrono::high_resolution_clock::now();
  secs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;
  PMAJOR("Dawn-to-GPU Bounce Throughput: %f MB/s", ((float) REDUCE_MB(obj_size * ITERATIONS)) / secs);

  cudaDeviceSynchronize();
  
  /* clean up */
  cuMemFreeHost(d_A);
  cuMemFree(d_B);
  store->unregister_direct_memory(handle);
  store->close_pool(pool);
  store->delete_pool("/poolsgpu0");
}

extern "C" void run_cuda_perf(Component::IKVStore * store)
{
  assert(store);
  run_cuda_perf_direct(store);
  run_cuda_perf_bounce(store);
}

