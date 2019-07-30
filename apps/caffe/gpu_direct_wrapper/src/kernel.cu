#include <stdio.h>
#include <gdrapi.h>
#include <common/logging.h>
#include <common/dump_utils.h>
#include <core/xms.h>
#include <api/kvstore_itf.h>
#include <api/memory_itf.h>


#define CUDA_CHECK(x)  if(x != cudaSuccess) \
    PERR("error: cuda err=%s", cudaGetErrorString (cudaGetLastError()));

__global__ void verify_memory(void * ptr)
{
  char * p = (char *) ptr;
  printf("From GPU: (%.50s)\n", p);
}

extern "C" void cuda_run_test(Component::IKVStore * _kvstore, \
			      std::string key)
{ 
  printf("run_test (gpu_direct_wrapper)\n");
  Component::IKVStore::pool_t pool;
  pool = _kvstore->open_pool("./", "test1.pool");

  int nDevices;
  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate (KHz): %d\n",
           prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n",
           prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
  }

  /* allocate GDR memory */
  void* device_buffer = NULL;
  void* device_buffer1 = NULL;
  size_t buffer_size = MB(32);

  /** 
   * Here's the issue. SPDK requires 2MB aligned memory.  Because CUDA
   * does not allow alignment request, we have to allocate more than
   * we need then calculate offset to aligned physical address and
   * then re-map a new virtual address which is 2MB aligned.  We will
   * basically have to allocate a large slab and then do the
   * management ourselves.
   * 
   * Yes, its that painful.
   */
  CUDA_CHECK(cudaMalloc(&device_buffer, buffer_size));
  CUDA_CHECK(cudaMalloc(&device_buffer1, buffer_size));

  gdr_t g = gdr_open();
  //gdr_t g1 = gdr_open();
  //assert(g1);
  assert(g);
  gdr_mh_t mh;
  gdr_mh_t mh1;
  if(gdr_pin_buffer(g,
                    (unsigned long) device_buffer,
                    buffer_size,
                    0,
                    0,
                    &mh))
    throw General_exception("gdr_pin_buffer failed");

  void *bar_ptr  = NULL;
  if(gdr_map(g, mh, &bar_ptr, buffer_size), 0)
    throw General_exception("gdr_map failed");

  PLOG("gdr_map returned bar_ptr=%p", bar_ptr);
  assert(bar_ptr);
  
  gdr_info_t info;
  if(gdr_get_info(g, mh, &info))
    throw General_exception("gdr_get_info failed");
  int off = ((unsigned long)device_buffer) - info.va;
  PLOG("offset: %d", off);

  if(gdr_pin_buffer(g,
                    (unsigned long) device_buffer1,
                    buffer_size,
                    0,
                    0,
                    &mh1))
    throw General_exception("gdr_pin_buffer mh1 failed");

  if(gdr_map(g, mh1, &bar_ptr, buffer_size), 0)
    throw General_exception("gdr_map1 failed");

  PLOG("gdr_map1 returned bar_ptr=%p", bar_ptr);
  assert(bar_ptr);
  gdr_info_t info1;
  if(gdr_get_info(g, mh1, &info1))
    throw General_exception("gdr_get_info1 failed");
  int off1 = ((unsigned long)device_buffer1) - info1.va;
  PLOG("offset1: %d", off1);

  void *host_vaddr = (void*) ((char *)bar_ptr + off1);
  //addr_t host_paddr = xms_get_phys(host_vaddr);
  //PLOG("GDR vaddr=%p paddr=%p", (void*)host_vaddr, (void*)host_paddr);

/*
  addr_t new_paddr = round_up(host_paddr, MB(2));
  unsigned offset = new_paddr - host_paddr;
  void * new_vaddr = ((char*)host_vaddr)+offset;

  new_vaddr = xms_mmap((void*) 0x900000000, new_paddr, MB(2));

  PLOG("new paddr=0x%lx vaddr=0x%lx", (addr_t) new_paddr, (addr_t)new_vaddr);
  PMAJOR("memory looks OK w.r.t 2MB alignment");
*/
  memset(host_vaddr, 0xb, MB(2));
  
  PLOG("berfore get_direct: perform get");
  void * value = nullptr;
  size_t value_len = 0;
  _kvstore->get(pool, key, value, value_len);
  PINF("Get Value=(%.50s) %lu", ((char*) value), value_len);

  /* try DMA from device into this buffer... here we go... */
  PLOG("about to perform get_direct operation...");
  _kvstore->get_direct(pool, key, host_vaddr, value_len);
  PMAJOR("get_direct operation OK!");
  
  hexdump(host_vaddr, 32);

  printf("From GPU should be: (%.50s)\n", (char*) value);
  /* verify on the GPU side the result */  
  verify_memory<<<1,1>>>(((char*)device_buffer) + off);
  verify_memory<<<1,1>>>(((char*)device_buffer1)+off);
  cudaDeviceSynchronize();
}



