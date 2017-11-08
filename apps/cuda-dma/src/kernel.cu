#include <stdio.h>
#include <gdrapi.h>
#include <common/logging.h>
#include <common/dump_utils.h>
#include <core/xms.h>
#include <api/block_itf.h>
#include <api/memory_itf.h>


#define CUDA_CHECK(x)  if(x != cudaSuccess) \
    PERR("error: cuda err=%s", cudaGetErrorString (cudaGetLastError()));

__global__ void verify_memory(void * ptr)
{
  char * p = (char *) ptr;
  printf("From GPU: should be 0f 0f 0f ...\n");
  printf("From GPU: %02x %02x %02x ...\n", p[0], p[1], p[2]);
}

extern "C" void cuda_run_test(Component::IBlock_device * block_device)
{ 
  printf("run_test (cuda app lib)\n");

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

  Component::VOLUME_INFO vi;
  block_device->get_volume_info(vi);
  PLOG("volume info: blocksize=%d", vi.block_size);

  /* allocate GDR memory */
  void* device_buffer = NULL;
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

  gdr_t g = gdr_open();
  assert(g);
  gdr_mh_t mh;
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

  void *host_vaddr = (void*) ((char *)bar_ptr + off);
  addr_t host_paddr = xms_get_phys(host_vaddr);
  PLOG("GDR vaddr=%p paddr=%p", (void*)host_vaddr, (void*)host_paddr);


  addr_t new_paddr = round_up(host_paddr, MB(2));
  unsigned offset = new_paddr - host_paddr;
  void * new_vaddr = ((char*)host_vaddr)+offset;

  new_vaddr = xms_mmap((void*) 0x900000000, new_paddr, MB(2));

  PLOG("new paddr=0x%lx vaddr=0x%lx", (addr_t) new_paddr, (addr_t)new_vaddr);
  PMAJOR("memory looks OK w.r.t 2MB alignment");

  memset(new_vaddr, 0xb, KB(4));

  /* register GDR memory with block device */
  Component::io_buffer_t iob = block_device->register_memory_for_io(new_vaddr,
                                                                    new_paddr,
                                                                    MB(2));
  
  /* try DMA from device into this buffer... here we go... */
  PLOG("about to perform read operation...");
  block_device->read(iob, 0, 1, 1, 0);
  PMAJOR("read operation OK!");
  
  hexdump(new_vaddr, 32);

  /* verify on the GPU side the result */  
  verify_memory<<<1,1>>>(((char*)device_buffer) + offset);
  cudaDeviceSynchronize();
}



