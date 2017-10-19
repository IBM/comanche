#include <stdio.h>

#include <common/logging.h>

__managed__ int result;

__global__ void memory_test(void * ptr, size_t len)
{
  int tid = threadIdx.x;
  
  printf("Inside GPU!!: thread %d\n", tid);
  printf("ptr=%p size=%ld\n", ptr, len);

  char * p = (char *) ptr;
  for(unsigned i=0;i<10;i++) {
    printf("%c",p[i]+tid);
  }
  printf("\n");
  result = 666;
}

#define CUDA_CHECK(x)  if(x != cudaSuccess) \
    PERR("error: cuda err=%s", cudaGetErrorString (cudaGetLastError()));


extern "C" void cuda_run_test(void * ptr, size_t len)
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


  /* register SPDK allocated memory with CUDA */
  CUDA_CHECK(cudaHostRegister(ptr, len, cudaHostRegisterPortable));
  cudaPointerAttributes attr;
  CUDA_CHECK(cudaPointerGetAttributes (&attr,ptr));
  PLOG("register memory attrs: isManaged=%d", attr.isManaged);
  
  /* allocate memory on device and transfer from passed in buffer */
  void * dev_buffer;
  CUDA_CHECK(cudaMalloc(&dev_buffer, len));
  CUDA_CHECK(cudaMemcpy(dev_buffer, ptr, len, cudaMemcpyHostToDevice));

  /* pinned memory */
  void * dev_buffer2;
  void * buffer2;
  CUDA_CHECK(cudaHostAlloc(&buffer2, len, cudaHostAllocPortable | cudaHostAllocMapped));
  memset(buffer2,'a',len);
  CUDA_CHECK(cudaMalloc(&dev_buffer2, len));
  CUDA_CHECK(cudaMemcpy(dev_buffer2, buffer2, len, cudaMemcpyHostToDevice));

  /* slow pageable test */
  void * dev_buffer3;
  void * buffer3 = malloc(len);
  memset(buffer3,'b',len);
  CUDA_CHECK(cudaMalloc(&dev_buffer3, len));
  CUDA_CHECK(cudaMemcpy(dev_buffer3, buffer3, len, cudaMemcpyHostToDevice));

  
  memory_test<<<1,1>>>(dev_buffer, len);
  cudaDeviceSynchronize();

  CUDA_CHECK(cudaFree(dev_buffer));
  CUDA_CHECK(cudaFree(dev_buffer2));
  CUDA_CHECK(cudaFree(dev_buffer3));
  //  CUDA_CHECK(cudaHostFree(buffer2));
  printf("run_test result=%d\n", result);
}



