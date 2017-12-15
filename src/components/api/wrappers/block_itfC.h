#ifdef SWIG
%module IBlock
#endif

#ifdef SWIG
%{
#endif
  
#include <stdint.h>
#include <string.h>
  
#ifdef SWIG
%}
#endif

typedef int      status_t;
typedef uint64_t io_buffer_t;

typedef struct { void * obj; } IBlock_ref; /* allows stricter type checking */



extern "C"
{
  IBlock_ref IBlock_factory__create(const char * config,
                                    unsigned long cpu_mask,
                                    const char * lib_name = nullptr);

  status_t IBlock__release(IBlock_ref ref);

  /* IZerocopy_memory */
  io_buffer_t  IBlock__allocate_io_buffer(IBlock_ref ref, size_t size, unsigned alignment, int numa_node);
  status_t     IBlock__realloc_io_buffer(IBlock_ref ref, io_buffer_t io_mem, size_t size, unsigned alignment);
  status_t     IBlock__free_io_buffer(IBlock_ref ref, io_buffer_t io_mem);
  io_buffer_t  IBlock__register_memory_for_io(IBlock_ref ref, void * vaddr, addr_t paddr, size_t len);
  void         IBlock__unregister_memory_for_io(IBlock_ref ref, void * vaddr, size_t len);
  void *       IBlock__virt_addr(IBlock_ref ref, io_buffer_t buffer);
  addr_t       IBlock__phys_addr(IBlock_ref ref, io_buffer_t buffer);
  size_t       IBlock__get_size(IBlock_ref ref, io_buffer_t buffer);

}



  



                        
