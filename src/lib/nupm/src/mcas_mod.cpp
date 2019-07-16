#include "mcas_mod.h"
#include <common/errors.h>
#include <common/logging.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <fcntl.h>

/* todo: unify with module header file */

//------------------------------------------------------------
enum {
  IOCTL_CMD_EXPOSE = 8,
  IOCTL_CMD_REMOVE = 9,
  IOCTL_CMD_QUERY  = 10,
};

typedef struct
{
  uint64_t token; /* token that must be used with the mmap call */
  void*    vaddr; /* address of memory to share (from calling process perspective) */
  size_t   vaddr_size; /* size of region to share */
}
 __attribute__((packed)) IOCTL_EXPOSE_msg;

typedef struct
{
  uint64_t token; /* token of previously exposed memory */
}
__attribute__((packed)) IOCTL_REMOVE_msg;  

typedef struct
{
  union {
    uint64_t token; /* token of previously exposed memory */
    size_t size;
  };
}
__attribute__((packed)) IOCTL_QUERY_msg;  
//------------------------------------------------------------


bool nupm::check_mcas_kernel_module()
{
  int fd = open("/dev/mcas", O_RDWR, 0666);
  close(fd);
  return (fd != -1);
}  

status_t nupm::expose_memory(Memory_token token, void * vaddr, size_t vaddr_size)
{
  int fd = open("/dev/mcas", O_RDWR, 0666);
  if(fd == -1) {
    PERR("nupm::expose_memory: cannot access /dev/mcas");
    return E_FAIL;
  }

  IOCTL_EXPOSE_msg ioparam = {token, vaddr, vaddr_size};

  /* call kernel module to expose memory */
  status_t rc = ioctl(fd, IOCTL_CMD_EXPOSE, &ioparam); 
  close(fd);

  return rc;
}

status_t nupm::revoke_memory(Memory_token token)
{
  int fd = open("/dev/mcas", O_RDWR, 0666);
  if(fd == -1) {
    PERR("nupm::revoke_memory: cannot access /dev/mcas");
    return E_FAIL;
  }

  IOCTL_REMOVE_msg ioparam = {token};
  status_t rc = ioctl(fd, IOCTL_CMD_REMOVE, &ioparam);
  close(fd);
  
  return rc;
}

void * nupm::mmap_exposed_memory(Memory_token token,
                                 size_t& size,
                                 void* target_addr)
{
  int fd = open("/dev/mcas", O_RDWR, 0666);
  if(fd == -1) {
    PERR("nupm::map_exposed_memory: cannot access /dev/mcas");
    return nullptr;
  }

  /* use ioctl to get size of area before calling mmap */
  IOCTL_QUERY_msg msg = {token};
  status_t rc = ioctl(fd, IOCTL_CMD_QUERY, &msg);

  if(rc) {
    PERR("nupm::mmap_exposed_memory ioctl failed, with err code %d", rc);
    return nullptr;
  }
  
  size = msg.size;
  
  offset_t offset = ((offset_t)token) << 12; /* must be 4KB aligned */

  void * ptr = ::mmap(target_addr,
                      size,
                      PROT_READ | PROT_WRITE,
                      MAP_SHARED | MAP_FIXED, // | MAP_HUGETLB, // | MAP_HUGE_2MB,
                      fd,
                      offset); 

  close(fd);  
  return ptr;
}
