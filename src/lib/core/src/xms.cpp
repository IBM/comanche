#include <core/xms.h>
#include <sys/mman.h>

extern "C" addr_t xms_get_phys(void * vaddr)
{
  enum {
    IOCTL_CMD_GETBITMAP = 9,
    IOCTL_CMD_GETPHYS = 10,
  };

  typedef struct
  {
    addr_t vaddr;
    addr_t out_paddr;
  } __attribute__((packed)) IOCTL_GETPHYS_param;

  /* use xms to get physical memory address  */
  IOCTL_GETPHYS_param ioparam = {0};
  {
    int fd = open("/dev/xms", O_RDWR);    

    ioparam.vaddr = (addr_t) vaddr;

    int rc = ioctl(fd, IOCTL_CMD_GETPHYS, &ioparam);  //ioctl call
    if(rc != 0) {
      PERR("%s(): ioctl failed on xms module: %s\n",
              __func__, strerror(errno));
    }
    close(fd);
  }
  return ioparam.out_paddr;
}

extern "C" void * xms_mmap(void* vaddr, addr_t paddr, size_t size)
{
  int fd = open("/dev/xms", O_RDWR, 0666);
  if(fd == -1)
    throw General_exception("unable to open /dev/xms");
  
  void *ptr = mmap(vaddr,
                   size,
                   PROT_READ | PROT_WRITE,
                   MAP_FIXED | MAP_SHARED,
                   fd,
                   paddr);  // offset in the file is physical addres
  
  close(fd);
  if(ptr == (void*)-1)
    throw General_exception("mmap failed on xms module");
  
  return ptr;
}


