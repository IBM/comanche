#define XMS_MINOR 100
#undef DEBUG

typedef unsigned long addr_t;

enum {
  IOCTL_CMD_GETBITMAP = 9,
  IOCTL_CMD_GETPHYS = 10,
};

typedef struct 
{
  void * ptr;
  size_t size;
  u32    flags;
  void * out_data;
  size_t out_size;
}
__attribute__((packed)) IOCTL_GETBITMAP_param;

typedef struct
{
  addr_t vaddr;
  addr_t out_paddr;
}
__attribute__((packed)) IOCTL_GETPHYS_param;


