#define MCAS_MINOR 100

typedef uint64_t addr_t;

enum {
  IOCTL_CMD_EXPOSE = 1,
};

typedef struct
{
  uint64_t token; /* token that must be used with the mmap call */
  void*    vaddr; /* address of memory to share (from calling process perspective) */
  size_t   vaddr_size; /* size of region to share */
}
 __attribute__((packed)) IOCTL_EXPOSE_msg;

// enum {
//   IOCTL_CMD_GETBITMAP = 9,
//   IOCTL_CMD_GETPHYS = 10,
//   IOCTL_CMD_SETMEMORY = 11,
// };

// typedef enum {
//   MEMORY_UC = 1,
//   MEMORY_WB = 2,
//   MEMORY_WT = 3,
// } memory_t;

// typedef struct 
// {
//   void * ptr;
//   size_t size;
//   uint32_t    flags;
//   void * out_data;
//   size_t out_size;
// }
// __attribute__((packed)) IOCTL_GETBITMAP_param;

// typedef struct {
//   uint32_t magic;
//   uint32_t bitmap_size;
//   char     bitmap[0];
// } __attribute__((packed)) IOCTL_GETBITMAP_out_param;

// typedef struct
// {
//   addr_t vaddr;
//   addr_t out_paddr;
// }
// __attribute__((packed)) IOCTL_GETPHYS_param;


// typedef struct
// {
//   void*    vaddr;
//   memory_t type;
//   size_t   size;
// }
// __attribute__((packed)) IOCTL_SETMEMORY_param;



