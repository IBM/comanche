#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <stdint.h>
#include <inttypes.h>
#include <common/utils.h>
#include <common/exceptions.h>
#include "xms.h"

enum {
  IOCTL_PMINFO_PAGE2M=1,
  IOCTL_PMINFO_PAGE1G=2,
};

// typedef struct {
//   void * region_ptr;
//   size_t region_size;
//   uint32_t flags;
//   void * out_data;
//   size_t out_size;
// } __attribute__((packed)) IOCTL_GETBITMAP_param;


// typedef struct
// {
//   addr_t vaddr;
//   addr_t out_paddr;
// }
// __attribute__((packed)) IOCTL_GETPHYS_param;

// prints LSB to MSB
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <vector>

// enum {
//   IOCTL_CMD_GETBITMAP = 9,
//   IOCTL_CMD_GETPHYS = 10,
// };
  

class Page_state_bitmap
{
private:
  static constexpr bool option_DEBUG = false;
  
  int _fd;
  
public:
  typedef std::pair<addr_t,size_t> range_t;
  typedef std::vector<range_t> range_map_t;

  
  Page_state_bitmap() {
    _fd = open("/dev/xms", O_RDWR);
    if(_fd == -1)
      throw Constructor_exception("unable to open /dev/xms - check module is loaded.");
  }

  ~Page_state_bitmap() {
    close(_fd);
  }

  void get_process_dirty_pages_compacted(void * region, size_t region_size, range_map_t& range_map)
  {
    range_map.clear();
    
    IOCTL_GETBITMAP_param ioparam;
    ioparam.ptr = region;
    ioparam.size = region_size;
    ioparam.flags = 0;
    ioparam.out_size = bitmap_size(region_size);
    ioparam.out_data = malloc(ioparam.out_size);
  
    ioctl(_fd, IOCTL_CMD_GETBITMAP, &ioparam);  //ioctl call

    IOCTL_GETBITMAP_out_param * outparam = (IOCTL_GETBITMAP_out_param *) ioparam.out_data;

    if(option_DEBUG) {
      printf("magic: %" PRIx32 "\n",outparam->magic);
      printf("bitmap_size: %" PRIx32 "\n",outparam->bitmap_size);
      print_binary(outparam->bitmap,outparam->bitmap_size);
    }
    
    parse_bitmap(region, outparam->bitmap,outparam->bitmap_size, range_map);

    free(ioparam.out_data);
  }

  void get_process_dirty_pages(void * region, size_t region_size, std::vector<addr_t>& dirty_vector)
  {
    dirty_vector.clear();

    IOCTL_GETBITMAP_param ioparam;
    ioparam.ptr = region;
    ioparam.size = region_size;
    ioparam.flags = 0;
    ioparam.out_size = bitmap_size(region_size);
    ioparam.out_data = malloc(ioparam.out_size);
  
    ioctl(_fd, IOCTL_CMD_GETBITMAP, &ioparam);  //ioctl call

    IOCTL_GETBITMAP_out_param * outparam = (IOCTL_GETBITMAP_out_param *) ioparam.out_data;

    if(option_DEBUG) {
      printf("magic: %" PRIx32 "\n",outparam->magic);
      printf("bitmap_size: %" PRIx32 "\n",outparam->bitmap_size);
      print_binary(outparam->bitmap,outparam->bitmap_size);
    }
    
    parse_bitmap_individual(region, outparam->bitmap,outparam->bitmap_size, dirty_vector);

    free(ioparam.out_data);
  }
  
private:
  
  static void parse_bitmap(void * base, char * data, size_t length, range_map_t& range_map)
  {
    assert(length > 0);

    size_t qwords = length / sizeof(uint64_t);
    size_t remaining = length % sizeof(uint64_t);
    unsigned curr_bit = 0;
    addr_t base_addr = reinterpret_cast<addr_t>(base);
    unsigned segment_len = 0;
    addr_t segment_base = 0;
    
    assert(base_addr % PAGE_SIZE == 0);
    
    uint64_t * qptr = (uint64_t *) data; 
    while(qwords > 0) {

      for(unsigned i=0;i<64;i++) {

        if(*qptr & 0x1) {
          // contiguous with current segment
          if(segment_base == 0) {
            segment_base = base_addr + (curr_bit * PAGE_SIZE);
            segment_len = 1;
          }
          else { // extends current segment
            segment_len++;
          }
        }
        else {
          if(segment_base) {
            if(option_DEBUG)
              printf("0x%lx - %d\n",segment_base, segment_len);

            range_map.push_back(range_t{segment_base, segment_len});
            segment_len = 0;
            segment_base = 0;
          }
        }
        
        *qptr = *qptr >> 1;
        curr_bit++; 
        if(*qptr == 0) { // optimization: short-circuit for zero
          curr_bit += (63 - i);
          break;
        }
      }
      
      qptr++;
      qwords--;
    }
    unsigned char * bptr = (unsigned char *) qptr;

    while(remaining > 0) {
      for(unsigned i=0;i<8;i++) {
        
        if(*bptr & 0x1) {
          // contiguous with current segment
          if(segment_base == 0) {
            segment_base = base_addr + (curr_bit * PAGE_SIZE);
            segment_len = 1;
          }
          else { // extends current segment
            segment_len++;
          }
        }
        else {
          if(segment_base) {
            if(option_DEBUG)
              printf("0x%lx - %d\n",segment_base, segment_len);
            range_map.push_back(range_t{segment_base, segment_len});
            segment_len = 0;
            segment_base = 0;
          }
        }

        *bptr = *bptr >> 1;
        curr_bit++;
        if(*bptr == 0) { // optimization: short-circuit for zero
          curr_bit += (7 - i);
          break;
        }
      }

      bptr++;
      remaining--;
    }

    if(segment_base) {
      if(option_DEBUG)
        printf("0x%lx - %d\n",segment_base, segment_len);
      range_map.push_back(range_t{segment_base, segment_len});
    }          
  }

  static void parse_bitmap_individual(void * base, char * data, size_t length, std::vector<addr_t>& dirty_vector)
  {
    assert(length > 0);

    size_t qwords = length / sizeof(uint64_t);
    size_t remaining = length % sizeof(uint64_t);
    unsigned curr_bit = 0;
    addr_t base_addr = reinterpret_cast<addr_t>(base);
    
    assert(base_addr % PAGE_SIZE == 0);

    dirty_vector.clear();
    
    uint64_t * qptr = (uint64_t *) data; 
    while(qwords > 0) {

      for(unsigned i=0;i<64;i++) {

        if(*qptr & 0x1)
          dirty_vector.push_back(base_addr + (curr_bit * PAGE_SIZE));

        *qptr = *qptr >> 1;
        curr_bit++; 
        if(*qptr == 0) { // optimization: short-circuit for zero
          curr_bit += (63 - i);
          break;
        }
      }
      
      qptr++;
      qwords--;
    }
    unsigned char * bptr = (unsigned char *) qptr;

    while(remaining > 0) {
      for(unsigned i=0;i<8;i++) {
        
        if(*bptr & 0x1)
          dirty_vector.push_back(base_addr + (curr_bit * PAGE_SIZE));

        *bptr = *bptr >> 1;
        curr_bit++;
        if(*bptr == 0) { // optimization: short-circuit for zero
          curr_bit += (7 - i);
          break;
        }
      }

      bptr++;
      remaining--;
    }
  }

  static void print_binary(char * data, size_t length)
  {
    for(unsigned i=0;i<length;i++) {
      char byte = data[i];
      for(unsigned j=0;j<8;j++) {
        if(byte & 0x1) printf("1");
        else printf("0");
        byte = byte >> 1;
      }
      printf("-");
    }
    printf("\n");
  }

  static size_t bitmap_size(size_t data_len)
  {
    size_t size = data_len / 8;
    if(data_len % 8) size++;
    return size;
  }

};

#define FLAG_NONCACHED 0x1000000000000000ULL
#define FLAG_WC 0x2000000000000000ULL

#define TEST_PHYS_ADDR 0x0000002080000000
int main()
{
  
#if 1
  int fd = open("/dev/xms", O_RDWR);
  assert(fd != -1);
  
  void * ptr = aligned_alloc(KB(4), MB(4));

  assert(ptr);
  memset(ptr, 0, MB(4));
  PLOG("allocated ptr=%p", ptr);
  

  IOCTL_GETPHYS_param ioparam;
  ioparam.vaddr = (addr_t) ptr;

  int rc = ioctl(fd, IOCTL_CMD_GETPHYS, &ioparam);  //ioctl call
  if(rc != 0) {
    PERR("ioctl failed: rc=%d", rc);
    close(fd);
    return 0;
  }

  PINF("result: 0x%lx -> 0x%lx", ioparam.vaddr, ioparam.out_paddr);
  
  close(fd);
#endif
  
#if 0
  Page_state_bitmap psb;

  size_t test_size = PAGE_SIZE * 34;
  void * m = aligned_alloc(PAGE_SIZE, test_size);
  std::vector<addr_t> ranges;
  printf("Allocated memory: %p\n", m);

  psb.get_process_dirty_pages(m, test_size, ranges);
  
  memset(m,0,test_size);

  psb.get_process_dirty_pages(m, test_size, ranges);

  ((char*)m)[4096] = 'a';
  ((char*)m)[4096*2] = 'b';
  ((char*)m)[4096*6] = 'b';
  ((char*)m)[4096*7] = 'b';
  ((char*)m)[4096*8] = 'b';
  ((char*)m)[4096*12] = 'b';
  psb.get_process_dirty_pages(m, test_size, ranges);

  for(auto& i: ranges) {
    printf("dirty: %lx\n", i);
  }
  
  free(m);
#endif
  
}
