/*
   Copyright [2017] [IBM Corporation]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

/* 
 * Authors: 
 * 
 * Daniel G. Waddington (daniel.waddington@ibm.com)
 *
 */

#ifndef __PAGE_BITMAP_H__
#define __PAGE_BITMAP_H__

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <stdint.h>
#include <inttypes.h>
#include <common/exceptions.h>
#include <common/utils.h>
#include <vector>

namespace Core
{
class Page_state_bitmap
{
 private:
  static constexpr bool option_DEBUG = false;

  enum {
    IOCTL_PMINFO_PAGE2M = 1,
    IOCTL_PMINFO_PAGE1G = 2,
  };

  typedef struct {
    void *   region_ptr;
    size_t   region_size;
    uint32_t flags;
    void *   out_data;
    size_t   out_size;
  } __attribute__((packed)) IOCTL_param;

  typedef struct {
    uint32_t magic;
    uint32_t bitmap_size;
    char     bitmap[0];
  } __attribute__((packed)) IOCTL_out_param;

  enum {
    IOCTL_CMD_GETBITMAP = 9,
  };

  int _fd;

 public:
  typedef std::pair<addr_t, size_t> range_t;
  typedef std::vector<range_t> range_map_t;

  /** 
   * Constructor
   * 
   */
  Page_state_bitmap()
  {
    _fd = open("/dev/xms", O_RDWR);  // requires kivati kernel module
    if (_fd == -1) throw Constructor_exception("unable to open /dev/xms - check module is loaded.");
  }

  virtual ~Page_state_bitmap()
  {
    close(_fd);
  }

  void get_process_dirty_pages_compacted(void *region, size_t region_size, range_map_t &range_map)
  {
    range_map.clear();

    /* flush cache */
    clflush_area(region, region_size);

    IOCTL_param ioparam;
    ioparam.region_ptr  = region;
    ioparam.region_size = region_size;
    ioparam.flags       = 0;
    ioparam.out_size    = bitmap_size(region_size);
    ioparam.out_data    = malloc(ioparam.out_size);

    ioctl(_fd, IOCTL_CMD_GETBITMAP, &ioparam);  //ioctl call

    IOCTL_out_param *outparam = (IOCTL_out_param *) ioparam.out_data;

    if (option_DEBUG) {
      printf("magic: %" PRIx32 "\n", outparam->magic);
      printf("bitmap_size: %" PRIx32 "\n", outparam->bitmap_size);
      print_binary(outparam->bitmap, outparam->bitmap_size);
    }

    parse_bitmap(region, outparam->bitmap, outparam->bitmap_size, range_map);

    free(ioparam.out_data);
  }

  void get_process_dirty_pages(void *region, size_t region_size, std::vector<addr_t> &dirty_vector)
  {
    dirty_vector.clear();

    /* flush cache */
    clflush_area(region, region_size);

    IOCTL_param ioparam;
    ioparam.region_ptr  = region;
    ioparam.region_size = region_size;
    ioparam.flags       = 0;
    ioparam.out_size    = bitmap_size(region_size) + 8;
    ioparam.out_data    = malloc(ioparam.out_size);

    ioctl(_fd, IOCTL_CMD_GETBITMAP, &ioparam);  //ioctl call

    IOCTL_out_param *outparam = (IOCTL_out_param *) ioparam.out_data;

    if (option_DEBUG) {
      printf("magic: %" PRIx32 "\n", outparam->magic);
      printf("bitmap_size: %" PRIx32 "\n", outparam->bitmap_size);
      print_binary(outparam->bitmap, outparam->bitmap_size);
    }

    parse_bitmap_individual(region, outparam->bitmap, outparam->bitmap_size, dirty_vector);

    free(ioparam.out_data);
  }

 private:
  static void parse_bitmap(void *base, char *data, size_t length, range_map_t &range_map)
  {
    assert(length > 0);

    size_t   qwords       = length / sizeof(uint64_t);
    size_t   remaining    = length % sizeof(uint64_t);
    unsigned curr_bit     = 0;
    addr_t   base_addr    = reinterpret_cast<addr_t>(base);
    unsigned segment_len  = 0;
    addr_t   segment_base = 0;

    assert(base_addr % PAGE_SIZE == 0);

    uint64_t *qptr = (uint64_t *) data;
    while (qwords > 0) {
      for (unsigned i = 0; i < 64; i++) {
        if (*qptr & 0x1) {
          // contiguous with current segment
          if (segment_base == 0) {
            segment_base = base_addr + (curr_bit * PAGE_SIZE);
            segment_len  = 1;
          }
          else {  // extends current segment
            segment_len++;
          }
        }
        else {
          if (segment_base) {
            if (option_DEBUG) printf("0x%lx - %d\n", segment_base, segment_len);

            range_map.push_back(range_t{segment_base, segment_len});
            segment_len  = 0;
            segment_base = 0;
          }
        }

        *qptr = *qptr >> 1;
        curr_bit++;
        if (*qptr == 0) {  // optimization: short-circuit for zero
          curr_bit += (63 - i);
          break;
        }
      }

      qptr++;
      qwords--;
    }
    unsigned char *bptr = (unsigned char *) qptr;

    while (remaining > 0) {
      for (unsigned i = 0; i < 8; i++) {
        if (*bptr & 0x1) {
          // contiguous with current segment
          if (segment_base == 0) {
            segment_base = base_addr + (curr_bit * PAGE_SIZE);
            segment_len  = 1;
          }
          else {  // extends current segment
            segment_len++;
          }
        }
        else {
          if (segment_base) {
            if (option_DEBUG) printf("0x%lx - %d\n", segment_base, segment_len);
            range_map.push_back(range_t{segment_base, segment_len});
            segment_len  = 0;
            segment_base = 0;
          }
        }

        *bptr = *bptr >> 1;
        curr_bit++;
        if (*bptr == 0) {  // optimization: short-circuit for zero
          curr_bit += (7 - i);
          break;
        }
      }

      bptr++;
      remaining--;
    }

    if (segment_base) {
      if (option_DEBUG) printf("0x%lx - %d\n", segment_base, segment_len);
      range_map.push_back(range_t{segment_base, segment_len});
    }
  }

  static void parse_bitmap_individual(void *base, char *data, size_t length, std::vector<addr_t> &dirty_vector)
  {
    assert(length > 0);

    size_t   qwords    = length / sizeof(uint64_t);
    size_t   remaining = length % sizeof(uint64_t);
    unsigned curr_bit  = 0;
    addr_t   base_addr = reinterpret_cast<addr_t>(base);

    assert(base_addr % PAGE_SIZE == 0);

    dirty_vector.clear();

    uint64_t *qptr = (uint64_t *) data;
    while (qwords > 0) {
      for (unsigned i = 0; i < 64; i++) {
        if (*qptr & 0x1) dirty_vector.push_back(base_addr + (curr_bit * PAGE_SIZE));

        *qptr = *qptr >> 1;
        curr_bit++;
        if (*qptr == 0) {  // optimization: short-circuit for zero
          curr_bit += (63 - i);
          break;
        }
      }

      qptr++;
      qwords--;
    }
    unsigned char *bptr = (unsigned char *) qptr;

    while (remaining > 0) {
      for (unsigned i = 0; i < 8; i++) {
        if (*bptr & 0x1) dirty_vector.push_back(base_addr + (curr_bit * PAGE_SIZE));

        *bptr = *bptr >> 1;
        curr_bit++;
        if (*bptr == 0) {  // optimization: short-circuit for zero
          curr_bit += (7 - i);
          break;
        }
      }

      bptr++;
      remaining--;
    }
  }

  static void print_binary(char *data, size_t length)
  {
    for (unsigned i = 0; i < length; i++) {
      char byte = data[i];
      for (unsigned j = 0; j < 8; j++) {
        if (byte & 0x1)
          printf("1");
        else
          printf("0");
        byte = byte >> 1;
      }
      printf("-");
    }
    printf("\n");
  }

  static size_t bitmap_size(size_t data_len)
  {
    size_t size = data_len / 8;
    if (data_len % 8) size++;
    return size;
  }
};

}  // Core namespace

#endif  // __PAGE_BITMAP_H__
