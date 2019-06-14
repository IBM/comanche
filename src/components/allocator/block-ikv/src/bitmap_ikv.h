/*
   Copyright [2017-2019] [IBM Corporation]
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

#ifndef BITMAP_IKV_H_
#define BITMAP_IKV_H_

#include <component/base.h>
/* Note: these uuid decls are so we don't have to have access to the component
 * source code */

#ifndef MB
#define MB(x) (x >> 20)
#endif

#define BITS_PER_LONG (64)
#define BITS_TO_LONGS(nr) (((nr) + (BITS_PER_LONG - 1)) / (BITS_PER_LONG))

namespace block_alloc_ikv
{
using word_t = uint64_t;  // bitmap operation is done in word

class bitmap_ikv {
 public:
  /**
   * construct a bitmap in a IKVstore pool
   * @param store
   * @param pool
   * @param nbits number of
   */
  bitmap_ikv(const std::string& id, unsigned nbits) : _id(id) {}

  ~bitmap_ikv();

  status_t zero();
  /*
   * find a contiguous aligned mem region
   *
   * @param pop persist object pool
   * @param bitmap the bitmap to operate on
   * @param order, the order(2^order free blocks) to set from 0 to 1
   */
  int find_free_region(unsigned order);

  /*
   * free a region from bitmap
   *
   * @param pos the starting bit to reset
   * @param order, the order(2^order bits) to reset
   */
  int release_region(unsigned int pos, unsigned order);

 private:
  std::string             _id; /** identifier for this allocator/block device*/
  size_t                  _capacity; /** how many bit in this bitmap*/
  word_t*                 _bitdata;
  static constexpr size_t BITMAP_CHUNCK_SIZE = MB(4); /** intial bitmap*/
};

int bitmap_ikv::find_free_region(unsigned order) { return 0; }

int bitmap_ikv::release_region(unsigned int pos, unsigned order) { return 0; }

}  // namespace block_alloc_ikv

#endif
