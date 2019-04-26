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
#ifndef __MEMKIND_SLAB_H__
#define __MEMKIND_SLAB_H__

#include <asm/mman.h>
#include <common/exceptions.h>
#include <common/logging.h>
#include <common/spinlocks.h>
#include <common/str_utils.h>
#include <common/utils.h>
#include <fcntl.h>
#include <memkind.h>
#include <numa.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <map>
#include <mutex>
#include <string>

namespace nupm
{
/**
 * Memkind based allocator without reconstitution
 *
 */
class Memkind_allocator {
 private:
  static constexpr unsigned MAX_NUMA_ZONES = 2;
  static constexpr unsigned option_DEBUG   = 0;

 public:
  Memkind_allocator(const std::string &pmem_file, size_t size = GB(1))
  {
    if (pmem_file.substr(0, 8) == "/dev/dax") /* if devdax use whole device */
      size = 0;

    int err = memkind_create_pmem(pmem_file.c_str(), size, &_kind);
    if (err) {
      perror("memkind_create_pmem()");
      throw Constructor_exception("memkind_create_pmem failed");
    }
  }

  void add_new_region(addr_t region, size_t length)
  {
    throw API_exception(
        "add_new_region is not implemented in memkind allocator");
  }

  void *alloc_at(addr_t base, size_t size)
  {
    throw API_exception("allocate_at is not implemented in memkind allocator");
    return nullptr;
  }

  void *malloc(size_t size)
  {
    if (option_DEBUG > 1) PLOG("memkind: malloc(size=%lu)", size);

    return memkind_malloc(_kind, size);
  }

  int posix_memalign(void **memptr, size_t size, size_t alignment)
  {
    assert(size > 0);

    int rc = memkind_posix_memalign(_kind, memptr, alignment, size);

    if (option_DEBUG > 1)
      PLOG("memkind: memalign(memptr=%p, alignment=%lu, size=%lu)", *memptr,
           alignment, size);

    return rc;
  }

  void free(void *p)
  {
    if (option_DEBUG > 1) PLOG("memkind: free(p=%p)", p);

    return memkind_free(_kind, p);
  }

 private:
  memkind_t _kind;
};

}  // namespace nupm

#endif
