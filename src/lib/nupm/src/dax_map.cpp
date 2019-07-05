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

#include "dax_map.h"
#include "dax_data.h"

#include <common/exceptions.h>
#include <common/utils.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <sys/types.h>
#include <unistd.h>
#include <fstream>
#include <mutex>
#include <set>

//#define REGION_NAME "dawn-dax-pool"
#define DEBUG_PREFIX "Devdax_manager: "

#define MAP_HUGE_2MB (21 << MAP_HUGE_SHIFT)
#define MAP_HUGE_1GB (30 << MAP_HUGE_SHIFT)

#ifndef MAP_SYNC
#define MAP_SYNC 0x80000
#endif

#ifndef MAP_SHARED_VALIDATE
#define MAP_SHARED_VALIDATE 0x03
#endif


namespace dax_map {
static size_t use_dram = 0; // when DCPMM is not available
}
  
__attribute__((constructor))
static void __dax_map_ctr() 
{
  /* env variable USE_DRAM to trigger use of DRAM instead of PM */
  char* p = getenv("USE_DRAM");
  if(p != nullptr) {
    errno = 0;
    auto i = strtol(p,nullptr,10);

    if(errno == 0) {
      dax_map::use_dram = GB(1) * i;
      PLOG("using DRAM to emulate PM (%ld GB)", i);
    }
  }
}




static std::set<std::string> nupm_devdax_manager_mapped;
static std::mutex nupm_devdax_manager_mapped_lock;

static bool register_instance(const std::string& path)
{
  std::lock_guard<std::mutex> g(nupm_devdax_manager_mapped_lock);
  if(nupm_devdax_manager_mapped.find(path) != nupm_devdax_manager_mapped.end())
    return false;
  
  nupm_devdax_manager_mapped.insert(path);
  PLOG("Registered dax mgr instance: %s", path.c_str());
  return true;
}

static void unregister_instance(const std::string& path)
{
  std::lock_guard<std::mutex> g(nupm_devdax_manager_mapped_lock);
  nupm_devdax_manager_mapped.erase(path);
  PLOG("Unregistered dax mgr instance: %s", path.c_str());
}

namespace nupm
{

Devdax_manager::Devdax_manager(const std::vector<config_t>& dax_configs,
                               bool force_reset) : _dax_configs(dax_configs)
{
  unsigned idx = 0;

  /* set up each configuration */
  for(auto& config: dax_configs) {

    if (_debug_level > 0)
      PLOG(DEBUG_PREFIX "region (%s,%lx)",
           config.path.c_str(),
           config.addr);

    auto pathstr = config.path.c_str();

    if(register_instance(config.path) == false) /*< only one instance of this class per dax path */
      throw Constructor_exception("Devdax_manager instance already managing path (%s)", pathstr);

    void *p = map_region(pathstr,config.addr);

    if(dax_map::use_dram > 0) /* for DRAM, we always reset */
      force_reset = true;
    
    recover_metadata(pathstr,
                     p,
                     _mapped_regions[pathstr].iov_len,
                     force_reset);

    idx++;
  }
}

Devdax_manager::~Devdax_manager()
{
  for (auto &i : _mapped_regions) {
    munmap(i.second.iov_base, i.second.iov_len);
    unregister_instance(i.first);
  }
}

const char * Devdax_manager::lookup_dax_device(unsigned region_id)
{
  for(auto& config: _dax_configs) {
    if(config.region_id == region_id) return config.path.c_str();
  }
  throw Logic_exception("lookup_dax_device could not find path for region (%d)",
                        region_id);
  return nullptr;
}


void Devdax_manager::debug_dump(unsigned region_id)
{
  guard_t g(_reentrant_lock);
  _region_hdrs[lookup_dax_device(region_id)]->debug_dump();
}

void *Devdax_manager::open_region(uint64_t uuid,
                                  unsigned region_id,
                                  size_t * out_length)
{
  guard_t           g(_reentrant_lock);
  const char *      device = lookup_dax_device(region_id);
  DM_region_header *hdr    = _region_hdrs[device];
  if (hdr == nullptr)
    throw General_exception("no region header for device (%s)", device);

  return hdr->get_region(uuid, out_length);
}

void *Devdax_manager::create_region(uint64_t uuid, unsigned region_id, const size_t size)
{
  guard_t           g(_reentrant_lock);
  const char *      device = lookup_dax_device(region_id);

  DM_region_header *hdr    = _region_hdrs[device];
  if (hdr == nullptr)
    throw General_exception("no region header for device (%s)", device);

  auto size_in_GB = (size / GB(1)) + 1;
  PLOG("Devdax_manager::create_region rounding up to %lu GB\n", size_in_GB);
  return hdr->allocate_region(uuid, size_in_GB); /* allocates n GiB */
}

void Devdax_manager::erase_region(uint64_t uuid, unsigned region_id)
{
  guard_t           g(_reentrant_lock);
  const char *      device = lookup_dax_device(region_id);
  DM_region_header *hdr    = _region_hdrs[device];
  if (hdr == nullptr)
    throw General_exception("no region header for device (%s)", device);

  hdr->erase_region(uuid);
}

size_t Devdax_manager::get_max_available(unsigned region_id)
{
  guard_t           g(_reentrant_lock);
  const char *      device = lookup_dax_device(region_id);
  DM_region_header *hdr    = _region_hdrs[device];
  if (hdr == nullptr)
    throw General_exception("no region header for device (%s)", device);

  return hdr->get_max_available();
}

void Devdax_manager::recover_metadata(const char *device_path,
                                      void *      p,
                                      size_t      p_len,
                                      bool        force_rebuild)
{
  assert(p);
  DM_region_header *rh = static_cast<DM_region_header *>(p);

  bool rebuild = force_rebuild;
  if (!rh->check_magic()) rebuild = true;

  if (rebuild) {
    PLOG("Devdax_manager: rebuilding.");
    rh = new (p) DM_region_header(p_len);
  }
  else {
    PLOG("Devdax_manager: no rebuild.");
    rh->check_undo_logs();
  }

  _region_hdrs[device_path] = rh;
}

void *Devdax_manager::get_devdax_region(const char *device_path,
                                        size_t *    out_length)
{
  auto r = _mapped_regions[device_path];
  if (out_length) *out_length = r.iov_len;
  return r.iov_base;
}

void *Devdax_manager::map_region(const char *path, addr_t base_addr)
{
  assert(base_addr);
  assert(check_aligned(base_addr, GB(1)));

  /* DRAM emulating PM */
  if(dax_map::use_dram > 0) {
    size_t size = dax_map::use_dram;
    void *p = mmap((void *) base_addr, size, /* length = 0 means whole device */
                   PROT_READ | PROT_WRITE,
                   MAP_ANONYMOUS | MAP_SHARED | MAP_FIXED | MAP_LOCKED,
                   0, /* file */
                   0 /* offset */);

    if (p != (void *) base_addr) {
      perror("");
      throw General_exception("mmap failed on DRAM for emulated devdax");
    }

    if(madvise(p, size, MADV_DONTFORK) != 0)
      throw General_exception("madvise 'don't fork' failed unexpectedly (%p %lu)", base_addr, size);

    _mapped_regions[std::string(path)] = {p, size};
    return p;
  }
  
  /* open device */
  int fd = open(path, O_RDWR, 0666);

  if (fd == -1) throw General_exception("map_region: inaccessible devdax path (%s)", path);

  if (_debug_level > 0) PLOG(DEBUG_PREFIX "region (%s) opened ok", path);

  /* get length of device */
  size_t size = 0;
  {
    struct stat statbuf;
    int         rc = fstat(fd, &statbuf);
    if (rc == -1) throw ND_control_exception("fstat call failed");
    char spath[PATH_MAX];
    snprintf(spath, PATH_MAX, "/sys/dev/char/%u:%u/size",
             major(statbuf.st_rdev), minor(statbuf.st_rdev));
    std::ifstream sizestr(spath);
    sizestr >> size;
  }
  PLOG(DEBUG_PREFIX "%s size=%lu", path, size);

  /* mmap it in */
  void *p;

  p = mmap((void *) base_addr,
	   size, /* length = 0 means whole device */
	   PROT_READ | PROT_WRITE,
	   MAP_SHARED_VALIDATE | MAP_FIXED | MAP_LOCKED | MAP_SYNC | MAP_HUGE_2MB,
	   fd, 0 /* offset */);

  if(p == ((void*) -1)) {
    p = mmap((void *) base_addr,
             size, /* length = 0 means whole device */
             PROT_READ | PROT_WRITE,
             MAP_SHARED_VALIDATE | MAP_FIXED | MAP_LOCKED | MAP_HUGE_2MB,
             fd, 0 /* offset */);
  }
    

  if (p != (void *) base_addr) {
    perror("");
    throw General_exception("mmap failed on devdax (request %p, got %p)", base_addr, p);
  }

  if(madvise(p, size, MADV_DONTFORK) != 0)
    throw General_exception("madvise 'don't fork' failed unexpectedly (%p %lu)",
			    base_addr, size);
  
  _mapped_regions[std::string(path)] = {p, size};

  close(fd);

  return p;
}
}  // namespace nupm
