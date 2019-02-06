#include <mutex>
#include <fstream>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/sysmacros.h>
#include <fcntl.h>
#include <unistd.h>
#include <common/utils.h>
// #include <libpmemobj.h>
// #include <libpmempool.h>
// #include <libpmemobj/base.h>
#include <common/exceptions.h>
#include "dax_data.h"
#include "dax_map.h"


//#define REGION_NAME "dawn-dax-pool"
#define DEBUG_PREFIX "Devdax_manager: "

#define MAP_HUGE_2MB    (21 << MAP_HUGE_SHIFT)
#define MAP_HUGE_1GB    (30 << MAP_HUGE_SHIFT)

#ifndef MAP_SYNC
#define MAP_SYNC 0x80000
#endif

#ifndef MAP_SHARED_VALIDATE
#define MAP_SHARED_VALIDATE 0x03
#endif


namespace nupm
{

/* static members */
ND_control Devdax_manager::_nd;
std::map<std::string, iovec> Devdax_manager::_mapped_regions;
std::map<std::string, DM_region_header *> Devdax_manager::_region_hdrs;
std::mutex Devdax_manager::_reentrant_lock;

  
struct device_t
{
  const char * path;
  addr_t       addr;
  int          numa_node;
};


/* currently one dax device per numa node */
static constexpr device_t dax_config[] = {{"/dev/dax0.3", 0x9000000000, 0},
                                          {"/dev/dax1.3", 0xa000000000, 1},
                                          {"", 0, 0}
};

static const char * lookup_dax_device(int numa_node) {
  unsigned idx=0;
  while(dax_config[idx].addr > 0) {
    if(dax_config[idx].numa_node == numa_node) return dax_config[idx].path;
  }
  return nullptr;
}

Devdax_manager::Devdax_manager(bool force_reset) {

  unsigned idx = 0;
  while(dax_config[idx].addr) {
    if(_debug_level > 0)
      PLOG(DEBUG_PREFIX "region (%s,%lx)", dax_config[idx].path, dax_config[idx].addr);

    void * p = map_region(dax_config[idx].path, dax_config[idx].addr);
    recover_metadata(dax_config[idx].path,
                     p,
                     _mapped_regions[dax_config[idx].path].iov_len,
                     force_reset);
    
    idx++;
  }
}

Devdax_manager::~Devdax_manager() {
  for(auto& i: _mapped_regions) {
    munmap(i.second.iov_base, i.second.iov_len);
  }
}

void Devdax_manager::debug_dump(int numa_node) {
  guard_t g(_reentrant_lock);
  _region_hdrs[lookup_dax_device(numa_node)]->debug_dump();
}
  

void * Devdax_manager::open_region(uint64_t uuid, int numa_node, size_t * out_length)
{
  guard_t g(_reentrant_lock);
  const char * device = lookup_dax_device(numa_node);
  DM_region_header * hdr = _region_hdrs[device];
  if(hdr == nullptr)
    throw General_exception("no region header for device (%s)", device);

  size_t region_length = 0;
  return hdr->get_region(uuid, &region_length);
}
  
void * Devdax_manager::create_region(uint64_t uuid, int numa_node, size_t size)
{
  guard_t g(_reentrant_lock);
  const char * device = lookup_dax_device(numa_node);
  DM_region_header * hdr = _region_hdrs[device];
  if(hdr == nullptr)
    throw General_exception("no region header for device (%s)", device);

  return hdr->allocate_region(uuid, (size/GB(1)+1));
}

void Devdax_manager::erase_region(uint64_t uuid, int numa_node)
{
  guard_t g(_reentrant_lock);
  const char * device = lookup_dax_device(numa_node);
  DM_region_header * hdr = _region_hdrs[device];
  if(hdr == nullptr)
    throw General_exception("no region header for device (%s)", device);

  hdr->erase_region(uuid);
}

size_t Devdax_manager::get_max_available(int numa_node)
{
  guard_t g(_reentrant_lock);
  const char * device = lookup_dax_device(numa_node);
  DM_region_header * hdr = _region_hdrs[device];
  if(hdr == nullptr)
    throw General_exception("no region header for device (%s)", device);

  return hdr->get_max_available();
}
  
void Devdax_manager::recover_metadata(const char * device_path,
                                      void * p,
                                      size_t p_len,
                                      bool force_rebuild)
{
  assert(p);
  DM_region_header * rh = (DM_region_header *) p;

  bool rebuild = force_rebuild;
  if(!rh->check_magic()) rebuild = true;
  
  if(rebuild) {
    PMAJOR("need to rebuild!");
    rh = new (p) DM_region_header(p_len);
  }
  else {
    PMAJOR("no need to rebuild!");
    rh->check_undo_logs();
  }
  
  _region_hdrs[device_path] = rh;
}

void * Devdax_manager::get_devdax_region(const char * device_path, size_t * out_length)
{
  auto r = _mapped_regions[device_path];
  if(out_length)
    *out_length = r.iov_len;
  return r.iov_base;
}
  
void * Devdax_manager::map_region(const char * path, addr_t base_addr)
{
  assert(base_addr);
  assert(check_aligned(base_addr, GB(1)));
         
  /* open device */
  int fd = open(path, O_RDWR, 0666);

  if(fd == -1)
    throw General_exception("inaccessible devdax path (%s)", path);

  if(_debug_level > 0)
    PLOG(DEBUG_PREFIX "region (%s) opened ok", path);

  /* get length of device */
  size_t size = 0;
  {
    struct stat statbuf;
    int rc = fstat(fd, &statbuf);
    if(rc == -1) throw ND_control_exception("fstat call failed");
    char spath[PATH_MAX];
    snprintf(spath, PATH_MAX, "/sys/dev/char/%u:%u/size", major(statbuf.st_rdev), minor(statbuf.st_rdev));
    std::ifstream sizestr(spath);
    sizestr >> size;
  }
  PLOG(DEBUG_PREFIX "%s size=%lu", path, size);
  
  /* mmap it in */
  void * p = mmap((void*) base_addr,
                  size, /* length = 0 means whole device */
                  PROT_READ | PROT_WRITE,
                  MAP_SHARED_VALIDATE | MAP_FIXED | MAP_SYNC, // TODO check | MAP_HUGETLB | MAP_HUGE_2MB,
                  fd,
                  0 /* offset */);

  if(p != (void*) base_addr) {
    perror("");
    throw General_exception("mmap failed on devdax");
  }
  
  _mapped_regions[std::string(path)] = {p,size};  
  
  close(fd);
  
  return p;
}
  
}

