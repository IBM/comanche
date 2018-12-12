#include <sys/sysmacros.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <iostream>
#include <fstream>
#include <fcntl.h>
#include <common/utils.h>

#include "nd_utils.h"

#ifndef MAP_SYNC
#define MAP_SYNC 0x80000
#endif

#ifndef MAP_SHARED_VALIDATE
#define MAP_SHARED_VALIDATE 0x03
#endif

#define MAP_HUGE_2MB    (21 << MAP_HUGE_SHIFT)
#define MAP_HUGE_1GB    (30 << MAP_HUGE_SHIFT)


static inline std::string& rtrim(std::string& s, const char* t = " \t\n\r\f\v")
{
  s.erase(s.find_last_not_of(t) + 1);
  return s;
}

using namespace libndctl;

namespace nupm
{

ND_control::ND_control() : _n_sockets(numa_num_configured_nodes())
{
  /* initialize context */
  if(ndctl_new(&_ctx) != 0)
		throw ND_control_exception("ndctl_new failed unexpectedly");

  /* get hold of NFIT bus, there should only be one */
  struct ndctl_bus * bus;
  ndctl_bus_foreach(_ctx, bus) {
    if(strcmp(ndctl_bus_get_provider(bus),"ACPI.NFIT") != 0 &&
       strcmp(ndctl_bus_get_provider(bus),"e820") != 0)
      throw ND_control_exception("unexpected bus (%s)", ndctl_bus_get_provider(bus));
    else _bus = bus;
    if(option_DEBUG)
      PLOG("ND: provider (%s)", ndctl_bus_get_provider(bus));
  }

  if(_bus == nullptr) {
    _pmem_present = false;
    return;
  }

  /* iterate DIMMs */
  struct ndctl_dimm *dimm;
  ndctl_dimm_foreach(_bus, dimm) {
    unsigned int handle = ndctl_dimm_get_handle(dimm);
    if(option_DEBUG)
      PLOG("dimm: 0x%04x enabled:%d proc-socket:%d imc:%d channel:%d",
           handle,
           ndctl_dimm_is_enabled(dimm),
           ndctl_dimm_handle_get_socket(dimm),
           ndctl_dimm_handle_get_imc(dimm),
           ndctl_dimm_handle_get_channel(dimm)
           );
  }

  /* iterate regions */
  struct ndctl_region *region;
  ndctl_region_foreach(_bus, region) {
    if(strcmp(ndctl_region_get_type_name(region),"pmem") == 0) {

      if(option_DEBUG)
        PLOG("region:%d (%p) type:%s interleaves:%d numa-node:%d dev:%s size:%llu",
             (void*) ndctl_region_get_resource(region),
             ndctl_region_get_id(region),
             ndctl_region_get_type_name(region),
             ndctl_region_get_interleave_ways(region),
             ndctl_region_get_numa_node(region),
             ndctl_region_get_devname(region),
             ndctl_region_get_size(region)
             );

      if(option_DEBUG) {
        struct ndctl_dimm *dimm;
        ndctl_dimm_foreach_in_region(region, dimm) {
          PLOG("\tdimm in region: dev:%s dimm: 0x%04x proc-socket:%d",         
               ndctl_region_get_devname(region),
               ndctl_dimm_get_handle(dimm),
               ndctl_dimm_handle_get_socket(dimm)           
               );
        }
      }

      struct ndctl_interleave_set * is = ndctl_region_get_interleave_set(region);
      if(option_DEBUG && is != nullptr)
	PLOG("interleave_set: active=%d cookie=%llx",
	     ndctl_interleave_set_is_active(is),
	     ndctl_interleave_set_get_cookie(is));
      

      struct ndctl_namespace * ndns;
      ndctl_namespace_foreach(region, ndns) {

        if(option_DEBUG)
          PLOG("namespace: dev-name:%s type:%s block-dev:%s alt:%s",
               ndctl_namespace_get_devname(ndns),
               ndctl_namespace_get_type_name(ndns),
               ndctl_namespace_get_block_device(ndns),
               ndctl_namespace_get_alt_name(ndns)
               );
        

        struct ndctl_dimm * dimm;
        ndctl_dimm_foreach_in_region(region, dimm) {
          /* DIMMs are paired */
          if(option_DEBUG)
            PLOG("%s->%d",
                 ndctl_namespace_get_devname(ndns),
                 ndctl_dimm_handle_get_socket(dimm));
          
          _ns_to_socket[ndctl_namespace_get_devname(ndns)] =
            ndctl_dimm_handle_get_socket(dimm);
        }
      }
    }
  }

}

ND_control::~ND_control() noexcept(false)
{
  if(_pmem_present) {
    for(auto& m: _mappings)
      for(auto& n: m.second)
        if(pmem_unmap(n.first, n.second))
          throw ND_control_exception("pmem_unmap failed unexpectedly");
  }
}

std::vector<std::pair<void*,size_t>> ND_control::get_regions(int numa_zone)
{
  return _mappings[numa_zone];
}

void ND_control::init_devdax()
{
  if(!_pmem_present) return;
  
  /* find out which dax devices belong to which NUMA zones */
  for(auto& p: fs::recursive_directory_iterator("/sys/bus/nd/devices"))
    {
      auto path = p.path().string();
      auto dev_type_path = path + "/devtype";
      std::ifstream path_strm(dev_type_path);

      /* basically, find the dax devices, see which are the nd pmem
         type and then look up the namesapce to /dev/dax id mapping */
      if(dev_type_path.substr(0, 23) != "/sys/bus/nd/devices/dax")
        continue;

      if(!path_strm.is_open())
        throw ND_control_exception("unable to open %s", dev_type_path.c_str());
      std::string dev_type((std::istreambuf_iterator<char>(path_strm)),
                           std::istreambuf_iterator<char>());
      dev_type = rtrim(dev_type);
      if(dev_type == "nd_dax") { /* narrow to ND-DIMM device DAX */
        auto namesp_path =  path + "/namespace";

        std::ifstream path_strm(namesp_path);
        
        if(!path_strm.is_open())
          throw ND_control_exception("unable to open %s", dev_type_path.c_str());
        std::string ns((std::istreambuf_iterator<char>(path_strm)),
                       std::istreambuf_iterator<char>());

        std::string dax_id = path.substr(path.find_last_of("/\\") + 1);
        
        ns = rtrim(ns);
               
        if(!ns.empty()) {
          PLOG("%s daxid=%s dev_type=%s ns=%s", path.c_str(),
               dax_id.c_str(), dev_type.c_str(), ns.c_str());
          /* add to map */
          _ns_to_dax[ns] = dax_id;
        }
      }
    }
}

void ND_control::map_regions(unsigned long base_addr)
{
  if(!_pmem_present) return;

  init_devdax();

  unsigned long vaddr = base_addr > 0 ? base_addr : ZONE_BASE;
  // unsigned long vaddr[MAX_NUMA_ZONES];
  // for(unsigned i=0;i<MAX_NUMA_ZONES;i++)
  //   vaddr[i] = ZONE_BASE + (ZONE_DELTA * i);
  
  PLOG("map_regions: numa-zones=%u", _n_sockets);
  for(unsigned numa_zone = 0; numa_zone < _n_sockets; numa_zone++) {
    for(auto& i : _ns_to_socket) {
      if(i.second == (int) numa_zone) {
        if(_ns_to_dax.find(i.first) != _ns_to_dax.end()) {

          /* map region into virtual memory */
          PLOG("Mapping [%s] is in zone [%d]", _ns_to_dax[i.first].c_str(), numa_zone);
          std::string path = "/dev/" + _ns_to_dax[i.first];

          int fd = open(path.c_str(), O_RDWR, 0666);
          if(fd < 0)
            throw ND_control_exception("unable to open (%s)", path.c_str());

          /* get size of the DAX device */
          struct stat statbuf;
          int rc;
          rc = fstat(fd, &statbuf);
          if(rc == -1) throw ND_control_exception("fstat call failed");
          char spath[PATH_MAX];
          snprintf(spath, PATH_MAX, "/sys/dev/char/%u:%u/size", major(statbuf.st_rdev), minor(statbuf.st_rdev));
          size_t size = 0;
          {
            std::ifstream sizestr(spath);
            sizestr >> size;
          }
          PLOG("size: %lu", size);
          assert(size % 2*1024*1024 == 0);

          /* do mmap */
          void  * pmem_addr = mmap((void*) vaddr,
                                   size,
                                   PROT_READ | PROT_WRITE,
                                   MAP_SHARED | MAP_FIXED | MAP_SYNC, // | MAP_HUGETLB,
                                   fd,
                                   0);
          
          if(pmem_addr == MAP_FAILED) {
            perror("");
            throw ND_control_exception("mmap failed on PM");
          }

#if 0
          PLOG("Touching all memory...");
          { // parallel memset
            unsigned page_size = MB(2);
            char * region = (char *) pmem_addr;
            size_t pages = (size / page_size) + 1;
            assert(size % page_size == 0);
#pragma omp parallel
            {
#pragma omp for
              for(size_t p = 0; p < pages ; p++) {
                memset(&region[p*page_size], 0, page_size);
              }
            }
          }
          PLOG("Done");
#endif
          
          close(fd);
	  vaddr += size;
          
          /* save mapping */
          _mappings[numa_zone].push_back(std::make_pair(pmem_addr, size));
        }
      }
    }
  }
}


} // namespace nupm
