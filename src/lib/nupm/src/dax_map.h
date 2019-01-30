#ifndef __NUPM_DAX_MAP_H__
#define __NUPM_DAX_MAP_H__

#include <string>
#include "nd_utils.h"

namespace nupm
{

class DM_region_header;
  
struct device_t
{
  const char * path;
  addr_t       addr;
  int          numa_node;
};

class Devdax_manager
{
private:
  static constexpr unsigned _debug_level = 3;
  
public:
  Devdax_manager(bool force_reset = false);
  ~Devdax_manager();

  void * open_region(uint64_t uuid, int numa_node, size_t * out_length);
  void * create_region(uint64_t uuid, int numa_node, size_t size);
  void erase_region(uint64_t uuid, int numa_node);
  void debug_dump(int numa_node);
  
private:
  void * get_devdax_region(const char * device_path, size_t * out_length);
  void * map_region(const char * path, addr_t base_addr);
  void recover_metadata(const char * device_path, void * p, size_t p_len, bool force_rebuild = false);
  
private:
  ND_control                                _nd;
  std::map<std::string, iovec>              _mapped_regions;
  std::map<std::string, DM_region_header *> _region_hdrs;
};


}

#endif
