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


/*
 * Authors:
 *
 * Daniel G. Waddington (daniel.waddington@ibm.com)
 *
 */

#ifndef __NUPM_DAX_MAP_H__
#define __NUPM_DAX_MAP_H__

#include <mutex>
#include <string>
#include <tuple>
#include "nd_utils.h"

namespace nupm
{
class DM_region_header;

/**
 * Lowest level persisent manager for devdax devices. See dax_map.cc for static
 * configuration.
 *
 */
class Devdax_manager {
 private:
  static constexpr unsigned _debug_level = 3;

 public:

  typedef struct {
    std::string path;
    addr_t addr;
    unsigned region_id;
  } config_t;
  
  /** 
   * Constructor e.g.  
     nupm::Devdax_manager ddm({{"/dev/dax0.3", 0x9000000000, 0},
                               {"/dev/dax1.3", 0xa000000000, 1}},
                                true); 
   * 
   * @param dax_config Vector of dax-path, address, region_id tuples.
   * @param force_reset 
   */
  Devdax_manager(const std::vector<config_t>& dax_config,
                 bool force_reset = false);

  /**
   * Destructor will not unmap memory/nor invalidate pointers?
   *
   */
  ~Devdax_manager();

  /**
   * Open a region of memory
   *
   * @param uuid Unique identifier
   * @param region_id Region identifier (normally 0)
   * @param out_length Out length of region in bytes
   *
   * @return Pointer to mapped memory or nullptr on not found
   */
  void *open_region(uint64_t uuid, unsigned region_id, size_t *out_length);

  /**
   * Create a new region of memory
   *
   * @param uuid Unique identifier
   * @param region_id Region identifier (normally 0)
   * @param size Size of the region requested in bytes
   *
   * @return Pointer to mapped memory
   */
  void *create_region(uint64_t uuid, unsigned region_id, size_t size);

  /**
   * Erase a previously allocated region
   *
   * @param uuid Unique region identifier
   * @param region_id Region identifier (normally 0)
   */
  void erase_region(uint64_t uuid, unsigned region_id);

  /**
   * Get the maximum "hole" size.
   *
   *
   * @return Size in bytes of max hole
   */
  size_t get_max_available(unsigned region_id);

  /**
   * Debugging information
   *
   * @param region_id Region identifier
   */
  void debug_dump(unsigned region_id);

 private:
  void *get_devdax_region(const char *device_path, size_t *out_length);
  void *map_region(const char *path, addr_t base_addr);
  void  recover_metadata(const char *device_path,
                         void *      p,
                         size_t      p_len,
                         bool        force_rebuild = false);
  const char * lookup_dax_device(unsigned region_id);

 private:
  using guard_t = std::lock_guard<std::mutex>;

  const std::vector<config_t>               _dax_configs;
  ND_control                                _nd;
  std::map<std::string, iovec>              _mapped_regions;
  std::map<std::string, DM_region_header *> _region_hdrs;
  std::mutex                                _reentrant_lock;
};
}  // namespace nupm

#endif
