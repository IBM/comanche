#ifndef __ND_UTILS_H__
#define __ND_UTILS_H__

#include <string>
#include <map>
#include <vector>
#include <utility>
#include <experimental/filesystem>
#include <iostream>
#include <fstream>
#include <daxctl/libdaxctl.h>
#include <libpmem.h>
#include <numa.h>
#include <common/exceptions.h> // base Exception class

namespace libndctl
{
#include <ndctl/libndctl.h>
}

#define DIMM_HANDLE(n, s, i, c, d)                              \
		(((n & 0xfff) << 16) | ((s & 0xf) << 12) | ((i & 0xf) << 8) \
		 | ((c & 0xf) << 4) | (d & 0xf))

namespace fs = std::experimental::filesystem;

namespace nupm
{

class ND_control_exception : public Exception
{
public:
  ND_control_exception() : Exception("ND_control error"), _err_code(E_FAIL)
  {
  }

  __attribute__((__format__(__printf__, 2, 0))) ND_control_exception(const char* fmt, ...) : Exception()
  {
    va_list args;
    va_start(args, fmt);
    char msg[255] = {0};
    vsnprintf(msg, 254, fmt, args);
    set_cause(msg);
  }

  status_t error_code()
  {
    return _err_code;
  }

private:
  status_t _err_code;
};

/** 
 * NVDIMM control class
 * 
 */
class ND_control
{
private:
  static constexpr unsigned MAX_NUMA_ZONES = 4;
  static constexpr bool option_DEBUG = false;
  static constexpr unsigned long ZONE_BASE   = 0x90000000000ULL; // could use ASLR 
  static constexpr unsigned long ZONE_DELTA  = 0x50000000000ULL;
public:

  /** 
   * Constructor
   * 
   */
  ND_control();

  /** 
   * Destructor
   * 
   */
  virtual ~ND_control() noexcept(false);

  /** 
   * Get the mapped regions for a specific NUMA zone
   * 
   * @param numa_zone 
   * 
   * @return Copy of a vector of region base, size pairs
   */
  std::vector<std::pair<void*,size_t>> get_regions(int numa_zone) __attribute__((deprecated));

  /** 
   * Map regions with device DAX
   * 
   * @param Hint for base address
   */
  void map_regions(unsigned long base_addr = 0) __attribute__((deprecated));
  
private:

  void init_devdax();

  
protected:
  bool                                                _pmem_present = true;
  unsigned                                            _n_sockets;
  struct libndctl::ndctl_ctx *                        _ctx;
  struct libndctl::ndctl_bus *                        _bus;
  std::map<std::string, int>                          _ns_to_socket;
  std::map<std::string, std::string>                  _ns_to_dax;
  std::map<std::string, std::string>                  _dax_to_ns;

  
  std::map<int, std::vector<std::pair<void*,size_t>>> _mappings;
  
};


} // namespace nupm

#endif
