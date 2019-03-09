#ifndef _GET_CPU_MASK_FROM_STRING_
#define _GET_CPU_MASK_FROM_STRING_

#include "get_vector_from_string.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#include <common/logging.h> /* PERR */
#pragma GCC diagnostic pop
#include <common/cpu.h> /* cpu_mask_t */

#include <sstream>
#include <string>
#include <thread>

namespace
{
  void _cpu_mask_add_core_wrapper(cpu_mask_t &mask, unsigned core_first, unsigned core_last, unsigned mac_cores)
  {
    if ( core_last < core_first )
    {
      std::ostringstream e;
      e << "invalid core range specified: start (" << core_first << ") > end (" << core_last << ")";
      PERR("%s.", e.str().c_str());
      throw std::runtime_error(e.str());
    }
    else if ( mac_cores < core_last )  // mac_cores is zero indexed
    {
      std::ostringstream e;
      e << "specified core end (-" << core_last << "exceeds physical core count. Valid range is [0.." << mac_cores << ")";
      PERR("%s", e.str().c_str());
      throw std::runtime_error(e.str());
    }

    try
    {
      for (unsigned core = core_first; core != core_last; ++core)
      {
        mask.add_core(core);
      }
    }
    catch ( const Exception &e )
    {
      PERR("failed while adding core to mask: %s.", e.cause());
      throw;
    }
    catch(...)
    {
      PERR("%s", "failed while adding core to mask.");
      throw;
    }
  }

  cpu_mask_t get_cpu_mask_from_string(std::string core_string)
  {
    auto cores = get_vector_from_string<int>(core_string);
    cpu_mask_t mask;
    int hardware_total_cores = std::thread::hardware_concurrency();

    for ( auto c : cores )
    {
      _cpu_mask_add_core_wrapper(mask, c, c+1, hardware_total_cores);
    }

    return mask;
  }
}

#endif
