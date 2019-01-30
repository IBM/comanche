#ifndef __NUPM_DAX_MAP_H__
#define __NUPM_DAX_MAP_H__

#include <string>
#include "tx_cache.h"

namespace nupm
{

/** 
 * Map a dev dax device to a region of memory
 * 
 * @param device_path Path of device (e.g., /dev/dax0.0)
 * @param hint Address at which to map to
 * 
 */
void * allocate_dax_subregion(const std::string& device_path, void * hint = nullptr);

}

#endif
