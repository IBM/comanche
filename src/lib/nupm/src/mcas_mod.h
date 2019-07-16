/*
   Copyright [2019] [IBM Corporation]
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

#ifndef __NUPM_MCAS_MOD_H__
#define __NUPM_MCAS_MOD_H__

#include <common/types.h>
#include <stdint.h>
#include <unistd.h>

namespace nupm
{
using Memory_token = uint64_t;

/* NOTE: these APIs require the MCAS kernel module to be loaded */

bool check_mcas_kernel_module();
status_t expose_memory(Memory_token token, void * vaddr, size_t vaddr_size);
status_t revoke_memory(Memory_token token);
void *   mmap_exposed_memory(Memory_token token,
                             size_t& size,
                             void* target_addr = nullptr);

}

#endif // __NUPM_MCAS_MOD_H__
