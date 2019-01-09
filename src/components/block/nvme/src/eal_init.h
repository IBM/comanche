/*
   Copyright [2017] [IBM Corporation]

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

#ifndef __EAL_INIT_H__
#define __EAL_INIT_H__

#include <rte_config.h>
#include <rte_malloc.h>
#include <rte_mempool.h>
#include <string>

#if !defined(__cplusplus)
#error "eal_init.h is C++ only"
#endif

namespace DPDK
{
void eal_init(size_t memory_limit_MB);

}

#endif
