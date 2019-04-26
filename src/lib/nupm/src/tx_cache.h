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
#ifndef __TX_CACHE_H__
#define __TX_CACHE_H__

#include <common/types.h>
#include <sys/mman.h>

#define MAP_HUGE_2MB (21 << MAP_HUGE_SHIFT)
#define MAP_HUGE_1GB (30 << MAP_HUGE_SHIFT)

namespace nupm
{
void *allocate_virtual_pages(size_t n_pages, size_t page_size, addr_t hint = 0);
int   free_virtual_pages(void *p);
}  // namespace nupm
#endif  // __TX_CACHE_H__
