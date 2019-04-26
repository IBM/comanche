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
#ifndef __COMPONENT_TYPES_H__
#define __COMPONENT_TYPES_H__

#include <stdint.h>

/* general types */
typedef uint64_t lba_t;
typedef int64_t  index_t;

enum {
  FLAGS_CREATE      = 0x1,
  FLAGS_FORMAT      = 0x2, /*< force region manager to format block device */
  FLAGS_ITERATE_ALL = 0x4,
  FLAGS_READONLY    = 0x8,
};

#endif
