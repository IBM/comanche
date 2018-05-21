/*
   Copyright [2018] [IBM Corporation]

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

#ifndef _ALLOCATOR_ALIGNED_H_
#define _ALLOCATOR_ALIGNED_H_

#include "bad_aligned_alloc.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wvariadic-macros"
#include <common/utils.h> /* round_up */
#pragma GCC diagnostic pop

#include <unistd.h> /* sysconf */

#include <cstddef> /* size_t */
#include <memory> /* allocator */

#include <stdlib.h> /* aligned_alloc */

template <typename T>
  class aligned_allocator
    : public std::allocator<T>
  {
  public:
    using base = std::allocator<T>;
    typename base::pointer allocate(std::size_t sz_)
    {
      auto align = ::sysconf(_SC_PAGESIZE);
      if ( align <= 0 )
      {
        throw bad_aligned_alloc{sz_, align};
      }
      auto ualign = static_cast<unsigned long>(align);
      auto b = ::aligned_alloc(round_up(sz_, ualign), ualign);
      if ( b == nullptr )
      {
        throw bad_aligned_alloc{sz_, align};
      }
      return b;
    }
  };

#endif
