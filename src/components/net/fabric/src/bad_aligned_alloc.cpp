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

#include "bad_aligned_alloc.h"

bad_aligned_alloc::bad_aligned_alloc(std::size_t sz_, long align_)
  : std::bad_alloc()
  , _what{"Could not aligned_alloc " + std::to_string(sz_) + "," + std::to_string(align_)}
{}

const char *bad_aligned_alloc::what() const noexcept
{
  return _what.c_str();
}
