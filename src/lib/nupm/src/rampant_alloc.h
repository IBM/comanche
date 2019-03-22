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
#ifndef __NUPM_RAMPANT_ALLOC_H__
#define __NUPM_RAMPANT_ALLOC_H__

#include <memory>

namespace nupm
{

/* wrapping rpmalloc and RCA-AVL */

class Rampant_allocator
{
public:
  Rampant_allocator();
  virtual ~Rampant_allocator();
  
private:
  class Rpmalloc;
  
  std::unique_ptr<Rpmalloc> _allocator;
};


} // namespace nupm

#endif // __NUPM_TS_ALLOC_H__
