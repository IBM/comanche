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
#include "registered_memory.h"

registered_memory::registered_memory(Component::IFabric_connection &cnxn_, std::size_t size_, std::uint64_t remote_key_)
  : _memory(size_)
  , _registration(cnxn_, &*_memory.begin(), _memory.size(), remote_key_, 0U)
{}
