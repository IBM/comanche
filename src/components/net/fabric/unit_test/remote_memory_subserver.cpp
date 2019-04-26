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
#include "remote_memory_subserver.h"

#include "server_grouped_connection.h"
#include <common/errors.h> /* S_OK */
#include <gtest/gtest.h>

remote_memory_subserver::remote_memory_subserver(
  server_grouped_connection &parent_
  , std::size_t memory_size_
  , std::uint64_t remote_key_index_
  )
  : _parent(parent_)
  , _cnxn(_parent.allocate_group())
  , _rm_out{_parent.cnxn(), memory_size_, remote_key_index_ * 2U}
  , _rm_in{_parent.cnxn(), memory_size_, remote_key_index_ * 2U + 1}
{
}
