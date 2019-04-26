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


#ifndef COMANCHE_HSTORE_NUPM_REGION_H
#define COMANCHE_HSTORE_NUPM_REGION_H

/* requires persist_data_t definition */
#include "persist_data.h"

template <typename PersistData, typename Heap>
  class region
  {
    static constexpr std::uint64_t magic_value = 0xc74892d72eed493a;
    std::uint64_t magic;
  public:
    using heap_type = Heap;
    using persist_data_type = PersistData;

    persist_data_type persist_data;
    heap_type heap;

    void initialize() { magic = magic_value; }
    bool is_initialized() const noexcept { return magic == magic_value; }
    /* region used by heap_cc follows */
  };

#endif
