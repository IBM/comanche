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

#include <sys/uio.h>
#include <memory>

class Devdax_manager;

template <typename PersistData, typename Heap>
  class region
  {
    static constexpr std::uint64_t magic_value = 0xc74892d72eed493a;
  public:
    using heap_type = Heap;
    using persist_data_type = PersistData;

  private:
    std::uint64_t magic;
	std::uint64_t _uuid;
    heap_type _heap;
    persist_data_type _persist_data;

  public:

    region(
      std::uint64_t uuid_
      , std::size_t size_
      , std::size_t expected_obj_count
      , unsigned numa_node_
    )
      : magic(0)
      , _uuid(uuid_)
      , _heap(this+1, size_ - sizeof(*this), numa_node_)
      , _persist_data(expected_obj_count, typename PersistData::allocator_type(heap()))
    {
      magic = magic_value;
      persister_nupm::persist(this, sizeof *this);
    }

	heap_rc heap() { return heap_rc(&_heap); }
	persist_data_type &persist_data() { return _persist_data; }
    bool is_initialized() const noexcept { return magic == magic_value; }
	void animate(const std::unique_ptr<Devdax_manager> & devdax_manager) { _heap.animate(devdax_manager); }
	void quiesce() { _heap.quiesce(); }
    std::vector<::iovec> get_regions()
    {
      std::vector<::iovec> regions;
      regions.push_back(_heap.region());
      return regions;
    }
	auto grow(
		const std::unique_ptr<Devdax_manager> & devdax_manager_
		, std::size_t increment_
	) -> std::size_t
	{
		return _heap.grow(devdax_manager_, _uuid, increment_);
	}
    /* region used by heap_cc follows */
  };

#endif
