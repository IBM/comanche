#include <algorithm> /* copy, move */
#include <stdexcept> /* out_of_range */
#include <string>
#include <vector>

template <typename Table>
	impl::atomic_controller<Table>::atomic_controller(
			persist_atomic<allocator_type> &persist_
			, table_t &map_
		)
			: allocator_type(map_.get_allocator())
			, _persist(&persist_)
			, _map(&map_)
		{
			redo();
		}

template <typename Table>
	auto impl::atomic_controller<Table>::redo() -> void
	{
		if ( _persist->mod_size != 0 )
		{
			try
			{
				char *src = _persist->mod_mapped.data();
				char *dst = _map->at(_persist->mod_key).data();
				auto mod_ctl = &*(_persist->mod_ctl);
				for ( auto i = mod_ctl; i != &mod_ctl[_persist->mod_size]; ++i )
				{
					std::size_t o_s = i->offset_src;
					auto src_first = &src[o_s];
					std::size_t sz = i->size;
					auto src_last = src_first + sz;
					std::size_t o_d = i->offset_dst;
					auto dst_first = &dst[o_d];
					/* NOTE: could be replaced with a pmem persistent memcpy */
					persist_range(dst_first, std::copy(src_first, src_last, dst_first), "atomic ctl");
				}
			}
			catch ( std::out_of_range & )
			{
				/* no such key */
			}
		}
		_persist->mod_size = 0;
		persist_range(&_persist->mod_size, &_persist->mod_size + 1, "atomic size");
	}

template <typename Table>
	void impl::atomic_controller<Table>::persist_range(const void *first_, const void *last_, const char *what_)
	{
		persist_switch_t::persist(*this, first_, last_, what_);
	}

template <typename Table>
	auto impl::atomic_controller<Table>::enter(
		PMEMobjpool *pop
		, persist_fixed_string<char> &key
		, uint64_t type_num_data
		, std::vector<Component::IKVStore::Operation *>::const_iterator first
		, std::vector<Component::IKVStore::Operation *>::const_iterator last
	) -> typename Component::status_t
	{
		std::vector<char> src;
		std::vector<mod_control> mods;
		for ( ; first != last ; ++first )
		{
			switch ( (*first)->type() )
			{
			case Component::IKVStore::Op_type::WRITE:
				{
					const Component::IKVStore::Operation_write &wr = *static_cast<Component::IKVStore::Operation_write *>(*first);
					auto src_offset = src.size();
					auto dst_offset = wr.offset();
					auto size = wr.size();
					auto op_src = static_cast<const char *>(wr.data());
					/* No source for data yet, use Xs */
					std::copy(op_src, op_src + size, std::back_inserter(src));
					mods.emplace_back(src_offset, dst_offset, size);
				}
				break;
			default:
				return E_NOT_SUPPORTED;
			};
		}
		_persist->mod_key = key;
		_persist->mod_mapped = persist_fixed_string<char>(src.begin(), src.end(), pop, type_num_data, "atomic data"); /* PERSISTED? */
		using void_allocator_t = typename allocator_type::template rebind<void>::other;
		_persist->mod_ctl =
			allocator_type(*this).address(
				*new (&*allocator_type(*this).allocate(mods.size(), typename void_allocator_t::const_pointer(), "mod_ctl"))
				mod_control[mods.size()]
		);

		std::copy(mods.begin(), mods.end(), &*_persist->mod_ctl);
		/* 8-byte atomic write */
		_persist->mod_size = mods.size();
		persist_range(&*_persist->mod_ctl, &*_persist->mod_ctl + mods.size(), "mod control");
		redo();
		return S_OK;
	}
