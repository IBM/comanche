/*
 * (C) Copyright IBM Corporation 2018, 2019. All rights reserved.
 * US Government Users Restricted Rights - Use, duplication or disclosure restricted by GSA ADP Schedule Contract with IBM Corp.
 */

#include <algorithm> /* copy, move */
#include <stdexcept> /* out_of_range */
#include <string>
#include <vector>

/* NOTE: assumes a valid map, so must be constructed *after* the map
 */
template <typename Table>
	impl::atomic_controller<Table>::atomic_controller(
			persist_atomic<typename Table::value_type> &persist_
			, table_t &map_
			, construction_mode mode_
		)
			: allocator_type(map_.get_allocator())
			, _persist(&persist_)
			, _map(&map_)
		{
			if ( mode_ == construction_mode::reconstitute )
			{
				/* reconstitute allocated memory */
				_persist->mod_key.reconstitute(allocator_type(*this));
				_persist->mod_mapped.reconstitute(allocator_type(*this));
				if ( 0 < _persist->mod_size )
				{
					allocator_type(*this).reconstitute(_persist->mod_size, _persist->mod_ctl);
				}
				else
				{
				}
			}
			redo();
		}

template <typename Table>
	auto impl::atomic_controller<Table>::redo() -> void
	{
		if ( _persist->mod_size != 0 )
		{
			if ( 0 < _persist->mod_size )
			{
				redo_update();
			}
			else /* Issue 41-style replacement */
			{
				redo_replace();
			}
		}
	}

template <typename Table>
	auto impl::atomic_controller<Table>::redo_finish() -> void
	{
		_persist->mod_size = 0;
		persist_range(&_persist->mod_size, &_persist->mod_size + 1, "atomic size");
	}

template <typename Table>
	auto impl::atomic_controller<Table>::redo_replace() -> void
	{
		_map->erase(_persist->mod_key);
		const auto *data_begin = _persist->mod_mapped.data();
		const auto *data_end = data_begin + _persist->mod_mapped.size();
                _map->emplace(
			std::piecewise_construct
			, std::forward_as_tuple(std::move(_persist->mod_key))
			, std::forward_as_tuple(data_begin, data_end, allocator_type(*this))
		);
		redo_finish();
	}

template <typename Table>
	auto impl::atomic_controller<Table>::redo_update() -> void
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
				persist_range(
					dst_first
					, std::copy(src_first, src_last, dst_first)
					, "atomic ctl"
				);
			}
		}
		catch ( const std::out_of_range & )
		{
			/* no such key */
		}
		std::size_t ct = _persist->mod_size;
		redo_finish();
		allocator_type(*this).deallocate(_persist->mod_ctl, ct);
	}

template <typename Table>
	void impl::atomic_controller<Table>::persist_range(
		const void *first_
		, const void *last_
		, const char *what_
	)
	{
		this->persist(first_, static_cast<const char *>(last_) - static_cast<const char *>(first_), what_);
	}

template <typename Table>
	void impl::atomic_controller<Table>::enter_replace(
		typename Table::allocator_type al_
		, typename Table::key_type &key
		, const char *data_
		, std::size_t data_len_
	)
	{
		_persist->mod_key = key;
		_persist->mod_mapped = typename Table::mapped_type(data_, data_ + data_len_, al_);
		/* 8-byte atomic write */
		_persist->mod_size = -1;
		this->persist(&_persist->mod_size, sizeof _persist->mod_size);
		redo();
	}

template <typename Table>
	void impl::atomic_controller<Table>::enter_update(
		typename Table::allocator_type al_
		, typename Table::key_type &key
		, std::vector<Component::IKVStore::Operation *>::const_iterator first
		, std::vector<Component::IKVStore::Operation *>::const_iterator last
	)
	{
		std::vector<char> src;
		std::vector<mod_control> mods;
		for ( ; first != last ; ++first )
		{
			switch ( (*first)->type() )
			{
			case Component::IKVStore::Op_type::WRITE:
				{
					const Component::IKVStore::Operation_write &wr =
						*static_cast<Component::IKVStore::Operation_write *>(
							*first
						);
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
				throw std::invalid_argument("Unknown update code " + std::to_string(int((*first)->type())));
			};
		}
		_persist->mod_key = key;
		_persist->mod_mapped =
			typename Table::mapped_type(
				src.begin()
				, src.end()
				, al_
			);

		using void_allocator_t =
			typename allocator_type::template rebind<void>::other;

		{
			auto ptr =
				allocator_type(*this).allocate(
					mods.size()
					, typename void_allocator_t::const_pointer()
					, "mod_ctl"
				);
			new (&*ptr) mod_control[mods.size()];
			_persist->mod_ctl = ptr;
		}

		std::copy(mods.begin(), mods.end(), &*_persist->mod_ctl);
		persist_range(
			&*_persist->mod_ctl
			, &*_persist->mod_ctl + mods.size()
			, "mod control"
		);
		/* 8-byte atomic write */
		_persist->mod_size = mods.size();
		this->persist(&_persist->mod_size, sizeof _persist->mod_size);
		redo();
	}
