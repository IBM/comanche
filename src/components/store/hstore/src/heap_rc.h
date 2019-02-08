#ifndef COMANCHE_HSTORE_HEAP_RC_H
#define COMANCHE_HSTORE_HEAP_RC_H

#include "bad_alloc_cc.h"

#include "rc_alloc_wrapper.h"

#include <memory>
#include <cstddef> /* size_t, ptrdiff_t */

class heap_rc_shared
{
	std::size_t _size;
	unsigned _numa_node;
	nupm::Rca_AVL _heap;
public:
	heap_rc_shared(std::size_t sz_, unsigned numa_node_)
		: _size(sz_ - sizeof *this)
		, _numa_node(numa_node_)
		, _heap()
	{
		_heap.add_managed_region(this + 1, _size, _numa_node);
	}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Winit-self"
	heap_rc_shared()
		: _size(this->_size)
		, _numa_node(this->_numa_node)
		, _heap()
	{
		_heap.add_managed_region(this + 1, _size, _numa_node);
	}
#pragma GCC diagnostic pop

	void *alloc(std::size_t sz_)
	{
		return _heap.alloc(sz_, _numa_node, 64);
	}

	void inject_allocation(void * p_, std::size_t sz_)
	{
		return _heap.inject_allocation(p_, sz_, _numa_node);
	}

	void free(void *p_)
	{
		return _heap.free(p_, _numa_node);
	}
};

class heap_rc
{
	heap_rc_shared *_heap;
public:
	explicit heap_rc(void *area, std::size_t sz_, unsigned numa_node_)
		: _heap(new (area) heap_rc_shared(sz_, numa_node_))
	{}

	explicit heap_rc(void *area)
		: _heap(new (area) heap_rc_shared())
	{}

	heap_rc(const heap_rc &) = default;

	heap_rc & operator=(const heap_rc &) = default;

	void *alloc(std::size_t sz_)
	{
		return _heap->alloc(sz_);
	}

	void inject_allocation(void * p_, std::size_t sz_)
	{
		return _heap->inject_allocation(p_, sz_);
	}

	void free(void *p_)
	{
		return _heap->free(p_);
	}
};

#endif
