#include "rp_alloc.h"

using namespace nupm;

template<>
Arena_allocator_volatile
Rp_allocator_volatile<0>::_arena(Rp_allocator_volatile::ARENA_PAGE_SIZE);

template<>
Arena_allocator_volatile
Rp_allocator_volatile<1>::_arena(Rp_allocator_volatile::ARENA_PAGE_SIZE);
