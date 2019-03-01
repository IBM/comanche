#include <assert.h>
#include "rampant_alloc.h"

namespace nupm
{


class Rampant_allocator::Rpmalloc
{
public:

};


Rampant_allocator::Rampant_allocator() :
  _allocator(std::make_unique<Rpmalloc>())
{
  assert(_allocator);
}

Rampant_allocator::~Rampant_allocator()
{
}

}


nupm::Rampant_allocator a;
