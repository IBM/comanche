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
