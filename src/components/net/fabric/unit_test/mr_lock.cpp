#include "mr_lock.h"

#include "eyecatcher.h"
#include <sys/mman.h> /* mlock, munlock */
#include <exception>
#include <iostream>
#include <string>
#include <system_error>

mr_lock::mr_lock(const void *addr_, std::size_t len_)
  : _addr(addr_)
  , _len(len_)
{
  if ( 0 != ::mlock(_addr, _len) )
  {
    auto e = errno;
    auto context = std::string("in ") + __func__;
    throw std::system_error{std::error_code{e, std::system_category()}, context};
  }
}

mr_lock::mr_lock(mr_lock &&other)
  : _addr(other._addr)
  , _len(other._len)
{
  other._addr = nullptr;
}

mr_lock &mr_lock::operator=(mr_lock &&other)
{
  using std::swap;
  swap(_addr, other._addr);
  swap(_len, other._len);
  return *this;
}

mr_lock::~mr_lock()
try
{
  if ( _addr )
  {
    ::munlock(_addr, _len);
  }
}
catch ( std::exception &e )
{
  std::cerr << __func__ << " exception " << e.what() << eyecatcher << std::endl;
}
