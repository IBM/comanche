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


/*
 * Authors:
 *
 */

#include "fabric_memory_control.h"

#include "fabric.h"
#include "fabric_check.h" /* CHECK_FI_ERR */
#include "fabric_ptr.h" /* FABRIC_TACE_FID */
#include "fabric_runtime_error.h"
#include "fabric_util.h" /* make_fi_infodup */
#include "pointer_cast.h"

#include <sys/uio.h> /* iovec */

#include <algorithm> /* find_if */
#include <iterator> /* back_inserter */
#include <memory> /* make_unique */
#include <stdexcept> /* domain_error, range_error */
#include <sstream> /* ostringstream */
#include <string>

using guard = std::unique_lock<std::mutex>;

namespace
{
  void *iov_end(const ::iovec &v)
  {
    return static_cast<char *>(v.iov_base) + v.iov_len;
  }
  /* True if range of a is a superset of range of b */
  bool covers(const ::iovec &a, const ::iovec &b)
  {
    return a.iov_base <= b.iov_base && iov_end(b) <= iov_end(a);
  }
  std::ostream &operator<<(std::ostream &o, const ::iovec &v)
  {
    return o << "[" << v.iov_base << ".." << iov_end(v) << ")";
  }
}

/**
 * Fabric/RDMA-based network component
 *
 */

/* Note: the info is owned by the caller, and must be copied if it is to be saved. */
Fabric_memory_control::Fabric_memory_control(
  Fabric &fabric_
  , fi_info &info_
)
  : _fabric(fabric_)
  , _domain_info(make_fi_infodup(info_, "domain"))
  /* NOTE: "this" is returned for context when domain-level events appear in the event queue bound to the domain
   * and not bound to a more specific entity (an endpoint, mr, av, pr scalable_ep).
   */
  , _domain(_fabric.make_fid_domain(*_domain_info, this))
  , _m{}
  , _mr_addr_to_mra{}
{
}

Fabric_memory_control::~Fabric_memory_control()
{
}

struct mr_and_address
{
  mr_and_address(::fid_mr *mr_, const void *addr_, std::size_t size_)
    : mr(fid_ptr(mr_))
    , v{const_cast<void *>(addr_), size_}
  {}
  std::shared_ptr<::fid_mr> mr;
  ::iovec v;
};

auto Fabric_memory_control::register_memory(const void * addr_, std::size_t size_, std::uint64_t key_, std::uint64_t flags_) -> Component::IFabric_connection::memory_region_t
{
  auto mra =
    std::make_unique<mr_and_address>(
      make_fid_mr_reg_ptr(addr_,
                                size_,
                                std::uint64_t(FI_SEND|FI_RECV|FI_READ|FI_WRITE|FI_REMOTE_READ|FI_REMOTE_WRITE),
                                key_,
                                flags_)
      , addr_
      , size_
    );

  assert(mra->mr);

  /* operations which access local memory will need the "mr." Record it here. */
  guard g{_m};

  auto it = _mr_addr_to_mra.emplace(addr_, std::move(mra));

  /*
   * Operations which access remote memory will need the memory key.
   * If the domain has FI_MR_PROV_KEY set, we need to return the actual key.
   */
#if 0
  std::cerr << "Registered mr " << it->second->mr << " " << it->second->v << "\n";
#endif
  return pointer_cast<Component::IFabric_memory_region>(&*it->second);
}

void Fabric_memory_control::deregister_memory(const memory_region_t mr_)
{
  /* recover the memory region as a unique ptr */
  auto mra = pointer_cast<mr_and_address>(mr_);

  guard g{_m};

  auto lb = _mr_addr_to_mra.lower_bound(mra->v.iov_base);
  auto ub = _mr_addr_to_mra.upper_bound(mra->v.iov_base);

  map_addr_to_mra::size_type scan_count = 0;
  auto it =
    std::find_if(
      lb
      , ub
      , [&mra, &scan_count] ( const map_addr_to_mra::value_type &m ) { ++scan_count; return m.second->mr == mra->mr; }
  );

  if ( it == ub )
  {
    std::ostringstream err;
    err << __func__ << " mr " << mra->mr << " (with range " << mra->v << ")"
      << " not found in " << scan_count << " of " << _mr_addr_to_mra.size() << " registry entries";
    throw std::logic_error(err.str());
  }
#if 0
  std::cerr << "Deregistered addr " << it->second->mr << " " << it->second->v << "\n";
#endif
  _mr_addr_to_mra.erase(it);
}

std::uint64_t Fabric_memory_control::get_memory_remote_key(const memory_region_t mr_) const noexcept
{
  /* recover the memory region */
  auto mr = &*pointer_cast<mr_and_address>(mr_)->mr;
  /* ask fabric for the key */
  return ::fi_mr_key(mr);
}

void *Fabric_memory_control::get_memory_descriptor(const memory_region_t mr_) const noexcept
{
  /* recover the memory region */
  auto mr = &*pointer_cast<mr_and_address>(mr_)->mr;
  /* ask fabric for the descriptor */
  return ::fi_mr_desc(mr);
}

/* If local keys are needed, one local key per buffer. */
std::vector<void *> Fabric_memory_control::populated_desc(const std::vector<::iovec> & buffers)
{
  return populated_desc(&*buffers.begin(), &*buffers.end());
}

/* find a registered memory region which covers the iovec range */
::fid_mr *Fabric_memory_control::covering_mr(const ::iovec &v)
{
  /* _mr_addr_to_mr is sorted by starting address.
   * Find the last acceptable starting address, and iterate
   * backwards through the map until we find a covering range
   * or we reach the start of the table.
   */

  guard g{_m};

  auto ub = _mr_addr_to_mra.upper_bound(v.iov_base);

  auto it =
    std::find_if(
      map_addr_to_mra::reverse_iterator(ub)
      , _mr_addr_to_mra.rend()
      , [&v] ( const map_addr_to_mra::value_type &m ) { return covers(m.second->v, v); }
    );

  if ( it == _mr_addr_to_mra.rend() )
  {
    std::ostringstream e;
    e << "No mapped region covers " << v;
    throw std::range_error(e.str());
  }

#if 0
  std::cerr << "covering_mr( " << v << ") found mr " << it->second->mr << " with range " << it->second->v << "\n";
#endif
  return &*it->second->mr;
}

std::vector<void *> Fabric_memory_control::populated_desc(const ::iovec *first, const ::iovec *last)
{
  std::vector<void *> desc;

  std::transform(
    first
    , last
    , std::back_inserter(desc)
    , [this] (const ::iovec &v) { return ::fi_mr_desc(covering_mr(v)); }
  );

  return desc;
}

/* (no context, synchronous only) */
/*
 * ERROR: the sixth parameter is named "requested key" in fi_mr_reg doc, but
 * if the user asks for a key which is unavailable the error returned is
 * "Required key not available." The parameter name and the error disagree:
 * "requested" is not the same as "required."
 */
fid_mr * Fabric_memory_control::make_fid_mr_reg_ptr(
  const void *buf
  , size_t len
  , uint64_t access
  , uint64_t key
  , uint64_t flags
) const
{
  ::fid_mr *f;
  auto constexpr offset = 0U; /* "reserved and must be zero" */
  /* used iff the registration completes asynchronously
   * (in which case the domain has been bound to an event queue with FI_REG_MR)
   */
  auto constexpr context = nullptr;
  try
  {
    /* Note: this was once observed to return "Cannot allocate memory" when called from JNI code. */
    /* Note: this was once observed to return an error when the DAX persistent Apache Pass memory
     * seemed properly aligned. The work-around was to issue a pre-emptive madvise(MADV_DONTFORK)
     * against the entire memory space of the DAX device.
     */
    /* Note: this was once observed to return "Bad address" when the (GPU) memory seemed properly aligned. */
    CHECK_FI_ERR(::fi_mr_reg(&*_domain, buf, len, access, offset, key, flags, &f, context));
  }
  catch ( const fabric_runtime_error &e )
  {
    std::ostringstream s;
    s << " in " << __func__ << " calling ::fi_mr_reg(domain " << &*_domain << " buf " << buf << ", len " << len << ", access " << access << ", offset " << offset << ", key " << key << ", flags " << flags << ", fid_mr " << &f << ", context " << static_cast<void *>(context) << ")";
    throw e.add(s.str());
  }
  FABRIC_TRACE_FID(f);
  return f;
}
