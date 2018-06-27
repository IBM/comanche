/*
   Copyright [2018] [IBM Corporation]

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
#include "fabric_error.h"
#include "fabric_ptr.h" /* FABRIC_TACE_FID */
#include "fabric_util.h" /* make_fi_infodup */

#include <sys/uio.h> /* iovec */

#include <string>

using guard = std::unique_lock<std::mutex>;

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
  , _mr_addr_to_desc{}
  , _mr_desc_to_addr{}
{
}

Fabric_memory_control::~Fabric_memory_control()
{
}


/**
 * Register buffer for RDMA
 *
 * @param contig_addr Pointer to contiguous region
 * @param size Size of buffer in bytes
 * @param flags Flags e.g., FI_REMOTE_READ|FI_REMOTE_WRITE
 *
 * @return Memory region handle
 */
auto Fabric_memory_control::register_memory(const void * contig_addr, size_t size, std::uint64_t key, std::uint64_t flags) -> Component::IFabric_connection::memory_region_t
{
  auto mr = make_fid_mr_reg_ptr(contig_addr, size, std::uint64_t(FI_SEND|FI_RECV|FI_READ|FI_WRITE|FI_REMOTE_READ|FI_REMOTE_WRITE), key, flags);

  /* operations which access local memory will need the memory "descriptor." Record it here. */
  auto desc = ::fi_mr_desc(mr);
  {
    guard g{_m};
    _mr_addr_to_desc[contig_addr] = desc;
    _mr_desc_to_addr[desc] = contig_addr;
  }

  /*
   * Operations which access remote memory will need the memory key.
   * If the domain has FI_MR_PROV_KEY set, we need to return the actual key.
   */
  return Component::IFabric_connection::memory_region_t{mr, ::fi_mr_key(mr)};
}

  /**
   * De-register memory region
   *
   * @param memory_region
   */
void Fabric_memory_control::deregister_memory(const memory_region_t mr_)
{
  /* recover the memory region as a unique ptr */
  auto mr = fid_ptr(static_cast<::fid_mr *>(std::get<0>(mr_)));

  {
    auto desc = ::fi_mr_desc(&*mr);
    guard g{_m};
    auto itr = _mr_desc_to_addr.find(desc);
    assert(itr != _mr_desc_to_addr.end());
    _mr_addr_to_desc.erase(itr->first);
    _mr_desc_to_addr.erase(itr);
  }
}

/* If local keys are needed, one local key per buffer. */
std::vector<void *> Fabric_memory_control::populated_desc(const std::vector<iovec> & buffers)
{
  std::vector<void *> desc;
  for ( const auto it : buffers )
  {
    {
      guard g{_m};
      /* find a key equal to k or, if none, the largest key less than k */
      auto dit = _mr_addr_to_desc.lower_bound(it.iov_base);
      /* If not at k, lower_bound has left us with an iterator beyond k. Back up */
      if ( dit->first != it.iov_base && dit != _mr_addr_to_desc.begin() )
      {
        --dit;
      }

      desc.emplace_back(dit->second);
    }
  }
  return desc;
}

/* (no context, synchronous only) */
fid_mr * Fabric_memory_control::make_fid_mr_reg_ptr(
  const void *buf
  , size_t len
  , uint64_t access
  , uint64_t key
  , uint64_t flags
) const
try
{
  ::fid_mr *f;
  auto constexpr offset = 0U; /* "reserved and must be zero" */
  /* used iff the registration completes asynchronously
   * (in which case the domain has been bound to an event queue with FI_REG_MR)
   */
  auto constexpr context = nullptr;
  CHECK_FI_ERR(::fi_mr_reg(&*_domain, buf, len, access, offset, key, flags, &f, context));
  FABRIC_TRACE_FID(f);
  return f;
}
catch ( const fabric_error &e )
{
  throw e.add(std::string(std::string(" in ") + __func__ + " " + std::to_string(len)));
}
