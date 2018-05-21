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

#include "fabric_util.h"

#include "fabric_error.h"
#include "fabric_json.h"
#include "fabric_ptr.h"

#include <rdma/fi_cm.h>
#include <rdma/fi_endpoint.h> /* fi_endpoint */

#include <cassert>
#include <cstring> /* strdup */
#include <memory> /* shared_ptr */
#include <mutex>

/**
 * Fabric/RDMA-based network component
 *
 */

/* bind fi types to their type ids */
template <typename T>
  struct str_attr;

template <>
  struct str_attr<fi_info>
  {
    static constexpr auto id = FI_TYPE_INFO;
  };

template <>
  struct str_attr<fi_fabric_attr>
  {
    static constexpr auto id = FI_TYPE_FABRIC_ATTR;
  };

namespace
{
  using guard = std::lock_guard<std::mutex>;
  std::mutex m_fi_tostr;
}

/* typesafe and threadsafe version of fi_tostr */
template <typename T>
  std::string tostr(const T &f)
  {
    guard g{m_fi_tostr};
    return fi_tostr(&f, str_attr<T>::id);
  }

void fi_void_connect(fid_ep &ep_, const fi_info &ep_info_, const void *addr_, const void *param_, size_t paramlen_)
try
{
  CHECKZ(fi_connect(&ep_, addr_, param_, paramlen_));
}
catch ( const fabric_error &e )
{
  throw e.add(tostr(ep_info_));
  // throw e.add(fi_tostr(&ep_info_, FI_TYPE_INFO));
}

#include <cassert>
std::shared_ptr<fid_domain> make_fid_domain(fid_fabric &fabric_, fi_info &info_, void *context_)
try
{
  fid_domain *f(nullptr);
  CHECKZ(fi_domain(&fabric_, &info_, &f, context_));
  return fid_ptr(f);
}
catch ( const fabric_error &e )
{
  throw e.add(tostr(info_));
}

std::shared_ptr<fid_fabric> make_fid_fabric(fi_fabric_attr &attr_, void *context_)
try
{
  fid_fabric *f(nullptr);
  CHECKZ(fi_fabric(&attr_, &f, context_));
  return fid_ptr(f);
}
catch ( const fabric_error &e )
{
  throw e.add(tostr(attr_));
}

#if 0
#include <iostream>
#endif
std::shared_ptr<fi_info> make_fi_info(
  std::uint32_t version_
  , const char *node_
  , const char *service_
  , const fi_info *hints_
)
try
{
  fi_info *f;
#if 0
  if ( hints_ )
  {
    std::cerr << "INFO hint for " << "make_fi_info" << ": " << tostr(*hints_) << "\n";
  }
#endif

  CHECKZ(fi_getinfo(version_, node_, service_, 0, hints_, &f));
  return std::shared_ptr<fi_info>(f,fi_freeinfo);
}
catch ( const fabric_error &e_ )
{
  if ( hints_ )
  {
    throw e_.add(tostr(*hints_));
  }
  throw;
}

std::shared_ptr<fi_info> make_fi_info(const std::string &, std::uint64_t /* caps */, const fi_info &hints)
{
  /* the preferred provider string is ignored for now. */
  return make_fi_info(FI_VERSION(FI_MAJOR_VERSION,FI_MINOR_VERSION), nullptr, nullptr, &hints);
}

std::shared_ptr<fi_info> make_fi_info(const fi_info &hints)
{
  return make_fi_info(FI_VERSION(FI_MAJOR_VERSION,FI_MINOR_VERSION), nullptr, nullptr, &hints);
}

class hints
{
  std::shared_ptr<fi_info> _info;
public:
  explicit hints()
    : _info(make_fi_info())
  {}
  explicit hints(std::shared_ptr<fi_info> info_)
    : _info(info_)
  {}
  hints &caps(uint64_t c) { _info->caps = c; return *this; }
  hints &mode(uint64_t c) { _info->mode = c; return *this; }
  hints &mr_mode(int m) { _info->domain_attr->mr_mode = m; return *this; }
  hints &prov_name(const char *n) {
    assert(! _info->fabric_attr->prov_name);
    _info->fabric_attr->prov_name = ::strdup(n); return *this;
  } // e.g. FI_MR_LOCAL | FI_MR_ALLOCATED | FI_MR_PROV_KEY | FI_MR_VIRT_ADDR;
  const char *prov_name() const { return _info->fabric_attr->prov_name; }
  const fi_info &data() { return *_info; }
};

std::shared_ptr<fi_info> make_fi_info(const std::string& json_configuration)
{
  auto h = hints(parse_info(json_configuration));

  const auto default_provider = "verbs";
  if ( h.prov_name() == nullptr )
  {
    h.prov_name(default_provider);
  }

  return
    make_fi_info(h.mode(FI_CONTEXT | FI_CONTEXT2).data());
}

std::shared_ptr<fid_ep> make_fid_aep(fid_domain &domain, fi_info &info, void *context)
try
{
  fid_ep *e;
  CHECKZ(fi_endpoint(&domain, &info, &e, context));
  static_assert(0 == FI_SUCCESS, "FI_SUCCESS not 0, which means that we need to distinguish between these types of \"successful\" returns");
  return fid_ptr(e);
}
catch ( const fabric_error &e )
{
  throw e.add(tostr(info));
}

std::shared_ptr<fid_pep> make_fid_pep(fid_fabric &fabric, fi_info &info, void *context)
{
  fid_pep *e;
  CHECKZ(fi_passive_ep(&fabric, &info, &e, context));
  static_assert(0 == FI_SUCCESS, "FI_SUCCESS not 0, which means that we need to distinguish between these types of \"successful\" returns");
  return fid_ptr(e);
}

std::shared_ptr<fid_pep> make_fid_pep_listener(fid_fabric &fabric, fi_info &info, fid_eq &eq, void *context)
{
  auto e = make_fid_pep(fabric, info, context);
  CHECKZ(fi_pep_bind(&*e, &eq.fid, 0U));
  CHECKZ(fi_listen(&*e));
  return e;
}

std::shared_ptr<fi_info> make_fi_info()
{
  std::shared_ptr<fi_info> info(fi_allocinfo(), fi_freeinfo);
  if ( ! info )
  {
    throw fabric_bad_alloc("fi_info (alloc)");
  }
  return info;
}

std::shared_ptr<fi_fabric_attr> make_fi_fabric_attr()
{
  std::shared_ptr<fi_fabric_attr> fabric_attr(new fi_fabric_attr{});
  if ( ! fabric_attr )
  {
    throw fabric_bad_alloc("fi_fabric_attr (alloc)");
  }
  return fabric_attr;
}

std::shared_ptr<fi_info> make_fi_infodup(const fi_info &info_, const std::string &)
{
  auto info = std::shared_ptr<fi_info>(fi_dupinfo(&info_), fi_freeinfo);
  if ( ! info )
  {
    throw fabric_bad_alloc("fi_info (dup)");
  }

  return info;
}

std::shared_ptr<fid_eq> make_fid_eq(fid_fabric &fabric, fi_eq_attr &attr, void *context)
{
  fid_eq *e;
  CHECKZ(fi_eq_open(&fabric, &attr, &e, context));
  return fid_ptr(e);
}

/* (no context, synchronous only) */
fid_mr * make_fid_mr_reg_ptr(
  fid_domain &domain, const void *buf, size_t len,
  uint64_t access, uint64_t key,
  uint64_t flags)
try
{
  fid_mr *mr;
  auto constexpr offset = 0U; /* "reserved and must be zero" */
  /* used iff the registration completes asynchronously
   * (in which case the domain has been bound to an event queue with FI_REG_MR)
   */
  auto constexpr context = nullptr;
  CHECKZ(fi_mr_reg(&domain, buf, len, access, offset, key, flags, &mr, context));
  return mr;
}
catch ( const fabric_error &e )
{
  throw e.add(std::string(std::string(" in ") + __func__ + " " + std::to_string(len)));
}

fid_unique_ptr<fid_cq> make_fid_cq(fid_domain &domain, fi_cq_attr &attr, void *context)
{
  fid_cq *cq;
  CHECKZ(fi_cq_open(&domain, &attr, &cq, context));
  return fid_unique_ptr<fid_cq>(cq);
}

[[noreturn]] void not_implemented(const std::string &who)
{
  throw std::runtime_error{who + " not_implemented"};
}
[[noreturn]] void not_expected(const std::string &who)
{
  throw std::logic_error{who + " not_expected"};
}

/* fi_fabric, fi_close (when called on a fabric) and most fi_poll functions FI_SUCCESS; others return 0 */
static_assert(FI_SUCCESS == 0, "FI_SUCCESS not zero");
/* most (all?) fabric functions return negative on error and 0 or positive on success */
void check_ge_zero(int r, const char *file, int line)
{
  if ( r < 0 )
  {
    throw fabric_error(-r, file, line);
  }
}

void check_ge_zero(ssize_t r, const char *file, int line)
{
  if ( r < 0 )
  {
    throw fabric_error(int(-r), file, line);
  }
}

auto get_name(fid_t fid) -> addr_ep_t
{
  size_t addrlen = 0;

  fi_getname(fid, nullptr, &addrlen);

  std::vector<char> name(addrlen);

  CHECKZ(fi_getname(fid, &*name.begin(), &addrlen));
  return addr_ep_t(name);
}
