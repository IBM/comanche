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

#include "fabric_help.h"

#include "fabric_error.h"
#include "fabric_json.h"
#include "fabric_ptr.h"

#include <rdma/fi_cm.h>
#include <rdma/fi_endpoint.h> /* fi_endpoint */

#include <cassert>
#include <cstring> /* strdup */
#include <map>
#include <memory> /* shared_ptr */

/** 
 * Fabric/RDMA-based network component
 * 
 */

std::shared_ptr<fid_domain> make_fid_domain(fid_fabric &fabric, fi_info &info, void *context)
{
  fid_domain *f(nullptr);
  CHECKZ(fi_domain(&fabric, &info, &f, context));
  return fid_ptr(f);
}

std::shared_ptr<fid_fabric> make_fid_fabric(fi_fabric_attr &attr, void *context)
{
  fid_fabric *f(nullptr);
  try
  {
    CHECKZ(fi_fabric(&attr, &f, context));
  }
  catch ( const fabric_error &e )
  {
    throw e.add(fi_tostr(&attr, FI_TYPE_FABRIC_ATTR));
  }
  return fid_ptr(f);
}

std::shared_ptr<fi_info> make_fi_info(
  int version
  , const char *node
  , const char *service
  , const struct fi_info *hints
)
{
  fi_info *f;
  CHECKZ(fi_getinfo(version, node, service, 0, hints, &f));
  return std::shared_ptr<fi_info>(f,fi_freeinfo);
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
  hints &caps(int c) { _info->caps = c; return *this; }
  hints &mr_mode(int m) { _info->domain_attr->mr_mode = m; return *this; } // e.g. FI_MR_LOCAL | FI_MR_ALLOCATED | FI_MR_PROV_KEY | FI_MR_VIRT_ADDR;
  hints &prov_name(const char *n) {
    assert(! _info->fabric_attr->prov_name);
    _info->fabric_attr->prov_name = ::strdup(n); return *this;
  } // e.g. FI_MR_LOCAL | FI_MR_ALLOCATED | FI_MR_PROV_KEY | FI_MR_VIRT_ADDR;
  const char *prov_name() const { return _info->fabric_attr->prov_name; }
  const fi_info &data() { return *_info; }
};

std::shared_ptr<fi_info> make_fi_fabric_spec(const std::string& json_configuration)
{
  auto h = hints(parse_info(nullptr, json_configuration));
#if 0
  const auto default_provider = "sockets";
#else
  const auto default_provider = "verbs";
#endif
  if ( h.prov_name() == nullptr )
  {
    h.prov_name(default_provider);
  }
 
  return make_fi_info(h.mr_mode(FI_MR_LOCAL | FI_MR_ALLOCATED | FI_MR_PROV_KEY | FI_MR_VIRT_ADDR).data());
}

std::shared_ptr<fid_ep> make_fid_aep(fid_domain &domain, fi_info &info, void *context)
{
  fid_ep *e;
  try
  {
    CHECKZ(fi_endpoint(&domain, &info, &e, context));
  }
  catch ( const fabric_error &e )
  {
    throw e.add(fi_tostr(&domain, FI_TYPE_DOMAIN_ATTR));
  }
  static_assert(0 == FI_SUCCESS, "FI_SUCCESS not 0, which means that we need to distinguish between these types of \"successful\" returns");
  return fid_ptr(e);
}

std::shared_ptr<fid_pep> make_fid_pep(fid_fabric &fabric, fi_info &info, void *context)
{
  fid_pep *e;
  CHECKZ(fi_passive_ep(&fabric, &info, &e, context));
  static_assert(0 == FI_SUCCESS, "FI_SUCCESS not 0, which means that we need to distinguish between these types of \"successful\" returns");
  return fid_ptr(e);
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

std::shared_ptr<fi_info> make_fi_infodup(const fi_info &info_, const std::string &)
{
  auto info = std::shared_ptr<fi_info>(fi_dupinfo(&info_), fi_freeinfo);
  if ( ! info )
  {
    throw fabric_bad_alloc("fi_info (dup)");
  }
  return info;
}

std::shared_ptr<fid_eq> make_fid_eq(fid_fabric &fabric, struct fi_eq_attr *attr, void *context)
{
  fid_eq *e;
  CHECKZ(fi_eq_open(&fabric, attr, &e, context));
  return fid_ptr(e);
}

/* (no context, syncornous only) */
fid_unique_ptr<fid_mr> make_fid_mr_reg(
  fid_domain &domain, const void *buf, size_t len,
  uint64_t access, uint64_t key,
  uint64_t flags)
{
  fid_mr *mr;
  auto constexpr offset = 0U; /* "reserved and must be zero" */
  auto constexpr context = nullptr; /* used iff domain has been bound to an event queue with FI_REG_MR */
  CHECKZ(fi_mr_reg(&domain, buf, len, access, offset, key, flags, &mr, context));
  return fid_unique_ptr<fid_mr>(mr);
}

fid_unique_ptr<fid_cq> make_fid_cq(fid_domain &domain, fi_cq_attr &attr, void *context)
{
  fid_cq *cq;
  CHECKZ(fi_cq_open(&domain, &attr, &cq, context));
  return fid_unique_ptr<fid_cq>(cq);
}

fid_unique_ptr<fid_av> make_fid_av(fid_domain &domain, fi_av_attr &attr, void *context)
{
  fid_av *av;
  CHECKZ(fi_av_open(&domain, &attr, &av, context));
  return fid_unique_ptr<fid_av>(av);
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
void check_ge_zero(int r, const char *file, unsigned line)
{
  if ( r < 0 )
  {
    throw fabric_error(-r, file, line);
  }
}
