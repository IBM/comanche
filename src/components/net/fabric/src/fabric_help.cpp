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
#include "fabric_ptr.h"

#include <rapidjson/document.h>

/* fi_connect / fi_listen / fi_accept / fi_reject / fi_shutdown : Manage endpoint connection state.
 *      fi_setname / fi_getname / fi_getpeer : Set local, or return local or peer endpoint address.
 *     fi_join / fi_close / fi_mc_addr : Join, leave, or retrieve a multicast address.
 */
#include <rdma/fi_cm.h>
#include <rdma/fi_endpoint.h> /* fi_endpoint */

#include <cassert>
#include <iostream>
#include <map>
#include <memory> /* shared_ptr */

/** 
 * Fabric/RDMA-based network component
 * 
 */

std::map<std::string, int> cap_fwd = {
#define X(Y) {#Y, (Y)},
#include "fabric_caps.h"
#undef X
};

std::shared_ptr<fid_domain> make_fid_domain(fid_fabric &fabric, fi_info &info, void *context)
{
  fid_domain *f(nullptr);
  int i = fi_domain(&fabric, &info, &f, context);
  if ( i != FI_SUCCESS )
  {
    std::cout << "FABRIC at " << static_cast<void *>(&fabric) << "\n";
    std::cout << "INFO at " << static_cast<void *>(&info) << ":" << fi_tostr(&info, FI_TYPE_INFO) << "\n";
    throw fabric_error(i,__LINE__);
  }
  return fid_ptr(f);
}

std::shared_ptr<fid_fabric> make_fid_fabric(fi_fabric_attr &attr, void *context)
{
  fid_fabric *f(nullptr);
  int i = fi_fabric(&attr, &f, context);
  if ( i != FI_SUCCESS )
  {
    throw fabric_error(i,__LINE__);
  }
  return fid_ptr(f);
}

std::shared_ptr<fi_info> make_fi_info(
  int version
  , const char *node
  , const char *service
#if 0
  , uint64_t flags
#endif
  , const struct fi_info *hints
)
{
  fi_info *f;
  int i = fi_getinfo(version, node, service, 0, hints, &f);
  if ( i != FI_SUCCESS )
  {
    throw fabric_error(i,__LINE__);
  }
#if 0
  for ( auto j = f; j; j = j->next )
  {
    std::cout << "info at " << j << static_cast<void *>(j) << ":" << fi_tostr(j, FI_TYPE_INFO) << "\n";
  }
#endif
  return std::shared_ptr<fi_info>(f,fi_freeinfo);
}

std::shared_ptr<fi_info> make_fi_info(const std::string &, std::uint64_t /* caps */, fi_info &hints)
{
  /* the preferred provider string is ignored for now. */
  return make_fi_info(FI_VERSION(FI_MAJOR_VERSION,FI_MINOR_VERSION), nullptr, nullptr, &hints);
}

std::shared_ptr<fi_info> make_fi_info(fi_info &hints)
{
  return make_fi_info(FI_VERSION(FI_MAJOR_VERSION,FI_MINOR_VERSION), nullptr, nullptr, &hints);
}

std::shared_ptr<fi_info> make_info()
{
  std::shared_ptr<fi_info> info(fi_allocinfo(), fi_freeinfo);
  if ( ! info )
  {
    throw fabric_bad_alloc("fi_info");
  }
  return info;
}

class hints
{
  std::shared_ptr<fi_info> _info;
public:
  hints()
  : _info(make_info())
  {}
  hints &caps(int c) { _info->caps = c; return *this; }
  hints &mr_mode(int m) { _info->domain_attr->mr_mode = m; return *this; } // e.g. FI_MR_LOCAL | FI_MR_ALLOCATED | FI_MR_PROV_KEY | FI_MR_VIRT_ADDR;
  hints &prov_name(const char *n) {
    assert(! _info->fabric_attr->prov_name);
    _info->fabric_attr->prov_name = ::strdup(n); return *this;
  } // e.g. FI_MR_LOCAL | FI_MR_ALLOCATED | FI_MR_PROV_KEY | FI_MR_VIRT_ADDR;
  std::shared_ptr<fi_info> data() { return _info; }
};

std::shared_ptr<fi_info> make_fi_info_hints(std::uint64_t caps, int mr_mode)
{
  auto info = make_info();
  info->caps = caps;
  info->domain_attr->mr_mode = mr_mode; // FI_MR_LOCAL | FI_MR_ALLOCATED | FI_MR_PROV_KEY | FI_MR_VIRT_ADDR;
  return info;
}

std::shared_ptr<fi_info> make_fi_fabric_spec(const std::string& json_configuration)
{
  rapidjson::Document jdoc;
  jdoc.Parse(json_configuration.c_str());

  auto provider = jdoc.FindMember("preferred_provider");
  auto default_provider = "verbs";
  default_provider = "sockets";
  auto provider_str = std::string(provider != jdoc.MemberEnd() && provider->value.IsString() ? provider->value.GetString() : default_provider);

  std::uint64_t caps_int{0U};
  auto caps = jdoc.FindMember("caps");
  if ( caps != jdoc.MemberEnd() && caps->value.IsArray() )
  {
    for ( auto cap = caps->value.Begin(); cap != caps->value.End(); ++cap )
    {
      if ( cap->IsString() )
      {
        auto cap_int_i = cap_fwd.find(cap->GetString());
        if ( cap_int_i != cap_fwd.end() )
        {
          caps_int |= cap_int_i->second;
        }
      }
    }
  }
  return hints().caps(caps_int).mr_mode(FI_MR_LOCAL | FI_MR_ALLOCATED | FI_MR_PROV_KEY | FI_MR_VIRT_ADDR).prov_name(provider_str.c_str()).data();
}

std::shared_ptr<fid_ep> make_fid_aep(fid_domain &domain, fi_info &info, void *context)
{
  fid_ep *e;
  int i = fi_endpoint(&domain, &info, &e, context);
  static_assert(0 == FI_SUCCESS, "FI_SUCCESS not 0, which means that we need to distinguish between these types of \"successful\" returns");
  if ( i != FI_SUCCESS )
  {
    throw fabric_error(i,__LINE__);
  }
  return fid_ptr(e);
}

std::shared_ptr<fid_pep> make_fid_pep(fid_fabric &fabric, fi_info &info, void *context)
{
  fid_pep *e;
  int i = fi_passive_ep(&fabric, &info, &e, context);
  static_assert(0 == FI_SUCCESS, "FI_SUCCESS not 0, which means that we need to distinguish between these types of \"successful\" returns");
  if ( i != FI_SUCCESS )
  {
    throw fabric_error(i,__LINE__);
  }
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

std::shared_ptr<fi_info> make_fi_infodup(const fi_info *info_)
{
  auto info = std::shared_ptr<fi_info>(fi_dupinfo(info_), fi_freeinfo);
  if ( ! info )
  {
    throw fabric_bad_alloc("fi_info (dup)");
  }
  return info;
}

std::shared_ptr<fid_eq> make_fid_eq(fid_fabric &fabric, struct fi_eq_attr *attr, void *context)
{
  fid_eq *e;
  int i = fi_eq_open(&fabric, attr, &e, context);
  if ( i != 0 )
  {
    throw fabric_error(i,__LINE__);
  }
  return fid_ptr(e);
}

[[noreturn]] void not_implemented(const std::string &who)
{
  throw std::runtime_error{who + " not_implemented"};
}
[[noreturn]] void not_expected(const std::string &who)
{
  throw std::logic_error{who + " not_expected"};
}
