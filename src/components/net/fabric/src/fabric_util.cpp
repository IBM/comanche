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

#include "fabric_check.h" /* CHECK_FI_ERR */
#include "fabric_error.h"
#include "fabric_json.h"
#include "fabric_ptr.h"
#include "fabric_str.h" /* tostr */
#include "hints.h"

#include <rdma/fi_cm.h>
#include <rdma/fi_endpoint.h> /* fi_endpoint */
#include <rdma/fi_errno.h> /* fi_strerror */

#include <map>
#include <memory> /* shared_ptr */

/**
 * Fabric/RDMA-based network component
 *
 */

void fi_void_connect(::fid_ep &ep_, const ::fi_info &ep_info_, const void *addr_, const void *param_, size_t paramlen_)
try
{
  CHECK_FI_ERR(::fi_connect(&ep_, addr_, param_, paramlen_));
}
catch ( const fabric_error &e )
{
  throw e.add(tostr(ep_info_));
}

std::shared_ptr<::fid_fabric> make_fid_fabric(::fi_fabric_attr &attr_, void *context_)
try
{
  ::fid_fabric *f(nullptr);
  CHECK_FI_ERR(::fi_fabric(&attr_, &f, context_));
  FABRIC_TRACE_FID(f);
  return fid_ptr(f);
}
catch ( const fabric_error &e )
{
  throw e.add(tostr(attr_));
}

namespace {
  std::shared_ptr<::fi_info> make_fi_info(
    std::uint32_t version_
    , const char *node_
    , const char *service_
    , const ::fi_info *hints_
  )
  try
  {
    ::fi_info *f;
    CHECK_FI_ERR(::fi_getinfo(version_, node_, service_, 0, hints_, &f));
    return std::shared_ptr<::fi_info>(f,::fi_freeinfo);
  }
  catch ( const fabric_error &e_ )
  {
    if ( hints_ )
    {
      throw e_.add(tostr(*hints_));
    }
    throw;
  }

  std::shared_ptr<::fi_info> make_fi_info(const ::fi_info &hints)
  {
    return make_fi_info(FI_VERSION(FI_MAJOR_VERSION,FI_MINOR_VERSION), nullptr, nullptr, &hints);
  }
}

std::shared_ptr<::fi_info> make_fi_info(const std::string& json_configuration)
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

std::shared_ptr<::fi_info> make_fi_info()
{
  std::shared_ptr<::fi_info> info(::fi_allocinfo(), ::fi_freeinfo);
  if ( ! info )
  {
    throw fabric_bad_alloc("fi_info (alloc)");
  }
  return info;
}

std::shared_ptr<::fi_fabric_attr> make_fi_fabric_attr()
{
  std::shared_ptr<::fi_fabric_attr> fabric_attr(new ::fi_fabric_attr{});
  if ( ! fabric_attr )
  {
    throw fabric_bad_alloc("fi_fabric_attr (alloc)");
  }
  return fabric_attr;
}

std::shared_ptr<::fi_info> make_fi_infodup(const ::fi_info &info_, const std::string &)
{
  auto info = std::shared_ptr<::fi_info>(::fi_dupinfo(&info_), ::fi_freeinfo);
  if ( ! info )
  {
    throw fabric_bad_alloc("fi_info (dup)");
  }

  return info;
}

auto get_name(::fid_t fid) -> fabric_types::addr_ep_t
{
  size_t addrlen = 0;

  ::fi_getname(fid, nullptr, &addrlen);

  std::vector<char> name(addrlen);

  CHECK_FI_ERR(::fi_getname(fid, &*name.begin(), &addrlen));
  return fabric_types::addr_ep_t(name);
}

const std::map<int, const char *> event_name
{
  { FI_NOTIFY, "FI_NOTIFY" },
  { FI_CONNREQ, "FI_CONNREQ" },
  { FI_CONNECTED, "FI_CONNECTED" },
  { FI_SHUTDOWN, "FI_SHUTDOWN" },
  { FI_MR_COMPLETE, "FI_MR_COMPLETE" },
  { FI_AV_COMPLETE, "FI_AV_COMPLETE" },
  { FI_JOIN_COMPLETE, "FI_JOIN_COMPLETE" },
};

const char *get_event_name(std::uint32_t e)
{
  auto it = event_name.find(e);
  return it == event_name.end() ? "(unknown event)" : it->second;
}
