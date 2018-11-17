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

#include "fabric_bad_alloc.h"
#include "fabric_check.h" /* CHECK_FI_ERR */

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wcast-align"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wshadow"
#include <rdma/fi_cm.h> /* fi_getname */
#pragma GCC diagnostic pop

#include <map>
#include <memory> /* shared_ptr */

/**
 * Fabric/RDMA-based network component
 *
 */

std::shared_ptr<::fi_info> make_fi_info()
{
  std::shared_ptr<::fi_info> info(::fi_allocinfo(), ::fi_freeinfo);
  if ( ! info )
  {
    throw fabric_bad_alloc("fi_info (alloc)");
  }
  return info;
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

const std::map<std::uint32_t, const char *> event_name
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
