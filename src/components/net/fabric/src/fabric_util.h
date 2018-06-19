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

#ifndef _FABRIC_UTIL_H_
#define _FABRIC_UTIL_H_

/*
 * Authors:
 *
 */

#include <rdma/fabric.h> /* fi_tostr */
#include <rdma/fi_errno.h> /* FI_AGAIN */

#include "fabric_types.h" /* addr_ep_t */
#include "fabric_ptr.h" /* fid_unique_ptr */

#include <unistd.h> /* ssize_t */

#include <cstddef> /* size_t */
#include <cstdint> /* uinat32_t, uint64_t */
#include <memory> /* shared_ptr */
#include <mutex>
#include <string>

struct fi_cq_attr;
struct fi_info;
struct fi_eq_attr;
struct fi_fabric_attr;

struct fid_cq;
struct fid_domain;
struct fid_fabric;
struct fid_ep;
struct fid_eq;
struct fid_mr;
struct fid_pep;

/**
 * Fabric/RDMA-based network component
 *
 */
void fi_void_connect(::fid_ep &ep, const ::fi_info &ep_info, const void *addr, const void *param, std::size_t paramlen);

std::shared_ptr<::fid_fabric> make_fid_fabric(::fi_fabric_attr &attr, void *context);

std::shared_ptr<::fi_info> make_fi_info(const std::string& json_configuration);

std::shared_ptr<::fi_info> make_fi_info();

std::shared_ptr<::fi_info> make_fi_infodup(const ::fi_info &info_, const std::string &why_);

auto get_name(::fid_t fid) -> fabric_types::addr_ep_t;

const char *get_event_name(std::uint32_t e);

#endif
