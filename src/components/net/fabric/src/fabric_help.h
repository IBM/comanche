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

#ifndef _FABRIC_HELP_H_
#define _FABRIC_HELP_H_

/* 
 * Authors: 
 * 
 */

#include <cstdint>
#include <map>
#include <memory> /* shared_ptr */

struct fi_info;
struct fi_eq_attr;
struct fi_fabric_attr;

struct fid_domain;
struct fid_fabric;
struct fid_ep;
struct fid_eq;
struct fid_pep;

#include <string>


/** 
 * Fabric/RDMA-based network component
 * 
 */

std::shared_ptr<fid_domain> make_fid_domain(fid_fabric &fabric, fi_info &info, void *context);

std::shared_ptr<fid_fabric> make_fid_fabric(fi_fabric_attr &attr, void *context);

std::shared_ptr<fi_info> make_fi_info(int version, const char *node, const char *service, const fi_info *hints);

std::shared_ptr<fi_info> make_fi_info(const std::string &, std::uint64_t caps, fi_info &hints);

std::shared_ptr<fi_info> make_fi_info(fi_info &hints);

std::shared_ptr<fi_info> make_fi_fabric_spec(const std::string& json_configuration);

std::shared_ptr<fid_ep> make_fid_aep(fid_domain &domain, fi_info &info, void *context);

std::shared_ptr<fid_pep> make_fid_pep(fid_fabric &fabric, fi_info &info, void *context);

std::shared_ptr<fid_eq> make_fid_eq(fid_fabric &fabric, fi_eq_attr *attr, void *context);

std::shared_ptr<fi_info> make_fi_info();

std::shared_ptr<fi_info> make_fi_infodup(const fi_info *info_);

[[noreturn]] void not_expected(const std::string &who);
[[noreturn]] void not_implemented(const std::string &who);

#endif
