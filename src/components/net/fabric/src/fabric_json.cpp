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

#include "fabric_json.h"

#include "fabric_util.h" /* make_fi_info */

#include <rapidjson/document.h>
#include <rapidjson/error/en.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#include <rdma/fabric.h> /* fi_info */
#pragma GCC diagnostic pop
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wcast-align"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wshadow"
#include <rdma/fi_endpoint.h> /* fi_tx_attr, fi_rx_attr */
#pragma GCC diagnostic pop

#include <cstdint> /* uint32_t, uint64_t */
#include <cstring> /* strdup */
#include <map>
#include <stdexcept> /* domain_error */
#include <string>
#include <vector>

/**
 * Fabric/RDMA-based network component
 *
 */

/*
 * map of strings to enumerates values
 */
template <typename T>
  using translate_map = const std::map<std::string, T>;

/*
 * map of strings to json parse functions
 */
template <typename S>
  using parse_map = translate_map<void (*)(const rapidjson::Value &, S *)>;

using json_value = const rapidjson::Value &;

namespace
{
  /*
   * translation of names (as strings) to named constants (defines, consts, or enums):
   * Each set of named strings is enumreated in an include file.
   */

  const translate_map<std::uint64_t> map_caps
  {
#define X(Y) { #Y, (Y) },
#include "bits_caps.h"
#undef X
  };

  const translate_map<std::uint64_t> map_mode
  {
#define X(Y) { #Y, (Y) },
#include "bits_mode.h"
#undef X
  };

  const translate_map<std::uint64_t> map_op_flags
  {
#define X(Y) { #Y, (Y) },
#include "bits_op_flags.h"
#undef X
  };

  const translate_map<std::uint64_t> map_msg_order
  {
#define X(Y) { #Y, (Y) },
#include "bits_msg_order.h"
#undef X
  };

  const translate_map<std::uint64_t> map_comp_order
  {
#define X(Y) { #Y, (Y) },
#include "bits_comp_order.h"
#undef X
  };

  const translate_map<std::uint32_t> map_addr_format
  {
#define X(Y) { #Y, (Y) },
#include "enum_addr_format.h"
#undef X
  };

  const translate_map<fi_ep_type> map_ep_type
  {
#define X(Y) { #Y, (Y) },
#include "enum_ep_type.h"
#undef X
  };

  const translate_map<std::uint32_t> map_ep_protocol
  {
#define X(Y) { #Y, (Y) },
#include "enum_ep_protocol.h"
#undef X
  };

  const translate_map<fi_threading> map_threading
  {
#define X(Y) { #Y, (Y) },
#include "enum_threading.h"
#undef X
  };

  const translate_map<fi_progress> map_progress
  {
#define X(Y) { #Y, (Y) },
#include "enum_progress.h"
#undef X
  };

  const translate_map<fi_resource_mgmt> map_resource_mgmt
  {
#define X(Y) { #Y, (Y) },
#include "enum_resource_mgmt.h"
#undef X
  };

  const translate_map<fi_av_type> map_av_type
  {
#define X(Y) { #Y, (Y) },
#include "enum_av_type.h"
#undef X
  };

  const translate_map<int> map_mr_mode
  {
#define X(Y) { #Y, (Y) },
#include "bits_mr_mode.h"
#undef X
  };

  /*
   * legend:
   *  V : the type of a structure field
   *  S : the type of the containing structure
   *  M : the offset of the structure field within the structure
   *  X : the translate or action "map" for the field
   */

  /* parsing */

  template <typename V>
    V parse_bitset(json_value &v, translate_map<V> &m)
    {
      if ( ! v.IsArray() )
      {
        throw std::domain_error(std::string{": flags specification not an array: "} + v.GetString());
      }
      V u{0U};
      for ( auto s = v.Begin(); s != v.End(); ++s )
      {
        if ( ! s->IsString() )
        {
          throw std::domain_error("non-string");
        }

        auto it = m.find(s->GetString());
        if ( it == m.end() )
        {
          throw std::domain_error(std::string{": unrecognized value "} + s->GetString());
        }

        u |= it->second;
      }
      return u;
    }

  template <typename V>
    V parse_scalar_mapped(json_value &v, translate_map<V> &m)
    {
      if ( ! v.IsString() )
      {
        throw std::domain_error("non-string");
      }

      auto s = v.GetString();
      auto it = m.find(s);
      if ( it == m.end() )
      {
        throw std::domain_error(std::string{"unrecognized value: "} + s);
      }
      return it->second;
    }

  template <typename V>
    V parse_scalar(json_value &v);

  template <>
    std::uint32_t parse_scalar<uint32_t>(json_value &v)
    {
      if ( ! v.IsUint() )
      {
        throw std::domain_error("not a uint32");
      }
      return v.GetUint();
    }

  template <>
    std::uint64_t parse_scalar<uint64_t>(json_value &v)
    {
      if ( ! v.IsUint64() )
      {
        throw std::domain_error("not a uint64");
      }
      return v.GetUint64();
    }

  template <>
    const char *parse_scalar<const char *>(json_value &v)
    {
      if ( ! v.IsString() )
      {
        throw std::domain_error("not a string");
      }
      return v.GetString();
    }

  std::vector<uint8_t> parse_array_uint8(json_value &v)
  {
    std::vector<uint8_t> a;
    if ( ! v.IsArray() )
    {
      throw std::domain_error("not an array");
    }
    for ( unsigned i = 0; i != v.Size(); ++i )
    {
      const auto &e = v[i];
      if ( ! e.IsUint() )
      {
        throw std::domain_error("not a uint element");
      }
      a.emplace_back(e.GetUint());
    }
    return a;
  }

  /* assignment */

  template <typename V>
    class Assign
    {
  public:
    template <typename S, V S::*M>
      static void assign_scalar(json_value &v, S *s)
      {
        s->*M = parse_scalar<V>(v);
      }
    };

    /* Setting a scalar char *value: parse and assign, allocating new memory and freeing old */
  template <>
    class Assign<char *>
    {
  public:
    using V = char *;
    template <typename S, V S::*M>
      static void assign_scalar(json_value &v, S *s)
      {
        auto st = parse_scalar<const char *>(v);
        ::free(s->*M);
        s->*M = ::strdup(st);
      }
    };

  /* GetString returns a char *, so unsuitable for binary data
   */
  std::size_t assign_array_void(json_value &v, void **p)
  {
    auto a = parse_array_uint8(v);
    auto k = static_cast<uint8_t *>(std::malloc(a.size()));
    std::move(a.begin(), a.end(), k);
    std::free(*p);
    *p = k;
    return a.size();
  }

  std::size_t assign_array_uint8(json_value &v, std::uint8_t **p)
  {
    auto a = parse_array_uint8(v);
    auto k = static_cast<uint8_t *>(std::malloc(a.size()));
    std::move(a.begin(), a.end(), k);
    std::free(*p);
    *p = k;
    return a.size();
  }

  std::size_t assign_addr_str(json_value &v, void **p)
  {
    if ( v.IsString() )
    {
      auto st = ::strdup(v.GetString());
      std::free(*p);
      *p = st;
      return std::strlen(st);
    }
    else
    {
      return assign_array_void(v, p);
    }
  }

  template <typename S, typename V, V S::*M, translate_map<V> *X>
    void assign_bitset(json_value &v, S *s)
    {
      s->*M = parse_bitset<V>(v, *X);
    }

  template <void * fi_info::*M, std::size_t fi_info::*L>
    void assign_addr(json_value &v, fi_info *info)
    {
      auto assign_fn =
        info->addr_format == FI_ADDR_STR
        /* special case: the address is a character string interpreted by the fi provider */
        ? assign_addr_str
        /* normal case: we do not interpret the address */
        : assign_array_void
        ;

      info->*L = assign_fn(v, &(info->*M));
    }

  template <typename S>
    void assign_auth_key(json_value &v, S *s)
    {
      s->auth_key_size = assign_array_uint8(v, &s->auth_key);
    }

  template <typename S, typename V, V S::*M, translate_map<V> *X>
    void assign_scalar_map(json_value &v, S *s)
    {
      s->*M = parse_scalar_mapped<V>(v, *X);
    }

  template <typename T, parse_map<T> *X, T *fi_info::*M>
    void parse_struct(json_value &v, fi_info *info)
    {
      if ( ! (info->*M) )
      {
        throw std::domain_error(": missing substructure");
      }

      if ( ! v.IsObject() )
      {
        throw std::domain_error{": not a JSON object (" + std::to_string(v.GetType()) + ")"};
      }

      for ( auto it = v.MemberBegin(); it != v.MemberEnd(); ++it )
      {
        auto k = it->name.GetString();
        auto iv = X->find(k);
        try
        {
          if ( iv == X->end() )
          {
            throw std::domain_error(": unrecognized key");
          }
          (iv->second)(it->value, info->*M);
        }
        catch (const std::domain_error &e)
        {
          throw std::domain_error{std::string{k} + " " + e.what()};
        }
      }
    }

#define SET_SCALAR(S,M) Assign<decltype(S::M)>::assign_scalar<S, &S::M>
#define SET_BITSET(S,M,X) assign_bitset<S, decltype(S::M), &S::M, &X>
#define SET_SCALAR_MAPPED(S,M,X) assign_scalar_map<S, decltype(S::M), &S::M, &X>

  parse_map<fi_tx_attr> map_tx_attr
  {
    { "caps", SET_BITSET(fi_tx_attr,caps,map_caps) },
    { "mode", SET_BITSET(fi_tx_attr,mode,map_mode) },
    { "op_flags", SET_BITSET(fi_tx_attr,op_flags,map_op_flags) },
    { "msg_order", SET_BITSET(fi_tx_attr,msg_order,map_msg_order) },
    { "comp_order", SET_BITSET(fi_tx_attr,comp_order,map_comp_order) },
    { "inject_size", SET_SCALAR(fi_tx_attr,inject_size) },
    { "size", SET_SCALAR(fi_tx_attr,size) },
    { "iov_limit", SET_SCALAR(fi_tx_attr,iov_limit) },
    { "rma_iov_limit", SET_SCALAR(fi_tx_attr,rma_iov_limit) },
  };

  parse_map<fi_rx_attr> map_rx_attr
  {
    { "caps", SET_BITSET(fi_rx_attr,caps,map_caps) },
    { "mode", SET_BITSET(fi_rx_attr,mode,map_mode) },
    { "op_flags", SET_BITSET(fi_rx_attr,op_flags,map_op_flags) },
    { "msg_order", SET_BITSET(fi_rx_attr,msg_order,map_msg_order) },
    { "comp_order", SET_BITSET(fi_rx_attr,comp_order,map_comp_order) },
    { "total_buffered_recv", SET_SCALAR(fi_rx_attr,total_buffered_recv) },
    { "size", SET_SCALAR(fi_rx_attr,size) },
    { "iov_limit", SET_SCALAR(fi_rx_attr,iov_limit) },
  };

  parse_map<fi_ep_attr> map_ep_attr
  {
    { "type", SET_SCALAR_MAPPED(fi_ep_attr,type,map_ep_type) },
    { "protocol", SET_SCALAR_MAPPED(fi_ep_attr,protocol,map_ep_protocol) },
    { "protocol_version", SET_SCALAR(fi_ep_attr,protocol_version) },
    { "max_msg_size", SET_SCALAR(fi_ep_attr,max_msg_size) },
    { "msg_prefix_size", SET_SCALAR(fi_ep_attr,msg_prefix_size) },
    { "max_order_raw_size", SET_SCALAR(fi_ep_attr,max_order_raw_size) },
    { "max_order_war_size", SET_SCALAR(fi_ep_attr,max_order_war_size) },
    { "max_order_waw_size", SET_SCALAR(fi_ep_attr,max_order_waw_size) },
    { "mem_tag_format", SET_SCALAR(fi_ep_attr,mem_tag_format) },
    { "tx_ctx_cnt", SET_SCALAR(fi_ep_attr,tx_ctx_cnt) },
    { "rx_ctx_cnt", SET_SCALAR(fi_ep_attr,rx_ctx_cnt) },
    { "auth_key", assign_auth_key },
  };

  parse_map<fi_domain_attr> map_domain_attr
  {
    { "name", SET_SCALAR(fi_domain_attr,name) },
    { "threading", SET_SCALAR_MAPPED(fi_domain_attr,threading,map_threading) },
    { "control_progress", SET_SCALAR_MAPPED(fi_domain_attr,control_progress,map_progress) },
    { "data_progress", SET_SCALAR_MAPPED(fi_domain_attr,data_progress,map_progress) },
    { "resource_mgmt", SET_SCALAR_MAPPED(fi_domain_attr,resource_mgmt,map_resource_mgmt) },
    { "av_type", SET_SCALAR_MAPPED(fi_domain_attr,av_type,map_av_type) },
    { "mr_mode", SET_BITSET(fi_domain_attr,mr_mode,map_mr_mode) },
    { "mr_key_size", SET_SCALAR(fi_domain_attr,mr_key_size) },
    { "cq_data_size", SET_SCALAR(fi_domain_attr,cq_data_size) },
    { "cq_cnt", SET_SCALAR(fi_domain_attr,cq_cnt) },
    { "ep_cnt", SET_SCALAR(fi_domain_attr,ep_cnt) },
    { "tx_ctx_cnt", SET_SCALAR(fi_domain_attr,tx_ctx_cnt) },
    { "rx_ctx_cnt", SET_SCALAR(fi_domain_attr,rx_ctx_cnt) },
    { "max_ep_tx_ctx", SET_SCALAR(fi_domain_attr,max_ep_tx_ctx) },
    { "max_ep_rx_ctx", SET_SCALAR(fi_domain_attr,max_ep_rx_ctx) },
    { "max_ep_stx_ctx", SET_SCALAR(fi_domain_attr,max_ep_stx_ctx) },
    { "max_ep_srx_ctx", SET_SCALAR(fi_domain_attr,max_ep_srx_ctx) },
    { "cntr_cnt", SET_SCALAR(fi_domain_attr,cntr_cnt) },
    { "mr_iov_limit", SET_SCALAR(fi_domain_attr,mr_iov_limit) },
    { "caps", SET_BITSET(fi_domain_attr,caps,map_caps) },
    { "mode", SET_BITSET(fi_domain_attr,mode,map_mode) },
    { "auth_key", assign_auth_key },
    { "max_err_data", SET_SCALAR(fi_domain_attr,max_err_data) },
    { "mr_cnt", SET_SCALAR(fi_domain_attr,mr_cnt) },
  };

  parse_map<fi_fabric_attr> map_fabric_attr
  {
    { "name", SET_SCALAR(fi_fabric_attr,name) },
    { "prov_name", SET_SCALAR(fi_fabric_attr,prov_name) },
    { "prov_version", SET_SCALAR(fi_fabric_attr,prov_version) },
    { "api_version", SET_SCALAR(fi_fabric_attr,api_version) },
  };

  parse_map<fi_info> map_info
  {
    { "caps", SET_BITSET(fi_info,caps,map_caps) },
    { "mode", SET_BITSET(fi_info,mode,map_mode) },
    { "addr_format", SET_SCALAR_MAPPED(fi_info,addr_format,map_addr_format) },
    { "src", assign_addr<&fi_info::src_addr, &fi_info::src_addrlen> },
    { "dest", assign_addr<&fi_info::dest_addr, &fi_info::dest_addrlen> },
    { "tx_attr", { parse_struct<fi_tx_attr, &map_tx_attr, &fi_info::tx_attr> } },
    { "rx_attr", { parse_struct<fi_rx_attr, &map_rx_attr, &fi_info::rx_attr> } },
    { "ep_attr", { parse_struct<fi_ep_attr, &map_ep_attr, &fi_info::ep_attr> } },
    { "domain_attr", { parse_struct<fi_domain_attr, &map_domain_attr, &fi_info::domain_attr> } },
    { "fabric_attr", { parse_struct<fi_fabric_attr, &map_fabric_attr, &fi_info::fabric_attr> } },
  };

  std::shared_ptr<fi_info> parse_info(std::shared_ptr<fi_info> info, const rapidjson::Document &v)
  {
    if ( ! v.IsObject() )
    {
      throw std::domain_error{": NOT a JSON object (" + std::to_string(v.GetType()) + ")"};
    }
    for ( auto it = v.MemberBegin(); it != v.MemberEnd(); ++it )
    {
      auto k = it->name.GetString();
      auto iv = map_info.find(k);
      try
      {
        if ( iv == map_info.end() )
        {
          throw std::domain_error(": unrecognized key");
        }
        (iv->second)(it->value, &*info);
      }
      catch (const std::domain_error &e)
      {
        throw std::domain_error{std::string{k} + " " + e.what()};
      }
    }
    return info;
  }
}

#include <cassert>
std::shared_ptr<fi_info> parse_info(const std::string &s_, std::shared_ptr<fi_info> info_)
{
  assert(info_);
  rapidjson::Document jdoc;
  jdoc.Parse(s_.c_str());
  if ( jdoc.HasParseError() )
  {
    throw std::domain_error{std::string{"JSON parse error \""} + rapidjson::GetParseError_En(jdoc.GetParseError()) + "\" at " + std::to_string(jdoc.GetErrorOffset())};
  }
  try
  {
    return parse_info(info_, jdoc);
  }
  catch ( const std::domain_error &e )
  {
    throw std::domain_error{std::string{"JSON parse <root> "} + e.what()};
  }
}

std::shared_ptr<fi_info> parse_info(const std::string &s_)
{
  return parse_info(s_, make_fi_info());
}

/*
 * Add json specifications to configuration info
 *
 *  {
 *    "caps" : [ array of capabilities specified by strings enumerated in fabric_caps.h ]
 *    "mode" : [ array of modes specified by strings enumerated in fabric_modes.h ]
 *    "addr_format" : scalar value from strings enumerated in fabric_addr_format.h
 *    "src_addr" : string
 *    "dest_addr" : string
 *    "tx_attr" : {
 *      "caps" : (see above)
 *      "mode" : (see above)
 *      "op_flags" : [ array of flags specified by strings enumerated in fabric_op_flags.h ]
 *      "msg_order" : [ array of flags specified by strings enumerated in fabric_msg_order.h ]
 *      "comp_order: [ array of flags specified by strings enumerated in fabric_comp_order.h ]
 *      "inject_size" : unsigned
 *      "size" : unsigned
 *      "iov_limit" : unsigned
 *      "rma_iov_limit" : unsigned
 *    }
 *
 *    "rx_attr" : {
 *      "caps" : (see above)
 *      "mode" : (see above)
 *      "op_flags" : [ array of flags specified by strings enumerated in fabric_op_flags.h ]
 *      "msg_order" : [ array of flags specified by strings enumerated in fabric_msg_order.h ]
 *      "comp_order: [ array of flags specified by strings enumerated in fabric_comp_order.h ]
 *      "total_buffered_recv" : unsigned
 *      "size" : unsigned
 *      "iov_limit" : unsigned
 *    }
 *
 *    "ep_attr" : {
 *      "type" : [ array of flags specified by strings enumerated in fabric_ep_type.h ]
 *      "protocol" : scalar value from strings enumerated in fabric_protocol.h
 *      "protocol_version" : unsigned
 *      "max_msg_size" : unsigned
 *      "msg_prefix_size" : unsigned
 *      "max_order_raw_size" : unsigned
 *      "max_order_war_size" : unsigned
 *      "max_order_waw_size" : unsigned
 *      "mem_tag_format" : unsigned
 *      "tx_ctx_cnt" : unsigned
 *      "rx_ctx_cnt" : unsigned
 *      "auth_key" : string
 *    }
 *
 *    "domain_attr" : {
 *
 *               char                  *name;
 *                enum fi_threading     threading;
 *                enum fi_progress      control_progress;
 *                enum fi_progress      data_progress;
 *                enum fi_resource_mgmt resource_mgmt;
 *                enum fi_av_type       av_type;
 *                int                   mr_mode;
 *                size_t                mr_key_size" : unsigned
 *                size_t                cq_data_size" : unsigned
 *                size_t                cq_cnt" : unsigned
 *                size_t                ep_cnt" : unsigned
 *                size_t                tx_ctx_cnt" : unsigned
 *                size_t                rx_ctx_cnt" : unsigned
 *                size_t                max_ep_tx_ctx" : unsigned
 *                size_t                max_ep_rx_ctx" : unsigned
 *                size_t                max_ep_stx_ctx" : unsigned
 *                size_t                max_ep_srx_ctx" : unsigned
 *                size_t                cntr_cnt" : unsigned
 *                size_t                mr_iov_limit" : unsigned
 *                uint64_t              caps;
 *                uint64_t              mode;
 *                uint8_t               *auth_key;
 *                size_t                auth_key_size" : unsigned
 *                size_t                max_err_data" : unsigned
 *                size_t                mr_cnt" : unsigned
 *    }
 *
 *    "fabric_attr" : {
 *                char              *name;
 *                char              *prov_name;
 *                uint32_t          prov_version" : unsigned
 *                uint32_t          api_version" : unsigned
 *    }
 *  }
 *
 */
