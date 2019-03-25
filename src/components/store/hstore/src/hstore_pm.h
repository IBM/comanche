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


#ifndef COMANCHE_HSTORE_PM_H
#define COMANCHE_HSTORE_PM_H

#include <string>

class pool_path;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

enum class pool_ec
{
  pool_fail,
  pool_unsupported_mode,
  region_fail,
  region_fail_general_exception,
  region_fail_api_exception,
};

class pool_category
  : public std::error_category
{
  const char* name() const noexcept override { return "pool_category"; }
  std::string message( int condition ) const noexcept override
  {
    switch ( condition )
    {
    case int(pool_ec::pool_fail):
      return "default pool failure";
    case int(pool_ec::pool_unsupported_mode):
      return "pool unsupported flags";
    case int(pool_ec::region_fail):
      return "region-backed pool failure";
    case int(pool_ec::region_fail_general_exception):
      return "region-backed pool failure (General_exception)";
    case int(pool_ec::region_fail_api_exception):
      return "region-backed pool failure (API_exception)";
    default:
      return "unknown pool failure";
    }
  }
};

namespace
{
  pool_category pool_error_category;
}

class pool_error
  : public std::error_condition
{
  std::string _msg;
public:
  pool_error(const std::string &msg_, pool_ec val_)
    : std::error_condition(int(val_), pool_error_category)
    , _msg(msg_) 
  {}

};

class pool_manager
{
  bool _debug;
public:
  pool_manager(bool debug_) : _debug(debug_) {}
  virtual ~pool_manager() {}
  bool debug() const { return _debug; }

  virtual auto pool_create_check(const std::size_t size_) -> status_t = 0;

  virtual void pool_close_check(const std::string &) { }  // = 0;

  virtual status_t pool_get_regions(void *, std::vector<::iovec>&) = 0;

  /*
   * throws pool_error if create_region fails
   */
  virtual auto pool_create(
    const pool_path &path_
    , std::size_t size_
    , int flags_
    , std::size_t expected_obj_count_
  ) -> std::unique_ptr<tracked_pool> = 0;

  virtual auto pool_open(
    const pool_path &path_
    , int flags_
  ) -> std::unique_ptr<tracked_pool> = 0;

  virtual void pool_delete(const pool_path &path) = 0;
};
#pragma GCC diagnostic pop

#endif
