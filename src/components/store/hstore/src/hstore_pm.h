#ifndef COMANCHE_HSTORE_PM_H
#define COMANCHE_HSTORE_PM_H

#include <string>

class pool_path;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

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

  virtual auto pool_create(
    const pool_path &path_
    , std::size_t size_
    , std::size_t expected_obj_count_
  ) -> std::unique_ptr<tracked_pool> = 0;

  virtual auto pool_open(
    const pool_path &path_
  ) -> std::unique_ptr<tracked_pool> = 0;

  virtual void pool_delete(const pool_path &path) = 0;
};
#pragma GCC diagnostic pop

#endif
