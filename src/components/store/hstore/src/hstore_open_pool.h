#ifndef COMANCHE_HSTORE_OPEN_POOL_H
#define COMANCHE_HSTORE_OPEN_POOL_H

class open_pool
{
  std::string               _dir;
  std::string               _name;
  open_pool_handle          _pop;
public:
  explicit open_pool(
    const std::string &dir_
    , const std::string &name_
    , open_pool_handle &&pop_
  )
    : _dir(dir_)
    , _name(name_)
    , _pop(std::move(pop_))
  {}
  open_pool(const open_pool &) = delete;
  open_pool& operator=(const open_pool &) = delete;
  virtual ~open_pool() {}

#if 1
  /* get_pool_regions only */
  auto *pool() const { return _pop.get(); }
#endif
#if 1
  /* delete_pool only */
  const std::string &dir() const noexcept { return _dir; }
  const std::string &name() const noexcept { return _name; }
#endif
};

#endif
