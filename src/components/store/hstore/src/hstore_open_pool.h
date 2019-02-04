#ifndef COMANCHE_HSTORE_OPEN_POOL_H
#define COMANCHE_HSTORE_OPEN_POOL_H

#include <string>

class pool_path
{
  std::string _dir;
  std::string _name;

public:
  explicit pool_path(
    const std::string &dir_
    , const std::string &name_
  )
    : _dir(dir_)
    , _name(name_)
  {}

  pool_path()
    : pool_path("/", "")
  {}

  std::string str() const
  {
    /* ERROR: fails if _dir is the empty string */
    return _dir + ( _dir[_dir.length()-1] != '/' ? "/" : "") + _name;
  }
#if 1
  /* delete_pool only */
  const std::string &dir() const noexcept { return _dir; }
  const std::string &name() const noexcept { return _name; }
#endif
};

class tracked_pool
	: protected pool_path
	{
	public:
		explicit tracked_pool(const pool_path &p_)
			: pool_path(p_)
		{}
		virtual ~tracked_pool() {}
		pool_path path() const { return *this; }
	};

template <typename Handle>
	class open_pool
		: public tracked_pool
	{
		Handle _pop;
	public:
		explicit open_pool(
			const pool_path &path_
			, Handle &&pop_
		)
			: tracked_pool(path_)
			, _pop(std::move(pop_))
		{}
		open_pool(const open_pool &) = delete;
		open_pool& operator=(const open_pool &) = delete;
#if 1
		/* get_pool_regions only */
		auto *pool() const { return _pop.get(); }
#endif
	};

#endif
