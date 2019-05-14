/*
   Copyright [2019] [IBM Corporation]
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

#ifndef _REGION_MOFIFICATIONS_
#define _REGION_MOFIFICATIONS_

#include <common/types.h> /* offset_t */

#include <boost/icl/interval_set.hpp>
#include <boost/icl/split_interval_set.hpp>
#include <boost/icl/right_open_interval.hpp>

namespace nupm
{
  class Region_modifications
    : private boost::icl::interval_set<const char *, std::less, boost::icl::right_open_interval<const char *>>
  {
    using base = boost::icl::interval_set<const char *, std::less, boost::icl::right_open_interval<const char *>>;

  public:
    static constexpr bool preserves_splits = false;
    void add(const void *p, std::size_t s)
    {
      auto pc = static_cast<const char *>(p);
      insert(boost::icl::right_open_interval<const char *>(pc, pc + s));
    }

    template <typename T>
      void add(const T &v)
      {
        this->add(&v, sizeof v);
      }

    void coalesce() {}

    /* get_region - iterate space of ALL modified regions collected - return region size or zero for end */

    using base::begin;
    using base::end;

    /* clear region tracking */

    using base::clear;
  };

  /* An implementation which retain splits until coalesce */
  class Region_modifications_split
    : private boost::icl::split_interval_set<const char *, std::less, boost::icl::right_open_interval<const char *>>
  {
    using base = boost::icl::split_interval_set<const char *, std::less, boost::icl::right_open_interval<const char *>>;

  public:
    static constexpr bool preserves_splits = true;
    void add(const void *p, std::size_t s)
    {
      auto pc = static_cast<const char *>(p);
      insert(boost::icl::right_open_interval<const char *>(pc, pc + s));
    }

    template <typename T>
      void add(const T &v)
      {
        this->add(&v, sizeof v);
      }

    void coalesce()
    {
      this->assign(
        boost::icl::interval_set<const char *, std::less, boost::icl::right_open_interval<const char *>>(*this)
      );
    }

    /* get_region - iterate space of ALL modified regions collected - return region size or zero for end */

    using base::begin;
    using base::end;

    /* clear region tracking */

    using base::clear;
  };

#if 1
  thread_local Region_modifications tls_modifications;
#else
  /* An implementation which is probably less efficient that Region_modifications
   * unless
   *  (1) splits must be retained until an explicit coalesce, or
   *  (2) an erase function is added to remove a specific region.
   */
  thread_local Region_modifications_split tls_modifications;
#endif

  bool region_tracker_preserves_splits()
  {
    return tls_modifications.preserves_splits;
  }

  /* add region */
  void region_tracker_add(void * p, size_t p_len)
  {
    tls_modifications.add(p, p_len);
  }

  void region_tracker_coalesce_across_TLS()
  {
    tls_modifications.coalesce();
  }

  /* offset_t is uint64_t. Presume that it is meant to be an index into the
   * interval set.
   * If examining all intervals, better to use
   *   for ( auto e : tls_modifications ) ...
   */
  std::size_t region_tracker_get_region(offset_t offset, const void*& p)
  {
    auto it = tls_modifications.begin();
    for ( ; offset && it != tls_modifications.end(); --offset, ++it )
    {
    }
    if ( it == tls_modifications.end() )
    {
      return 0;
    }
    auto pc = static_cast<const char *>(it->lower());
    p = pc;
    return std::size_t(static_cast<const char *>(it->upper()) - pc);
  }

  void region_tracker_clear()
  {
    tls_modifications.clear();
  }
}

#endif
