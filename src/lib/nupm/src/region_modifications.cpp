#include "region_modifications.h"

namespace nupm
{
  thread_local bool tracker_active = false;
  thread_local Region_modifications tls_modifications;

  /* add region */
  void region_tracker_add(void * p, size_t p_len, char tag)
  {
    if ( tracker_active )
    {
      tls_modifications.add(p, p_len, tag);
    }
  }
  
  void region_tracker_coalesce_across_TLS()
  { 
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
