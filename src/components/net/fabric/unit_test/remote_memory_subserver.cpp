#include "remote_memory_subserver.h"

#include "server_grouped_connection.h"
#include <common/errors.h> /* S_OK */
#include <gtest/gtest.h>

remote_memory_subserver::remote_memory_subserver(
  server_grouped_connection &parent_
  , std::size_t memory_size_
  , std::uint64_t remote_key_index_
  )
  : _parent(parent_)
  , _cnxn(_parent.allocate_group())
  , _rm_out{_parent.cnxn(), memory_size_, remote_key_index_ * 2U}
  , _rm_in{_parent.cnxn(), memory_size_, remote_key_index_ * 2U + 1}
{
}
