#ifndef _TEST_PINGPONG_BUFFER_STATE_H_
#define _TEST_PINGPONG_BUFFER_STATE_H_

#include "delete_copy.h"
#include "registered_memory.h"
#include <sys/uio.h> /* iovec */
#include <cstddef> /* size_t */
#include <cstdint> /* uint64_t */
#include <vector>

namespace Component
{
  class IFabric_connection;
}

struct buffer_state
{
  registered_memory _rm;
  std::vector<::iovec> v;
  std::vector<void *> d;
  explicit buffer_state(Component::IFabric_connection &cnxn, std::size_t size, std::uint64_t remote_key, std::size_t msg_size);
  DELETE_COPY(buffer_state);
};

#endif
