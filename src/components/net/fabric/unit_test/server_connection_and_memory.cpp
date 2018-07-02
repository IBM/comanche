#include "server_connection_and_memory.h"

#include "wait_poll.h"
#include <common/errors.h> /* S_OK */
#include <gtest/gtest.h>
#include <sys/uio.h> /* iovec */
#include <vector>

server_connection_and_memory::server_connection_and_memory(Component::IFabric_server_factory &ep_)
  : server_connection(ep_)
  , registered_memory(cnxn())
{
  /* send the address, and the key to memory */
  send_memory_info(cnxn(), *this);
}

server_connection_and_memory::~server_connection_and_memory()
{
  std::vector<::iovec> v;
  ::iovec iv;
  iv.iov_base = &((*this)[0]);
  iv.iov_len = 1;
  v.emplace_back(iv);
  cnxn().post_recv(v, this);
  wait_poll(
    cnxn()
    , [&v, this] (void *ctxt, ::status_t st) -> void
      {
        ASSERT_EQ(ctxt, this);
        ASSERT_EQ(st, S_OK);
        ASSERT_EQ(v[0].iov_len, 1);
      }
  );
}
