#include "server_connection_and_memory.h"

#include "wait_poll.h"
#include <common/errors.h> /* S_OK */
#include <gtest/gtest.h>
#include <sys/uio.h> /* iovec */
#include <vector>

server_connection_and_memory::server_connection_and_memory(Component::IFabric_server_factory &ep_, std::uint64_t remote_key_)
  : server_connection(ep_)
  , registered_memory(cnxn(), remote_key_)
{
  /* send the address, and the key to memory */
  send_memory_info(cnxn(), *this);
}

server_connection_and_memory::~server_connection_and_memory()
try
{
  std::vector<::iovec> v;
  ::iovec iv;
  iv.iov_base = &((*this)[0]);
  iv.iov_len = 1;
  v.emplace_back(iv);
  cnxn().post_recv(v, this);
  wait_poll(
    cnxn()
    , [this] (void *ctxt_, ::status_t stat_, std::uint64_t, std::size_t len_, void *) -> void
      {
        ASSERT_EQ(ctxt_, this);
        ASSERT_EQ(stat_, S_OK);
        ASSERT_EQ(len_, 1);
      }
  );
}
catch ( std::exception &e )
{
  std::cerr << "(destructor) " << __func__ << ": " << e.what() << "\n";
}
