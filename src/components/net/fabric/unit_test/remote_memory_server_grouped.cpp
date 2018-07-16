#include "remote_memory_server_grouped.h"

#include "eyecatcher.h"
#include "registered_memory.h"
#include "server_grouped_connection.h"
#include "wait_poll.h"
#include <api/fabric_itf.h> /* IFabric, IFabric_server_grouped_factory */
#include <common/errors.h> /* S_OK */
#include <gtest/gtest.h>
#include <sys/uio.h> /* iovec */
#include <exception>
#include <functional> /* ref */
#include <iostream> /* cerr */
#include <string>
#include <memory> /* make_shared, shared_ptr */
#include <thread>
#include <vector>

void remote_memory_server_grouped::listener(
  Component::IFabric_server_grouped_factory &ep_
  , std::uint64_t remote_key_index_
)
{
  auto quit = false;
  for ( ; ! quit ; ++remote_key_index_ )
  {
    /* Get a client to work with */
    /* Get a client to work with */
    server_grouped_connection sc(ep_);
    /* register an RDMA memory region */
    registered_memory rm{sc.cnxn(), remote_key_index_};
    /* send the client address and key to memory */
    auto &cnxn = sc.comm();
    send_memory_info(cnxn, rm);
    /* wait for client indicate exit (by sending one byte to us) */
    try
    {
      std::vector<::iovec> v;
      ::iovec iv;
      iv.iov_base = &rm[0];
      iv.iov_len = 1;
      v.emplace_back(iv);
      cnxn.post_recv(v, this);
      wait_poll(
        cnxn
        , [&v, &quit, &rm, this] (void *ctxt, ::status_t st) -> void
          {
            ASSERT_EQ(ctxt, this);
            ASSERT_EQ(st, S_OK);
            ASSERT_EQ(v[0].iov_len, 1);
            /* did client leave with the "quit byte" set to 'q'? */
            quit |= rm[0] == 'q';
          }
      );
    }
    catch ( std::exception &e )
    {
      std::cerr << "remote_memory_server_grouped::" << __func__ << ": " << e.what() << "\n";
      throw;
    }
  }
}

remote_memory_server_grouped::remote_memory_server_grouped(
  Component::IFabric &fabric_
  , const std::string &fabric_spec_
  , std::uint16_t control_port_
  , std::uint64_t remote_key_base_
)
  : _ep(fabric_.open_server_grouped_factory(fabric_spec_, control_port_))
  , _th(&remote_memory_server_grouped::listener, this, std::ref(*_ep), remote_key_base_)
{}

remote_memory_server_grouped::~remote_memory_server_grouped()
try
{
  _th.join();
}
catch ( std::exception &e )
{
  std::cerr << __func__ << " exception " << e.what() << eyecatcher << std::endl;
}
