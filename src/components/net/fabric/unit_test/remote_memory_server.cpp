#include "remote_memory_server.h"

#include "eyecatcher.h"
#include "registered_memory.h"
#include "server_connection.h"
#include "server_connection_and_memory.h"
#include "wait_poll.h"
#include <api/fabric_itf.h> /* IFabric, IFabric_server_factory */
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

void remote_memory_server::listener(Component::IFabric_server_factory &ep_, std::uint64_t remote_key_index_)
{
  auto quit = false;
  for ( ; ! quit; ++remote_key_index_ )
  {
    server_connection sc(ep_);
    /* register an RDMA memory region */
    registered_memory rm{sc.cnxn(), remote_key_index_};
    /* send the client address and key to memory */
    send_memory_info(sc.cnxn(), rm);
    /* wait for client indicate exit (by sending one byte to us) */
    try
    {
      std::vector<::iovec> v;
      ::iovec iv;
      iv.iov_base = &rm[0];
      iv.iov_len = 1;
      v.emplace_back(iv);
      sc.cnxn().post_recv(v, this);
      ::wait_poll(
        sc.cnxn()
        , [&v, &quit, &rm, this] (void *ctxt_, ::status_t stat_) -> void
          {
            ASSERT_EQ(ctxt_, this);
            ASSERT_EQ(stat_, S_OK);
            ASSERT_EQ(v[0].iov_len, 1);
            /* did client leave with the "quit byte" set to 'q'? */
            quit |= rm[0] == 'q';
          }
      );
    }
    catch ( std::exception &e )
    {
      std::cerr << "remote_memory_server::" << __func__ << ": " << e.what() << "\n";
      throw;
    }
  }
}

void remote_memory_server::listener_counted(Component::IFabric_server_factory &ep_, std::uint64_t remote_key_index_, unsigned cnxn_count_)
{
  std::vector<std::shared_ptr<server_connection_and_memory>> scrm;
  for ( auto i = 0U; i != cnxn_count_; ++i )
  {
    scrm.emplace_back(std::make_shared<server_connection_and_memory>(ep_, remote_key_index_ + i));
  }
}

remote_memory_server::remote_memory_server(
  Component::IFabric &fabric_
  , const std::string &fabric_spec_
  , std::uint16_t control_port_
  , const char *
  , std::uint64_t remote_key_base_
)
  : _ep(fabric_.open_server_factory(fabric_spec_, control_port_))
  , _th(&remote_memory_server::listener, this, std::ref(*_ep), remote_key_base_)
{
}

remote_memory_server::remote_memory_server(
Component::IFabric &fabric_
  , const std::string &fabric_spec_
  , std::uint16_t control_port_
  , const char *
  , std::uint64_t remote_key_base_
  , unsigned cnxn_limit_
)
  : _ep(fabric_.open_server_factory(fabric_spec_, control_port_))
  , _th(&remote_memory_server::listener_counted, this, std::ref(*_ep), remote_key_base_, cnxn_limit_)
{
}

remote_memory_server::~remote_memory_server()
try
{
  _th.join();
}
catch ( std::exception &e )
{
  std::cerr << __func__ << " exception " << e.what() << eyecatcher << std::endl;
}
