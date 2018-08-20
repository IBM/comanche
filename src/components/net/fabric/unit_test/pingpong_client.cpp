#include "pingpong_client.h"

#include "eyecatcher.h" /* pingpong_offset */
#include "patience.h" /* open_connection_patiently */
#include "registered_memory.h"
#include "wait_poll.h"

#include <api/fabric_itf.h> /* _Fabric_client */
#include <common/errors.h> /* S_OK */
#include <common/types.h> /* status_t */
#include <gtest/gtest.h>
#include <sys/uio.h> /* iovec */
#include <cstring> /* memcpy */
#include <exception>
#include <iostream> /* cerr */
#include <string>
#include <memory> /* make_shared */
#include <vector>

void pingpong_client::check_complete_static(void *t_, void *ctxt_, ::status_t stat_)
try
{
  /* The callback context must be the object which was polling. */
  ASSERT_EQ(t_, ctxt_);
  auto rmc = static_cast<pingpong_client *>(ctxt_);
  EXPECT_TRUE(rmc);
  EXPECT_EQ(stat_, ::S_OK);
}
catch ( std::exception &e )
{
  std::cerr << "pingpong_client::" << __func__ << e.what() << "\n";
}

pingpong_client::pingpong_client(
  Component::IFabric &fabric_
  , const std::string &fabric_spec_
  , const std::string ip_address_
  , std::uint16_t port_
  , std::uint64_t remote_key_base_
  , unsigned iteration_count_
  , std::uint64_t buffer_size_
)
try
  : _cnxn(open_connection_patiently(fabric_, fabric_spec_, ip_address_, port_))
{
  registered_memory rm{*_cnxn, remote_key_base_};
  std::vector<::iovec> v{{&rm[0], buffer_size_}};
  std::vector<void *> d{{rm.desc()}};
  for ( auto i = 0U ; i != iteration_count_ ; ++i )
  {
    _cnxn->post_send(&*v.begin(), &*v.end(), &*d.begin(), this);
    wait_poll(
        *_cnxn
      , [&v, buffer_size_, this] (void *ctxt_, ::status_t stat_, std::uint64_t, std::size_t, void *) -> void
        {
          EXPECT_EQ(ctxt_, this);
          EXPECT_EQ(stat_, ::S_OK);
#if 0
          EXPECT_EQ(len_, buffer_size_);
#endif
        }
      , test_type::performance
    );
    _cnxn->post_recv(&*v.begin(), &*v.end(), &*d.begin(), this);
    wait_poll(
        *_cnxn
      , [&v, buffer_size_, this] (void *ctxt_, ::status_t stat_, std::uint64_t, std::size_t len_, void *) -> void
        {
          EXPECT_EQ(ctxt_, this);
          EXPECT_EQ(stat_, ::S_OK);
          EXPECT_EQ(len_, buffer_size_);
        }
      , test_type::performance
    );
  }
}
catch ( std::exception &e )
{
  std::cerr << __func__ << ": " << e.what() << "\n";
  throw;
}

pingpong_client::~pingpong_client()
{
}

std::size_t pingpong_client::max_message_size() const
{
  return _cnxn->max_message_size();
}
