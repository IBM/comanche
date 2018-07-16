#include "remote_memory_client.h"

#include "eyecatcher.h" /* remote_memory_offset */
#include "patience.h" /* open_connection_patiently */
#include "registered_memory.h"
#include "wait_poll.h"

#include <api/fabric_itf.h> /* _Fabric_client */
#include <common/errors.h> /* S_OK */
#include <common/types.h> /* status_t */
#include <gtest/gtest.h>
#include <sys/uio.h> /* iovec */
#include <boost/io/ios_state.hpp>
#include <algorithm> /* copy */
#include <cstring> /* memcpy */
#include <exception>
#include <iomanip> /* hex */
#include <iostream> /* cerr */
#include <string>
#include <memory> /* make_shared */
#include <vector>

void remote_memory_client::check_complete_static(void *t_, void *rmc_, ::status_t stat_)
try
{
  /* The callback context must be the object which was polling. */
  ASSERT_EQ(t_, rmc_);
  auto rmc = static_cast<remote_memory_client *>(rmc_);
  ASSERT_TRUE(rmc);
  rmc->check_complete(stat_);
}
catch ( std::exception &e )
{
  std::cerr << "remote_memory_client::" << __func__ << e.what() << "\n";
}

void remote_memory_client::check_complete(::status_t stat_)
{
  _last_stat = stat_;
}

void remote_memory_client::do_quit()
{
  _quit_flag = 'q';
}

remote_memory_client::remote_memory_client(Component::IFabric &fabric_, const std::string &fabric_spec_, const std::string ip_address_, std::uint16_t port_, std::uint64_t remote_key_base_)
try
  : _cnxn(open_connection_patiently(fabric_, fabric_spec_, ip_address_, port_))
  , _rm_out{std::make_shared<registered_memory>(*_cnxn, remote_key_base_ * 2U)}
  , _rm_in{std::make_shared<registered_memory>(*_cnxn, remote_key_base_ * 2U + 1)}
  , _vaddr{}
  , _key{}
  , _quit_flag('n')
  , _last_stat(::E_FAIL)
{
  std::vector<::iovec> v;
  ::iovec iv;
  iv.iov_base = &rm_out()[0];
  iv.iov_len = (sizeof _vaddr) + (sizeof _key);
  v.emplace_back(iv);
  _cnxn->post_recv(v, this);
  wait_poll(
      *_cnxn
    , [&v, this] (void *ctxt, ::status_t st) -> void
      {
        ASSERT_EQ(ctxt, this);
        ASSERT_EQ(st, ::S_OK);
        ASSERT_EQ(v[0].iov_len, (sizeof _vaddr) + sizeof( _key));
        std::memcpy(&_vaddr, &rm_out()[0], sizeof _vaddr);
        std::memcpy(&_key, &rm_out()[sizeof _vaddr], sizeof _key);
      }
  );
  boost::io::ios_flags_saver sv(std::cerr);
  std::cerr << "Client: remote memory addr " << reinterpret_cast<void*>(_vaddr) << " key " << std::hex << _key << std::endl;
}
catch ( std::exception &e )
{
  std::cerr << __func__ << ": " << e.what() << "\n";
  throw;
}

void remote_memory_client::send_disconnect(Component::IFabric_communicator &cnxn_, registered_memory &rm_, char quit_flag_)
{
  send_msg(cnxn_, rm_, &quit_flag_, sizeof quit_flag_);
}

remote_memory_client::~remote_memory_client()
try
{
  if ( _cnxn )
  {
    send_disconnect(cnxn(), rm_out(), _quit_flag);
  }
}
catch ( std::exception &e )
{
  std::cerr << __func__ << " exception " << e.what() << eyecatcher << std::endl;
}

void remote_memory_client::write(const std::string &msg_)
{
  std::copy(msg_.begin(), msg_.end(), &rm_out()[0]);
  std::vector<::iovec> buffers(1);
  {
    buffers[0].iov_base = &rm_out()[0];
    buffers[0].iov_len = msg_.size();
    _cnxn->post_write(buffers, _vaddr + remote_memory_offset, _key, this);
  }
  wait_poll(
    *_cnxn
    , [this] (void *rmc_, ::status_t stat_) { check_complete_static(this, rmc_, stat_); }
  );
  EXPECT_EQ(_last_stat, ::S_OK);
  if ( _last_stat != ::S_OK )
  {
    std::cerr << "remote_memory_client::" << __func__ << ": " << _last_stat << "\n";
  }
}

void remote_memory_client::read_verify(const std::string &msg_)
{
  std::vector<::iovec> buffers(1);
  {
    buffers[0].iov_base = &rm_in()[0];
    buffers[0].iov_len = msg_.size();
    _cnxn->post_read(buffers, _vaddr + remote_memory_offset, _key, this);
  }
  wait_poll(
    *_cnxn
    , [this] (void *rmc_, ::status_t stat_) { check_complete_static(this, rmc_, stat_); }
  );
  EXPECT_EQ(_last_stat, ::S_OK);
  if ( _last_stat != ::S_OK )
  {
    std::cerr << "remote_memory_client::" << __func__ << ": " << _last_stat << "\n";
  }
  std::string remote_msg(&rm_in()[0], &rm_in()[0] + msg_.size());
  EXPECT_EQ(msg_, remote_msg);
}
