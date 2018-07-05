#include "remote_memory_subclient.h"

#include "eyecatcher.h" /* remote_memory_offset */
#include "remote_memory_client_grouped.h"
#include "wait_poll.h"
#include <api/fabric_itf.h> /* IFabric, IFabric_commuicator */
#include <common/errors.h> /* S_OK */
#include <gtest/gtest.h>
#include <sys/uio.h> /* iovec */

#include <string>
#include <memory> /* shared_ptr */
#include <algorithm> /* copy */
#include <vector>

void remote_memory_subclient::check_complete_static(void *rmc_, ::status_t stat_)
{
  auto rmc = static_cast<remote_memory_subclient *>(rmc_);
  ASSERT_TRUE(rmc);
  rmc->check_complete(stat_);
}
void remote_memory_subclient::check_complete_static_2(void *t_, void *rmc_, ::status_t stat_)
{
  /* The callback context must be the object which was polling. */
  ASSERT_EQ(t_, rmc_);
  check_complete_static(rmc_, stat_);
}

void remote_memory_subclient::check_complete(::status_t stat_)
{
  ASSERT_EQ(stat_, S_OK);
}

remote_memory_subclient::remote_memory_subclient(remote_memory_client_grouped &parent_)
  : _parent(parent_)
  , _cnxn(_parent.allocate_group())
  , _rm_out{_parent.cnxn()}
  , _rm_in{_parent.cnxn()}
{
}

void remote_memory_subclient::write(const std::string &msg_)
{
  std::copy(msg_.begin(), msg_.end(), &rm_out()[0]);
  std::vector<::iovec> buffers(1);
  {
    buffers[0].iov_base = &rm_out()[0];
    buffers[0].iov_len = msg_.size();
    _cnxn->post_write(buffers, _parent.vaddr() + remote_memory_offset, _parent.key(), this);
  }
  wait_poll(
    *_cnxn
    , [this] (void *rmc_, ::status_t stat_) { check_complete_static_2(this, rmc_, stat_); } /* WAS: check_complete_static */
  );
}

void remote_memory_subclient::read_verify(const std::string &msg_)
{
  std::vector<::iovec> buffers(1);
  {
    buffers[0].iov_base = &rm_in()[0];
    buffers[0].iov_len = msg_.size();
    _cnxn->post_read(buffers, _parent.vaddr() + remote_memory_offset, _parent.key(), this);
  }
  wait_poll(
    *_cnxn
    , [this] (void *rmc_, ::status_t stat_) { check_complete_static_2(this, rmc_, stat_); } /* WAS: check_complete_static */
  );
  std::string remote_msg(&rm_in()[0], &rm_in()[0] + msg_.size());
  ASSERT_EQ(msg_, remote_msg);
}
