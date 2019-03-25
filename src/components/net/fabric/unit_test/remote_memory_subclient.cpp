/*
   Copyright [2017-2019] [IBM Corporation]
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
       http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
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

void remote_memory_subclient::check_complete_static(void *t_, void *ctxt_, ::status_t stat_, std::size_t len_)
{
  /* The callback context must be the object which was polling. */
  ASSERT_EQ(t_, ctxt_);
  auto rmc = static_cast<remote_memory_subclient *>(ctxt_);
  ASSERT_TRUE(rmc);
  rmc->check_complete(stat_, len_);
}

void remote_memory_subclient::check_complete(::status_t stat_, std::size_t)
{
  EXPECT_EQ(stat_, S_OK);
}

remote_memory_subclient::remote_memory_subclient(
  remote_memory_client_grouped &parent_
  , std::size_t memory_size_
  , std::uint64_t remote_key_index_
  )
  : _parent(parent_)
  , _cnxn(_parent.allocate_group())
  , _rm_out{_parent.cnxn(), memory_size_, remote_key_index_ * 2U}
  , _rm_in{_parent.cnxn(), memory_size_, remote_key_index_ * 2U + 1U}
{
}

void remote_memory_subclient::write(const std::string &msg_)
try
{
  std::copy(msg_.begin(), msg_.end(), &rm_out()[0]);
  std::vector<::iovec> buffers(1);
  {
    buffers[0].iov_base = &rm_out()[0];
    buffers[0].iov_len = msg_.size();
    _cnxn->post_write(buffers, _parent.vaddr() + remote_memory_offset, _parent.key(), this);
  }
  ::wait_poll(
    *_cnxn
    , [this] (void *rmc_, ::status_t stat_, std::uint64_t, std::size_t len_, void *)
      {
        check_complete_static(this, rmc_, stat_, len_);
      }
  );
}
catch ( std::exception &e )
{
  std::cerr << "remote_memory_subclient::" << __func__ << ": " << e.what() << "\n";
  throw;
}

void remote_memory_subclient::read_verify(const std::string &msg_)
try
{
  std::vector<::iovec> buffers(1);
  {
    buffers[0].iov_base = &rm_in()[0];
    buffers[0].iov_len = msg_.size();
    _cnxn->post_read(buffers, _parent.vaddr() + remote_memory_offset, _parent.key(), this);
  }
  ::wait_poll(
    *_cnxn
    , [this] (void *rmc_, ::status_t stat_, std::uint64_t, std::size_t len_, void *)
      {
        check_complete_static(this, rmc_, stat_, len_);
      }
  );
  std::string remote_msg(&rm_in()[0], &rm_in()[0] + msg_.size());
  EXPECT_EQ(msg_, remote_msg);
}
catch ( std::exception &e )
{
  std::cerr << "remote_memory_subclient::" << __func__ << ": " << e.what() << "\n";
  throw;
}
