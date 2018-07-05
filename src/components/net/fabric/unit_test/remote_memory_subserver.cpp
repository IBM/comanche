#include "remote_memory_subserver.h"

#include "server_grouped_connection.h"
#include <common/errors.h> /* S_OK */
#include <gtest/gtest.h>

remote_memory_subserver::remote_memory_subserver(server_grouped_connection &parent_)
  : _parent(parent_)
  , _cnxn(_parent.allocate_group())
  , _rm_out{_parent.cnxn()}
  , _rm_in{_parent.cnxn()}
{
}

void remote_memory_subserver::check_complete_static(void *rmc_, ::status_t stat_)
{
  auto rmc = static_cast<remote_memory_subserver *>(rmc_);
  ASSERT_TRUE(rmc);
  rmc->check_complete(stat_);
}

void remote_memory_subserver::check_complete_static_2(void *t_, void *rmc_, ::status_t stat_)
{
  /* The callback context must be the object which was polling. */
  ASSERT_EQ(t_, rmc_);
  check_complete_static(rmc_, stat_);
}

void remote_memory_subserver::check_complete(::status_t stat_)
{
  ASSERT_EQ(stat_, S_OK);
}
