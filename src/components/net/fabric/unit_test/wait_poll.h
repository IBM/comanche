#ifndef _TEST_WAIT_POLL_H_
#define _TEST_WAIT_POLL_H_

#include <api/fabric_itf.h> /* IFabric_communicator */
#include <common/types.h> /* status_t */
#include <cstddef> /* size_t */
#include <cstdint> /* uint64_t */
#include <functional> /* function */

enum class test_type
{
  performance /* omit extra paths */
  , function /* add extra paths (block for event, deferred handling) */
};

void wait_poll(
  Component::IFabric_communicator &comm
  , std::function<void(
    void *context
    , ::status_t
    , std::uint64_t completion_flags
    , std::size_t len
    , void *error_data
  )> cb
  , test_type test_type_ = test_type::function
);

#endif
