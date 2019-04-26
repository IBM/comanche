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
#ifndef _TEST_WAIT_POLL_H_
#define _TEST_WAIT_POLL_H_

#include <api/fabric_itf.h> /* IFabric_communicator */
#include <common/types.h> /* status_t */
#include <cstddef> /* size_t */
#include <cstdint> /* uint64_t */
#include <functional> /* function */

enum class test_type
{
  performance /* omit extra paths, busy-poll for completion */
  , function /* add extra paths (block for event, deferred handling) */
};

/*
 * returns: number of polls (including the successful poll)
 */
unsigned wait_poll(
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
