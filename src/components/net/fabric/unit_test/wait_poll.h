#ifndef _TEST_WAIT_POLL_H_
#define _TEST_WAIT_POLL_H_

#include <api/fabric_itf.h> /* IFabric_communicator */
#include <common/types.h> /* status_t */
#include <functional> /* function */

void wait_poll(Component::IFabric_communicator &comm, std::function<void(void *context, ::status_t)> cb);

#endif
