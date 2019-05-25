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
#include "connection_handler.h"

namespace Dawn
{
int Connection_handler::tick()
{
  using namespace Dawn::Protocol;

  auto response = TICK_RESPONSE_CONTINUE;
  _tick_count++;
  
#if 0
  auto now = rdtsc();
  
  /* output IOPS */
  if (_stats.next_stamp == 0) {
    _stats.next_stamp = now + (_freq_mhz * 1000000.0);
    _stats.last_count = _stats.response_count;
  }
  if (now >= (_stats.next_stamp)) {
    PMAJOR("(%p) IOPS: %lu /s", this, _stats.response_count - _stats.last_count);
    _stats.next_stamp = now + (_freq_mhz * 1000000.0);
    _stats.last_count = _stats.response_count;
  }
#endif

  if (check_network_completions() == Fabric_connection_base::Completion_state::CLIENT_DISCONNECT) {
    PMAJOR("Client disconnected.");
    return Dawn::Connection_handler::TICK_RESPONSE_CLOSE;
  }

  switch (_state) {

  case POST_MSG_RECV: { /*< post buffer to receive new message */
    if (option_DEBUG > 2)
      PMAJOR("Shard State: %lu %p POST_MSG_RECV", _tick_count, this);
    post_recv_buffer(allocate());
    set_state(WAIT_NEW_MSG_RECV);
    stall(); /* we can stall because we know that there will be a little while before the next request */
    break;
  }      
  case WAIT_NEW_MSG_RECV: {
      
    if (check_for_posted_recv_complete()) { /*< check for recv completion */
        
      const auto iob = posted_recv();
      assert(iob);
        
      const Message *msg = Dawn::Protocol::message_cast(iob->base());
      assert(msg);
        
      switch (msg->type_id) {
      case MSG_TYPE_IO_REQUEST: {
        if (option_DEBUG > 2) PMAJOR("Shard: IO_REQUEST");
        _pending_msgs.push_back(iob);
        set_state(POST_MSG_RECV);
        break;
      }
      case MSG_TYPE_CLOSE_SESSION: {
        if (option_DEBUG > 2) PMAJOR("Shard: CLOSE_SESSION!");
        free_recv_buffer();
        response = TICK_RESPONSE_CLOSE;
        break;
      }
      case MSG_TYPE_POOL_REQUEST: {
        if (option_DEBUG > 2) PMAJOR("Shard: POOL_REQUEST");
        _pending_msgs.push_back(iob);
        set_state(POST_MSG_RECV); /* move state to new message recv */
        break;
      }
      case MSG_TYPE_INFO_REQUEST: {
        if (option_DEBUG > 2) PMAJOR("Shard: INFO_REQUEST");
        _pending_msgs.push_back(iob);
        set_state(POST_MSG_RECV); /* move state to new message recv */
        break;
      }
      default:
        throw Logic_exception("unhandled message (type:%x)", msg->type_id);
      }

      _stats.recv_msg_count++;

      if (option_DEBUG > 2)
        PMAJOR("Shard State: %lu %p WAIT_MSG_RECV complete", _tick_count,
               this);
    }
    else {
      _stats.wait_msg_recv_misses++;
    }

    break;
  }
  case WAIT_SEND_VALUE: 
  case WAIT_RECV_VALUE: {
      
    if (check_for_posted_value_complete()) {

      if (option_DEBUG > 2) {
        PMAJOR("Shard State: %lu %p WAIT_SEND/RECV_VALUE ok", _tick_count, this);
      }

      assert(_posted_value_buffer);

      /* add action to release lock */
      add_pending_action(action_t{ACTION_RELEASE_VALUE_LOCK,_posted_value_buffer->base()});

      delete _posted_value_buffer; /* delete descriptor */
      _posted_value_buffer = nullptr;

      if (option_DEBUG > 2)
        PMAJOR("Shard State: %lu %p WAIT_RECV_VALUE_COMPLETE", _tick_count,
               this);

      set_state(POST_MSG_RECV);
      if(_state == WAIT_RECV_VALUE)
        _stats.recv_msg_count++;
      else
        _stats.send_msg_count++;
    }
    else
      _stats.wait_recv_value_misses++;
    break;
  }
  case INITIALIZE: {
    set_state(POST_HANDSHAKE);
    break;
  }
  case POST_HANDSHAKE: {
    if (option_DEBUG > 2)
      PMAJOR("Shard State: %lu %p POST_HANDSHAKE", _tick_count, this);
    auto iob = allocate();
    post_recv_buffer(iob);
    set_state(WAIT_HANDSHAKE);
    break;
  }
  case WAIT_HANDSHAKE: {
        
    if (check_for_posted_recv_complete()) {
      if (option_DEBUG > 2)
        PMAJOR("Shard State: %lu %p WAIT_HANDSHAKE complete", _tick_count,
               this);

      const auto iob = posted_recv();
      assert(iob);

      auto msg =
        Dawn::Protocol::message_cast(iob->base())->ptr_cast<Message_handshake>();

      auto reply_iob = allocate();
      assert(reply_iob);
      auto reply_msg =
        new (reply_iob->base()) Dawn::Protocol::Message_handshake_reply(auth_id(),
                                                                        1 /* seq */,
                                                                        max_message_size(),
                                                                        (uint64_t) this);
      /* post response */
      reply_iob->set_length(reply_msg->msg_len);
      post_send_buffer(reply_iob);

      set_state(WAIT_HANDSHAKE_RESPONSE_COMPLETION);
    }
    break;
  }
  case WAIT_HANDSHAKE_RESPONSE_COMPLETION: {

    if (check_for_posted_send_complete()) {
      if (option_DEBUG > 2)
        PMAJOR("Shard State: %lu %p WAIT_HANDSHAKE_RESPONSE_COMPLETION "
               "complete.",
               _tick_count, this);
      set_state(POST_MSG_RECV);
    }
    break;
  }

  }  // end switch
  return response;
}

void Connection_handler::set_pending_value(void *          target,
                                           size_t          target_len,
                                           memory_region_t region)
{
  assert(target);
  assert(target_len);

  if (option_DEBUG > 2)
    PLOG("set_pending_value (target=%p, target_len=%lu, handle=%p)",
         target, target_len, region);

  auto iov  = new ::iovec{target, target_len};
  auto desc = get_memory_descriptor(region);
  assert(desc);
  _posted_value_buffer = new buffer_t(target_len); /* allocate buffer descriptor */
  _posted_value_buffer->iov    = iov;
  _posted_value_buffer->region = region;
  _posted_value_buffer->desc   = desc;
  _posted_value_buffer->flags =
    Buffer_manager<Fabric_connection_base>::BUFFER_FLAGS_EXTERNAL;
  _posted_value_buffer_outstanding = true;

  post_recv_value_buffer(_posted_value_buffer);
  set_state(State::WAIT_RECV_VALUE);
}

void Connection_handler::set_pending_send_value()
{
  /*
    assert(target);
    assert(target_len);

    if (option_DEBUG > 2)
    PLOG("set_pending_send_value (target=%p, target_len=%lu, handle=%p)",
    target, target_len, region);

    auto iov  = new ::iovec{target, target_len};
    auto desc = get_memory_descriptor(region);
    assert(desc);
    _posted_value_buffer = new buffer_t(target_len);  allocate buffer descriptor 
    _posted_value_buffer->iov    = iov;
    _posted_value_buffer->region = region;
    _posted_value_buffer->desc   = desc;
    _posted_value_buffer->flags =
    Buffer_manager<Fabric_connection_base>::BUFFER_FLAGS_EXTERNAL;
    _posted_value_buffer_outstanding = true;

    pos t_recv_value_buffer(_posted_value_buffer);
  */
  set_state(State::WAIT_SEND_VALUE);
}


}  // namespace Dawn
