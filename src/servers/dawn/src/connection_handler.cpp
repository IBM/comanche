#include "connection_handler.h"

namespace Dawn
{
int Connection_handler::tick() {
  using namespace Dawn::Protocol;

  _response = TICK_RESPONSE_CONTINUE;

  if (poll_completions()) {
    /* deferred unlocks */
    if (_deferred_unlock) {
      if (option_DEBUG)
        PLOG("adding action for deferred unlocking value @ %p",
             _deferred_unlock);
      add_pending_action(action_t{ACTION_RELEASE_VALUE_LOCK, _deferred_unlock});
    }
    _deferred_unlock = nullptr;
  }

  switch (_state) {
    case POST_MSG_RECV: {
      if (option_DEBUG)
        PMAJOR("Shard State: %lu %p POST_MSG_RECV", _tick_count, this);
      post_recv_buffer(allocate());
      set_state(NEW_MSG_RECV);
      break;
    }
      // case WAIT_POST_RESPONSE: {
      //   if(option_DEBUG)
      //     PMAJOR("Shard State: %u %p WAIT_POST_RESPONSE", _tick_count, this);

      //   if(_posted_send_buffer_outstanding == false) {
      //     set_state(POST_MSG_RECV);
      //     if(option_DEBUG)
      //       PMAJOR("Shared State: transitioning to POST_MSG_RECV");
      //   }
      //   break;
      // }
    case NEW_MSG_RECV: {
      if (check_for_posted_recv_complete()) {
        const auto iob = posted_recv();
        const Message *msg = static_cast<Message *>(iob->base());
        assert(msg);

        switch (msg->type_id) {
          case MSG_TYPE_IO_REQUEST: {
            if (option_DEBUG) PMAJOR("Shard: IO_REQUEST");
            _pending_msgs.push_back(iob);
            set_state(POST_MSG_RECV);
            break;
          }
          case MSG_TYPE_CLOSE_SESSION: {
            if (option_DEBUG) PMAJOR("Shard: CLOSE_SESSION!");
            free_recv_buffer();
            _response = TICK_RESPONSE_CLOSE;
            break;
          }
          case MSG_TYPE_POOL_REQUEST: {
            if (option_DEBUG) PMAJOR("Shard: POOL_REQUEST");
            _pending_msgs.push_back(iob);
            set_state(POST_MSG_RECV); /* move state to new message recv */
            break;
          }
          default:
            throw Logic_exception("unhandled message (type:%u)", msg->type_id);
        }

        _stats.recv_msg_count++;

        if (option_DEBUG)
          PMAJOR("Shard State: %lu %p WAIT_MSG_RECV complete", _tick_count,
                 this);
      }
      else {
        _stats.wait_msg_recv_misses++;
      }

      break;
    }
    case POST_RECV_VALUE: {
      if (option_DEBUG)
        PMAJOR("Shard State: %lu %p POST_RECV_VALUE", _tick_count, this);

      post_recv_value_buffer();
      set_state(WAIT_RECV_VALUE);
      break;
    }
    case WAIT_RECV_VALUE: {
      if (check_for_posted_value_complete()) {
        if (option_DEBUG) {
          PMAJOR("Shard State: %lu %p WAIT_RECV_VALUE ok", _tick_count, this);
        }

        /* add action to release lock */
        // add_pending_action(action_t{ACTION_RELEASE_VALUE_LOCK,
        //       _posted_value_buffer->base()});

        delete _posted_value_buffer; /* delete descriptor */
        _posted_value_buffer = nullptr;

        if (option_DEBUG)
          PMAJOR("Shard State: %lu %p WAIT_RECV_VALUE_COMPLETE", _tick_count,
                 this);

        set_state(POST_MSG_RECV);
        _stats.recv_msg_count++;
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
      if (option_DEBUG)
        PMAJOR("Shard State: %lu %p POST_HANDSHAKE", _tick_count, this);
      auto iob = allocate();
      post_recv_buffer(iob);

      set_state(WAIT_HANDSHAKE);
      break;
    }
    case WAIT_HANDSHAKE: {
      if (check_for_posted_recv_complete()) {
        if (option_DEBUG)
          PMAJOR("Shard State: %lu %p WAIT_HANDSHAKE complete", _tick_count,
                 this);

        auto iob = posted_recv();

        Message_handshake *msg = static_cast<Message_handshake *>(iob->base());
        if (msg->type_id == Dawn::Protocol::MSG_TYPE_HANDSHAKE) {
          auto reply_iob = allocate();
          assert(reply_iob);
          auto reply_msg =
              new (reply_iob->base()) Dawn::Protocol::Message_handshake_reply(
                  auth_id(), 1 /* seq */, max_message_size());
          /* post response */
          reply_iob->set_length(reply_msg->msg_len);
          post_send_buffer(reply_iob);

          set_state(WAIT_HANDSHAKE_RESPONSE_COMPLETION);
        }
        else {
          throw General_exception("expecting handshake request got type_id=%u",
                                  msg->type_id);
        }
      }
      break;
    }
    case WAIT_HANDSHAKE_RESPONSE_COMPLETION: {
      if (check_for_posted_send_complete()) {
        if (option_DEBUG)
          PMAJOR("Shard State: %lu %p WAIT_HANDSHAKE_RESPONSE_COMPLETION "
                 "complete.",
                 _tick_count, this);
        set_state(POST_MSG_RECV);
      }
      break;
    }

  }  // end switch
  return _response;
}

void Connection_handler::set_pending_value(void *target, size_t target_len,
                                           memory_region_t region) {
  assert(target);
  assert(target_len);

  if (option_DEBUG)
    PLOG("set_pending_value (target=%p, target_len=%lu)", target, target_len);

  auto iov = new ::iovec{target, target_len};
  auto desc = get_memory_descriptor(region);
  assert(desc);
  _posted_value_buffer =
      new buffer_t(target_len); /* allocate buffer descriptor */
  _posted_value_buffer->iov = iov;
  _posted_value_buffer->region = region;
  _posted_value_buffer->desc = desc;
  _posted_value_buffer->flags =
      Buffer_manager<Fabric_connection_base>::BUFFER_FLAGS_EXTERNAL;
  _posted_value_buffer_outstanding = true;

  post_recv_value_buffer(_posted_value_buffer);
  set_state(State::WAIT_RECV_VALUE);
}

}  // namespace Dawn
