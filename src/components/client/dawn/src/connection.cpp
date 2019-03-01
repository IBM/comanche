#include <city.h>
#include <common/cycles.h>
#include <common/utils.h>
#include <unistd.h>

#include "connection.h"
#include "protocol.h"

using namespace Component;

namespace Dawn
{
namespace Client
{
Connection_handler::Connection_handler(Connection_base::Transport* connection)
    : Connection_base(connection)
{
  char* env = getenv("SHORT_CIRCUIT_BACKEND");
  if (env && env[0] == '1') {
    _options.short_circuit_backend = true;
  }
  _max_inject_size = connection->max_inject_size();
}

Connection_handler::pool_t Connection_handler::open_pool(const std::string name,
                                                         uint32_t flags)
{
  API_LOCK();

  PMAJOR("open pool: %s", name.c_str());

  /* send pool request message */
  const auto iob = allocate();
  assert(iob);

  const auto msg = new (iob->base()) Dawn::Protocol::Message_pool_request(
      iob->length(), auth_id(), /* auth id */
      ++_request_id, 0,         /* size */
      Dawn::Protocol::OP_OPEN, name);

  iob->set_length(msg->msg_len);

  sync_inject_send(iob);

  sync_recv(iob); /* await response */

  const auto response_msg =
      new (iob->base()) Dawn::Protocol::Message_pool_response();
  if (response_msg->type_id != Dawn::Protocol::MSG_TYPE_POOL_RESPONSE)
    throw Protocol_exception("expected POOL_RESPONSE message - got %x",
                             response_msg->type_id);

  auto pool_id = response_msg->pool_id;
  auto status = msg->status;
  free_buffer(iob);
  if (status == E_FAIL) return Component::IKVStore::POOL_ERROR;
  return pool_id;
}

Connection_handler::pool_t Connection_handler::create_pool(
    const std::string name,
    const size_t      size,
    unsigned int      flags,
    uint64_t          expected_obj_count)
{
  API_LOCK();
  PMAJOR("create pool: %s (expected objs=%lu)", name.c_str(), expected_obj_count);

  /* send pool request message */
  const auto iob = allocate();
  assert(iob);

  PLOG("Connection_handler::create_pool");

  const auto msg = new (iob->base())
    Dawn::Protocol::Message_pool_request(iob->length(),
                                         auth_id(), /* auth id */
                                         ++_request_id,
                                         size,
                                         Dawn::Protocol::OP_CREATE,
                                         name);
  assert(msg->op);
  msg->flags = flags;
  msg->expected_object_count = expected_obj_count;
  
  iob->set_length(msg->msg_len);
  sync_inject_send(iob);

  sync_recv(iob);

  auto response_msg = new (iob->base()) Dawn::Protocol::Message_pool_response();
  if (response_msg->type_id != Dawn::Protocol::MSG_TYPE_POOL_RESPONSE)
    throw Protocol_exception("expected POOL_RESPONSE message - got %x",
                             response_msg->type_id);
  status_t rc = response_msg->status;

  auto pool_id = response_msg->pool_id;
  auto status = msg->status;

  free_buffer(iob);
  if (status != S_OK) return Component::IKVStore::POOL_ERROR;
  return pool_id;
}

status_t Connection_handler::close_pool(pool_t pool)
{
  API_LOCK();
  /* send pool request message */
  auto       iob = allocate();
  const auto msg = new (iob->base()) Dawn::Protocol::Message_pool_request(iob->length(),
                                                                          auth_id(),
                                                                          ++_request_id,
                                                                          Dawn::Protocol::OP_CLOSE);
  msg->pool_id = pool;

  iob->set_length(msg->msg_len);
  sync_inject_send(iob);

  sync_recv(iob);

  auto response_msg = new (iob->base()) Dawn::Protocol::Message_pool_response();
  if (response_msg->type_id != Dawn::Protocol::MSG_TYPE_POOL_RESPONSE)
    throw Protocol_exception("expected POOL_RESPONSE message - got %x",
                             response_msg->type_id);
  auto status = response_msg->status;
  free_buffer(iob);
  return status;
}

status_t Connection_handler::delete_pool(const std::string& name)

{
  API_LOCK();
  /* send pool request message */
  auto       iob = allocate();
  const auto msg = new (iob->base()) Dawn::Protocol::Message_pool_request(iob->length(),
                                                                          auth_id(),
                                                                          ++_request_id,
                                                                          0, // size
                                                                          Dawn::Protocol::OP_DELETE,
                                                                          name);
  iob->set_length(msg->msg_len);
  sync_inject_send(iob);

  sync_recv(iob);

  auto response_msg = new (iob->base()) Dawn::Protocol::Message_pool_response();
  if (response_msg->type_id != Dawn::Protocol::MSG_TYPE_POOL_RESPONSE)
    throw Protocol_exception("expected POOL_RESPONSE message - got %x",
                             response_msg->type_id);
  auto status = response_msg->status;
  free_buffer(iob);
  return status;
}

/**
 * Memcpy version; both key and value are copied
 *
 */
status_t Connection_handler::put(const pool_t      pool,
                                 const std::string key,
                                 const void*       value,
                                 const size_t      value_len,
                                 unsigned int      flags)
{
  return put(pool, key.c_str(), key.length(), value, value_len, flags);
}

status_t Connection_handler::put(const pool_t pool,
                                 const void*  key,
                                 const size_t key_len,
                                 const void*  value,
                                 const size_t value_len,
                                 uint32_t flags)
{
  API_LOCK();

  if (option_DEBUG)
    PINF("put: %.*s (key_len=%lu) (value_len=%lu)", (int) key_len, (char*) key,
         key_len, value_len);

  if ((key_len + value_len + sizeof(Dawn::Protocol::Message_IO_request)) >
      Buffer_manager<Component::IFabric_client>::BUFFER_LEN) {
    return IKVStore::E_TOO_LARGE;
  }

  const auto iob = allocate();

  const auto msg = new (iob->base()) Dawn::Protocol::Message_IO_request(
      iob->length(), auth_id(), ++_request_id, pool,
      Dawn::Protocol::OP_PUT,  // op
      key, key_len, value, value_len);

  if (_options.short_circuit_backend)
    msg->resvd |= Dawn::Protocol::MSG_RESVD_SCBE;

  iob->set_length(msg->msg_len);
  //  sync_inject_send(iob);
  sync_send(iob);

  sync_recv(iob);

  const auto response_msg =
      new (iob->base()) Dawn::Protocol::Message_IO_response();

  if (response_msg->type_id != Dawn::Protocol::MSG_TYPE_IO_RESPONSE)
    throw Protocol_exception("expected IO_RESPONSE message - got %x",
                             response_msg->type_id);

  if (option_DEBUG)
    PLOG("got response from PUT operation: status=%d request_id=%lu",
         response_msg->status, response_msg->request_id);

  auto status = response_msg->status;
  free_buffer(iob);
  return status;
}

status_t Connection_handler::two_stage_put_direct(
    const pool_t                         pool,
    const void*                          key,
    const size_t                         key_len,
    const void*                          value,
    const size_t                         value_len,
    Component::IKVStore::memory_handle_t handle,
    unsigned int                         flags)
{
  using namespace Dawn;

  assert(pool);

  if (option_DEBUG)
    PINF("multi_put_direct: key=(%.*s) key_len=%lu value=(%.20s...) "
         "value_len=%lu handle=%p",
         (int) key_len, (char*) key, key_len, (char*) value, value_len, handle);

  assert(value_len <= _max_message_size);
  assert(value_len > 0);
  
  const auto iob = allocate();

  /* send advance message, this will be followed by partial puts */
  const auto request_id = ++_request_id;
  const auto msg        = new (iob->base())
      Protocol::Message_IO_request(iob->length(), auth_id(), request_id, pool,
                                   Protocol::OP_PUT_ADVANCE,  // op
                                   key, key_len, value_len);
  msg->flags = flags;
  iob->set_length(msg->msg_len);
  sync_inject_send(iob);
  free_buffer(iob);

  /* send value, then wait for reply */
  buffer_t* value_buffer = reinterpret_cast<buffer_t*>(handle);
  value_buffer->set_length(value_len);
  assert(value_buffer->check_magic());

  if (option_DEBUG)
    PLOG("value_buffer: (iov_len=%lu, region=%p, desc=%p)",
         value_buffer->iov->iov_len, value_buffer->region, value_buffer->desc);

  sync_send(value_buffer);  // client owns buffer

  return S_OK;
}

status_t Connection_handler::put_direct(
    const pool_t                         pool,
    const std::string&                   key,
    const void*                          value,
    const size_t                         value_len,
    Component::IKVStore::memory_handle_t handle,
    unsigned int                         flags)
{
  API_LOCK();

  assert(_max_message_size);

  if(pool == 0) {
    PWRN("put_direct: invalid pool identifier");
    return E_INVAL;
  }

  const auto key_len = key.length();
  if ((key_len + value_len + sizeof(Dawn::Protocol::Message_IO_request)) >
      Buffer_manager<Component::IFabric_client>::BUFFER_LEN) {
    /* for large puts, we use a two-stage protocol */
    return two_stage_put_direct(pool,
                                key.c_str(),
                                key_len,
                                value,
                                value_len,
                                handle,
                                flags);
  }

  if (option_DEBUG)
    PLOG("put_direct: key=(%.*s) key_len=%lu value=(%.20s...) value_len=%lu",
         (int) key_len, (char*) key.c_str(), key_len, (char*) value, value_len);

  buffer_t* value_buffer = nullptr;

  /* on-demand register */
  if (handle == IKVStore::HANDLE_NONE)
    throw API_exception("put_direct: memory handle should be provided");

  value_buffer = reinterpret_cast<buffer_t*>(handle);

  value_buffer->set_length(value_len);
  if (!value_buffer->check_magic())
    throw General_exception("put_direct: memory handle is invalid");

  if (option_DEBUG)
    PLOG("value_buffer: (iov_len=%lu, mr=%p, desc=%p)",
         value_buffer->iov->iov_len, value_buffer->region, value_buffer->desc);

  const auto iob = allocate();
  const auto msg = new (iob->base()) Dawn::Protocol::Message_IO_request(
      iob->length(), auth_id(), ++_request_id, pool,
      Dawn::Protocol::OP_PUT,  // op
      key.c_str(), key_len, value_len);

  if (_options.short_circuit_backend)
    msg->resvd |= Dawn::Protocol::MSG_RESVD_SCBE;
  msg->flags = flags;

  iob->set_length(msg->msg_len);
  sync_send(iob, value_buffer); /* send two concatentated buffers */

  sync_recv(
      iob); /* re-using iob; if we want to issue before, we'll have to rework */

  auto response_msg = new (iob->base()) Dawn::Protocol::Message_IO_response();
  if (response_msg->type_id != Dawn::Protocol::MSG_TYPE_IO_RESPONSE)
    throw Protocol_exception("expected IO_RESPONSE message - got 0x%x",
                             response_msg->type_id);

  if (option_DEBUG)
    PLOG("got response from PUT_DIRECT operation: status=%d", msg->status);

  auto status = response_msg->status;
  free_buffer(iob);
  return status;
}

status_t Connection_handler::get(const pool_t       pool,
                                 const std::string& key,
                                 std::string&       value)
{
  API_LOCK();

  const auto iob = allocate();
  assert(iob);

  const auto msg = new (iob->base()) Dawn::Protocol::Message_IO_request(
      iob->length(), auth_id(), ++_request_id, pool,
      Dawn::Protocol::OP_GET,  // op
      key, "");

  if (_options.short_circuit_backend)
    msg->resvd |= Dawn::Protocol::MSG_RESVD_SCBE;

  iob->set_length(msg->msg_len);
  sync_inject_send(iob);

  // TODO post recv before send
  sync_recv(iob);

  auto response_msg = new (iob->base()) Dawn::Protocol::Message_IO_response();
  if (response_msg->type_id != Dawn::Protocol::MSG_TYPE_IO_RESPONSE)
    throw Protocol_exception("expected IO_RESPONSE message - got %x",
                             response_msg->type_id);

  if (option_DEBUG)
    PLOG("got response from GET operation: status=%d (%s)", msg->status,
         response_msg->data);

  status_t status = response_msg->status;

  /* copy result */
  if (status == S_OK) {
    value.reserve(response_msg->data_len + 1);
    value.insert(0, response_msg->data, response_msg->data_len);
    assert(response_msg->data);
  }

  free_buffer(iob);
  return status;
}

status_t Connection_handler::get(const pool_t       pool,
                                 const std::string& key,
                                 void*&             value,
                                 size_t&            value_len)
{
  API_LOCK();

  const auto iob = allocate();
  assert(iob);

  const auto msg = new (iob->base()) Dawn::Protocol::Message_IO_request(
      iob->length(), auth_id(), ++_request_id, pool,
      Dawn::Protocol::OP_GET,  // op
      key.c_str(), key.length(), 0);
  
  if (_options.short_circuit_backend)
    msg->resvd |= Dawn::Protocol::MSG_RESVD_SCBE;

  iob->set_length(msg->msg_len);
  sync_inject_send(iob);

  sync_recv(iob); /* TODO; could we issue the recv and send together? */

  const auto response_msg =
      new (iob->base()) Dawn::Protocol::Message_IO_response();
  if (response_msg->type_id != Dawn::Protocol::MSG_TYPE_IO_RESPONSE)
    throw Protocol_exception("expected IO_RESPONSE message - got %x",
                             response_msg->type_id);

  if (option_DEBUG)
    PLOG("got response from GET operation: status=%d request_id=%lu "
         "data_len=%lu",
         response_msg->status, response_msg->request_id,
         response_msg->data_length());

  if (response_msg->status != S_OK) return response_msg->status;

  if (option_DEBUG) PLOG("message value:(%s)", response_msg->data);

  if (response_msg->is_set_twostage_bit()) {
    /* two-stage get */
    const auto data_len = response_msg->data_length() + 1;
    value               = ::aligned_alloc(MiB(2), data_len);
    madvise(value, data_len, MADV_HUGEPAGE);

    auto region = register_memory(
        value, data_len); /* we could have some pre-registered? */
    auto desc = get_memory_descriptor(region);

    iovec iov{value, data_len - 1};
    post_recv(&iov, (&iov) + 1, &desc, &iov);

    /* synchronously wait for receive to complete */
    wait_for_completion(&iov);

    deregister_memory(region);
  }
  else {
    /* copy off value from IO buffer */
    value     = ::malloc(response_msg->data_len + 1);
    value_len = response_msg->data_len;

    memcpy(value, response_msg->data, response_msg->data_len);
    ((char*) value)[response_msg->data_len] = '\0';
  }

  auto status = response_msg->status;
  free_buffer(iob);
  return status;
}

status_t Connection_handler::get_direct(
    const pool_t                         pool,
    const std::string&                   key,
    void*                                value,
    size_t&                              out_value_len,
    Component::IKVStore::memory_handle_t handle)
{
  API_LOCK();

  if (!value || out_value_len == 0)
    throw API_exception("get_direct bad parameters");

  buffer_t* value_iob = reinterpret_cast<buffer_t*>(handle);
  if (!value_iob->check_magic())
    throw API_exception("bad handle parameter to get_direct");

  const auto iob = allocate();
  assert(iob);

  const auto msg = new (iob->base()) Dawn::Protocol::Message_IO_request(
      iob->length(), auth_id(), ++_request_id, pool, Dawn::Protocol::OP_GET,
      key.c_str(), key.length(), 0);

  iob->set_length(msg->msg_len);
  sync_inject_send(iob);

  sync_recv(iob); /* TODO; could we issue the recv and send together? */

  auto response_msg = new (iob->base()) Dawn::Protocol::Message_IO_response();
  if (response_msg->type_id != Dawn::Protocol::MSG_TYPE_IO_RESPONSE)
    throw Protocol_exception("expected IO_RESPONSE message - got %x",
                             response_msg->type_id);

  if (option_DEBUG)
    PLOG("got response from GET operation: status=%d request_id=%lu "
         "data_len=%lu",
         response_msg->status, response_msg->request_id,
         response_msg->data_length());

  if (response_msg->status != S_OK) return response_msg->status;

  if (option_DEBUG) PLOG("value:(%s)", response_msg->data);

  if (response_msg->is_set_twostage_bit()) {
    /* two-stage get */
    post_recv(value_iob);

    /* synchronously wait for receive to complete */
    wait_for_completion(value_iob);
  }
  else {
    /* copy off value from IO buffer */
    if (out_value_len < response_msg->data_len) {
      assert(0);
      free_buffer(iob);
      return E_INSUFFICIENT_SPACE;
    }

    out_value_len = response_msg->data_len;

    memcpy(value, response_msg->data, response_msg->data_len);
  }

  auto status = response_msg->status;
  free_buffer(iob);
  return status;
}

status_t Connection_handler::erase(const pool_t pool,
                                   const std::string& key)
{
  API_LOCK();

  const auto iob = allocate();
  assert(iob);

  const auto msg = new (iob->base()) Dawn::Protocol::Message_IO_request(
      iob->length(), auth_id(), ++_request_id, pool,
      Dawn::Protocol::OP_ERASE,
      key.c_str(), key.length(), 0);

  iob->set_length(msg->msg_len);
  sync_inject_send(iob);

  sync_recv(iob);

  auto response_msg = new (iob->base()) Dawn::Protocol::Message_IO_response();
  if (response_msg->type_id != Dawn::Protocol::MSG_TYPE_IO_RESPONSE)
    throw Protocol_exception("expected IO_RESPONSE message - got %x",
                             response_msg->type_id);

  if (option_DEBUG)
    PLOG("got response from ERASE operation: status=%d request_id=%lu "
         "data_len=%lu",
         response_msg->status, response_msg->request_id,
         response_msg->data_length());

  auto status = response_msg->status;
  free_buffer(iob);
  return status;
}


int Connection_handler::tick()
{
  using namespace Dawn::Protocol;

  switch (_state) {
    case INITIALIZE: {
      set_state(HANDSHAKE_SEND);
      break;
    }
    case HANDSHAKE_SEND: {
      PMAJOR("client : HANDSHAKE_SEND");
      auto iob = allocate();
      auto msg = new (iob->base()) Dawn::Protocol::Message_handshake(0, 1);

      iob->set_length(msg->msg_len);
      post_send(iob->iov, iob->iov + 1, &iob->desc, iob);

      wait_for_completion(iob);

      free_buffer(iob);
      set_state(HANDSHAKE_GET_RESPONSE);
      break;
    }
    case HANDSHAKE_GET_RESPONSE: {
      auto iob = allocate();
      post_recv(iob->iov, iob->iov + 1, &iob->desc, iob);

      wait_for_completion(iob);

      Message_handshake_reply* msg = (Message_handshake_reply*) iob->base();
      if (msg->type_id == Dawn::Protocol::MSG_TYPE_HANDSHAKE_REPLY)
        set_state(READY);
      else
        throw Protocol_exception("client: expecting handshake reply got type_id=%u len=%lu",
                                 msg->type_id, msg->msg_len);

      PMAJOR("client : HANDSHAKE_GET_RESPONSE (max_message_size=%lu MiB)",
             REDUCE_MiB(msg->max_message_size));
      _max_message_size = msg->max_message_size;
      free_buffer(iob);
      break;
    }
    case READY: {
      return 0;
      break;
    }
    case SHUTDOWN: {
      auto iob = allocate();
      auto msg = new (iob->base())
          Dawn::Protocol::Message_close_session((uint64_t) this);

      iob->set_length(msg->msg_len);
      post_send(iob->iov, iob->iov + 1, &iob->desc, iob);

      wait_for_completion(iob);

      free_buffer(iob);
      set_state(STOPPED);
      PLOG("Dawn_client: connection %p shutdown.", this);
      return 0;
    }
    case STOPPED: {
      assert(0);
      return 0;
    }
  }  // end switch

  return 1;
}

}  // namespace Client
}  // namespace Dawn
