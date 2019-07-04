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

#include <city.h>
#include <common/cycles.h>
#include <common/utils.h>
#include <unistd.h>

#include "connection.h"
#include "protocol.h"

#include <memory>

using namespace Component;

namespace
{
  /*
   * First, cast the response buffer to Message (checking version).
   * Second, cast the Message to a specific message type (checking message type).
   */
  template <typename Type>
    const Type *response_ptr(void *b)
    {
      const auto *const msg = Dawn::Protocol::message_cast(b);
      return msg->ptr_cast<Type>();
    }
}

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

Connection_handler::~Connection_handler()
{
  PLOG("Connection_handler::dtor (%p)", this);
}

/* The various embedded returns and throws suggest that the allocated
 * iobs should be automatically freed to avoid leaks.
 */

class iob_free
{
  Connection_handler *_h;
public:
  iob_free(Connection_handler *h_)
    : _h(h_)
  {}
  void operator()(Connection_handler::buffer_t *iob) { _h->free_buffer(iob); }
};

Connection_handler::pool_t Connection_handler::open_pool(const std::string name,
                                                         uint32_t flags)
{
  API_LOCK();

  PMAJOR("open pool: %s", name.c_str());

  /* send pool request message */

  /* Use unique_ptr to ensure that the dynamically buffers are freed.
   * unique_ptr protects against the code forgetting to call free_buffer,
   * which it usually did when the function exited by a throw or a
   * non-terminal return.
   *
   * The type std::unique_ptr<buffer_t, iob_free> is a bit ugly, and
   * could be simplified by a typedef, e.g.
   *   using buffer_ptr_t = std::unique_ptr<buffer_t, iob_free>
   * but it can stay as it is for now to reduce the levels of indirection
   * necessary to understand what it does.
   */
  const auto iobs = std::unique_ptr<buffer_t, iob_free>(allocate(), this);
  const auto iobr = std::unique_ptr<buffer_t, iob_free>(allocate(), this);
  assert(iobs);
  assert(iobr);

  Component::IKVStore::pool_t pool_id;

  assert(&*iobr != &*iobs);
  
  // temp
  memset(iobr->base(),0xbb,iobr->length());
  
  try {
    const auto msg = new (iobs->base()) Dawn::Protocol::Message_pool_request(iobs->length(),
									     auth_id(), /* auth id */
									     ++_request_id,
									     0,         /* size */
									     Dawn::Protocol::OP_OPEN,
									     name);
    
    iobs->set_length(msg->msg_len);

    /* The &* notation extracts a raw pointer form the "unique_ptr".
     * The difference is that the standard pointer does not imply
     * ownership; the function is simply "borrowing" the pointer.
     * Here, &*iobr is equivalent to iobr->get(). The choice is a
     * matter of style. &* uses two fewer tokens.
     */
    /* the sequence "post_recv, sync_inject_send, wait_for_completion"
     * is common enough that it probably deserves its own function.
     * But we may find that the entire single-exchange pattern as seen
     * in open_pool, create_pool and several others could be placed in
     * a single template function.
     */

    post_recv(&*iobr);
    sync_inject_send(&*iobs);
    wait_for_completion(&*iobr); /* await response */

    const auto response_msg =
      response_ptr<const Dawn::Protocol::Message_pool_response>(iobr->base());

    pool_id = response_msg->pool_id;
  }
  catch(...) {
    pool_id = Component::IKVStore::POOL_ERROR;
  }
  return pool_id;
}

Connection_handler::pool_t Connection_handler::create_pool(const std::string name,
                                                           const size_t      size,
                                                           unsigned int      flags,
                                                           uint64_t          expected_obj_count)
{
  API_LOCK();
  PMAJOR("create pool: %s (expected objs=%lu)", name.c_str(), expected_obj_count);

  /* send pool request message */
  const auto iobs = std::unique_ptr<buffer_t, iob_free>(allocate(), this);
  const auto iobr = std::unique_ptr<buffer_t, iob_free>(allocate(), this);
  assert(iobs);
  assert(iobr);

  PLOG("Connection_handler::create_pool");

  Component::IKVStore::pool_t pool_id;

  try {
    const auto msg = new (iobs->base())
      Protocol::Message_pool_request(iobs->length(),
				     auth_id(), /* auth id */
				     ++_request_id,
				     size,
				     Dawn::Protocol::OP_CREATE,
				     name);
    assert(msg->op);
    msg->flags = flags;
    msg->expected_object_count = expected_obj_count;

    iobs->set_length(msg->msg_len);

    post_recv(&*iobr);
    sync_inject_send(&*iobs);
    wait_for_completion(&*iobr);

    const auto response_msg =
      response_ptr<const Dawn::Protocol::Message_pool_response>(iobr->base());

    pool_id = response_msg->pool_id;
  }
  catch(...) {
    pool_id = Component::IKVStore::POOL_ERROR;
  }

  /* Note: most request/response pairs return status. This one returns a pool_id instead. */
  return pool_id;
}

status_t Connection_handler::close_pool(pool_t pool)
{
  API_LOCK();
  /* send pool request message */
  const auto iobs = std::unique_ptr<buffer_t, iob_free>(allocate(), this);
  const auto iobr = std::unique_ptr<buffer_t, iob_free>(allocate(), this);
  const auto msg = new (iobs->base()) Dawn::Protocol::Message_pool_request(iobs->length(),
                                                                          auth_id(),
                                                                          ++_request_id,
                                                                          Dawn::Protocol::OP_CLOSE);
  msg->pool_id = pool;

  iobs->set_length(msg->msg_len);

  post_recv(&*iobr);
  sync_inject_send(&*iobs);
  wait_for_completion(&*iobr);

  const auto response_msg =
    response_ptr<const Dawn::Protocol::Message_pool_response>(iobr->base());

  const auto status = response_msg->status;
  return status;
}

status_t Connection_handler::delete_pool(const std::string& name)

{
  API_LOCK();
  
  const auto iobs = std::unique_ptr<buffer_t, iob_free>(allocate(), this);
  const auto iobr = std::unique_ptr<buffer_t, iob_free>(allocate(), this);

  const auto msg = new (iobs->base()) Dawn::Protocol::Message_pool_request(iobs->length(),
                                                                          auth_id(),
                                                                          ++_request_id,
                                                                          0, // size
                                                                          Dawn::Protocol::OP_DELETE,
                                                                          name);
  iobs->set_length(msg->msg_len);

  post_recv(&*iobr);
  sync_inject_send(&*iobs);
  wait_for_completion(&*iobr);

  const auto response_msg =
    response_ptr<const Dawn::Protocol::Message_pool_response>(iobr->base());
  
  return response_msg->status;
}

status_t Connection_handler::configure_pool(const Component::IKVStore::pool_t pool,
                                            const std::string& json)
{
  API_LOCK();

  const auto iobs = std::unique_ptr<buffer_t, iob_free>(allocate(), this);
  const auto iobr = std::unique_ptr<buffer_t, iob_free>(allocate(), this);

  const auto msg = new (iobs->base()) Dawn::Protocol::Message_IO_request(iobs->length(),
                                                                        auth_id(),
                                                                        ++_request_id,
                                                                        pool,
                                                                        Dawn::Protocol::OP_CONFIGURE,  // op
                                                                        json);
  if ((json.length() + sizeof(Dawn::Protocol::Message_IO_request)) >
      Buffer_manager<Component::IFabric_client>::BUFFER_LEN)
    return IKVStore::E_TOO_LARGE;

  iobs->set_length(msg->msg_len);

  post_recv(&*iobr);
  sync_inject_send(&*iobs);
  wait_for_completion(&*iobr);

  const auto response_msg =
    response_ptr<const Dawn::Protocol::Message_IO_response>(iobr->base());

  if (option_DEBUG)
    PLOG("got response from CONFIGURE operation: status=%d request_id=%lu",
         response_msg->status, response_msg->request_id);

  return response_msg->status;
}

/**
 * Memcpy version; both key and value are copied
 *
 */
status_t Connection_handler::put(const pool_t      pool,
                                 const std::string key,
                                 const void*       value,
                                 const size_t      value_len,
                                 uint32_t      flags)
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

  /* check key length */
  if ((key_len + value_len + sizeof(Dawn::Protocol::Message_IO_request)) >
      Buffer_manager<Component::IFabric_client>::BUFFER_LEN) {
    PWRN("Dawn_client::put value length (%lu) too long. Use put_direct.", value_len);
    return IKVStore::E_TOO_LARGE;
  }

  const auto iobs = std::unique_ptr<buffer_t, iob_free>(allocate(), this);
  const auto iobr = std::unique_ptr<buffer_t, iob_free>(allocate(), this);

  status_t status;

  try {
    const auto msg = new (iobs->base()) Dawn::Protocol::Message_IO_request(iobs->length(),
                                                                          auth_id(),
                                                                          ++_request_id,
                                                                          pool,
                                                                          Dawn::Protocol::OP_PUT,  // op
                                                                          key,
                                                                          key_len,
                                                                          value,
                                                                          value_len,
                                                                          flags);

    if (_options.short_circuit_backend)
      msg->resvd |= Dawn::Protocol::MSG_RESVD_SCBE;

    iobs->set_length(msg->msg_len);

    post_recv(&*iobr);
    sync_send(&*iobs);
    wait_for_completion(&*iobr);

    const auto response_msg =
      response_ptr<const Dawn::Protocol::Message_IO_response>(iobr->base());

    if (option_DEBUG)
      PLOG("got response from PUT operation: status=%d request_id=%lu",
           response_msg->status, response_msg->request_id);

    status = response_msg->status;
  }
  catch(...) {
    status = E_FAIL;
  }

  return status;
}

status_t Connection_handler::two_stage_put_direct(const pool_t                         pool,
                                                  const void*                          key,
                                                  const size_t                         key_len,
                                                  const void*                          value,
                                                  const size_t                         value_len,
                                                  Component::IKVStore::memory_handle_t handle,
                                                  uint32_t                             flags)
{
  using namespace Dawn;

  assert(pool);

  assert(value_len <= _max_message_size);
  assert(value_len > 0);

  {
    const auto iobs = std::unique_ptr<buffer_t, iob_free>(allocate(), this);
    const auto iobr = std::unique_ptr<buffer_t, iob_free>(allocate(), this);

    /* send advance leader message */
    const auto request_id = ++_request_id;
    const auto msg        = new (iobs->base())
      Protocol::Message_IO_request(iobs->length(),
                                   auth_id(), request_id, pool,
                                   Protocol::OP_PUT_ADVANCE,  // op
                                   key, key_len, value_len, flags);
    msg->flags = flags;
    iobs->set_length(msg->msg_len);

    post_recv(&*iobr);
    sync_inject_send(&*iobs);
    wait_for_completion(&*iobr);

    const auto response_msg =
      response_ptr<const Dawn::Protocol::Message_IO_response>(iobr->base());
    
    /* wait for response from header before posting the value */    

    if(option_DEBUG)
      PMAJOR("got response (status=%u) from put direct header",response_msg->status);

    /* if response is not OK, don't follow with the value */
    if(response_msg->status != S_OK) {
      return msg->status;
    }
  }

  /* send value */
  buffer_t* value_buffer = reinterpret_cast<buffer_t*>(handle);
  value_buffer->set_length(value_len);
  assert(value_buffer->check_magic());

  if (option_DEBUG)
    PLOG("value_buffer: (iov_len=%lu, region=%p, desc=%p)",
         value_buffer->iov->iov_len, value_buffer->region, value_buffer->desc);

  sync_send(value_buffer);  // client owns buffer

  if (option_DEBUG) {
    PINF("two_stage_put_direct: complete");
  }

  return S_OK;
}

status_t Connection_handler::put_direct(const pool_t                         pool,
                                        const std::string&                   key,
                                        const void*                          value,
                                        const size_t                         value_len,
                                        Component::IKVStore::memory_handle_t handle,
                                        uint32_t                             flags)
{
  API_LOCK();

  if (handle == IKVStore::HANDLE_NONE) {
    PWRN("put_direct: memory handle should be provided");
    return E_BAD_PARAM;
  }

  assert(_max_message_size);

  if(pool == 0) {
    PWRN("put_direct: invalid pool identifier");
    return E_INVAL;
  }

  buffer_t* value_buffer = reinterpret_cast<buffer_t*>(handle);
  value_buffer->set_length(value_len);

  if (!value_buffer->check_magic()) {
    PWRN("put_direct: memory handle is invalid");
    return E_INVAL;
  }

  status_t status;

  try {

    const auto key_len = key.length();
    if ((key_len + value_len + sizeof(Dawn::Protocol::Message_IO_request)) >
      Buffer_manager<Component::IFabric_client>::BUFFER_LEN) {

      /* check value is not too large for underlying transport */
      if(value_len > _max_message_size) {
        return IKVStore::E_TOO_LARGE;
      }

      /* for large puts, where the receiver will not have
       * sufficient buffer space, we use a two-stage protocol */
      return two_stage_put_direct(pool,
                                  key.c_str(),
                                  key_len,
                                  value,
                                  value_len,
                                  handle,
                                  flags);
    }

    const auto iobs = std::unique_ptr<buffer_t, iob_free>(allocate(), this);
    const auto iobr = std::unique_ptr<buffer_t, iob_free>(allocate(), this);

    if (option_DEBUG ||1) {
      PLOG("put_direct: key=(%.*s) key_len=%lu value=(%.20s...) value_len=%lu",
           (int) key_len, (char*) key.c_str(), key_len, (char*) value, value_len);

      PLOG("value_buffer: (iov_len=%lu, mr=%p, desc=%p)",
           value_buffer->iov->iov_len, value_buffer->region, value_buffer->desc);
    }

    const auto msg = new (iobs->base()) Dawn::Protocol::Message_IO_request(iobs->length(),
                                                                          auth_id(),
                                                                          ++_request_id,
                                                                          pool,
                                                                          Dawn::Protocol::OP_PUT,  // op
                                                                          key.c_str(),
                                                                          key_len,
                                                                          value_len,
                                                                          flags);

    if (_options.short_circuit_backend)
      msg->resvd |= Dawn::Protocol::MSG_RESVD_SCBE;

    msg->flags = flags;

    iobs->set_length(msg->msg_len);

    post_recv(&*iobr);
    sync_send(&*iobs, value_buffer); /* send two concatentated buffers in single DMA */
    wait_for_completion(&*iobr); /* get response */

    const auto response_msg =
      response_ptr<const Dawn::Protocol::Message_IO_response>(iobr->base());

    if (option_DEBUG)
      PLOG("got response from PUT_DIRECT operation: status=%d", msg->status);

    status = response_msg->status;
  }
  catch(...) {
    status = E_FAIL;
  }

  return status;
}

status_t Connection_handler::get(const pool_t       pool,
                                 const std::string& key,
                                 std::string&       value)
{
  API_LOCK();

  const auto iobs = std::unique_ptr<buffer_t, iob_free>(allocate(), this);
  const auto iobr = std::unique_ptr<buffer_t, iob_free>(allocate(), this);
  assert(iobs);
  assert(iobr);

  status_t status;

  try {

    const auto msg = new (iobs->base()) Dawn::Protocol::Message_IO_request(iobs->length(),
                                                                          auth_id(),
                                                                          ++_request_id,
                                                                          pool,
                                                                          Dawn::Protocol::OP_GET,  // op
                                                                          key,
                                                                          "",
                                                                          0);

    if (_options.short_circuit_backend)
      msg->resvd |= Dawn::Protocol::MSG_RESVD_SCBE;

    iobs->set_length(msg->msg_len);

    post_recv(&*iobr);
    sync_inject_send(&*iobs);
    wait_for_completion(&*iobr);

    const auto response_msg =
      response_ptr<const Dawn::Protocol::Message_IO_response>(iobr->base());

    if (option_DEBUG)
      PLOG("got response from GET operation: status=%d (%s)", msg->status,
           response_msg->data);

    status = response_msg->status;
    value.reserve(response_msg->data_len + 1);
    value.insert(0, response_msg->data, response_msg->data_len);
    assert(response_msg->data);
  }
  catch(...) {
    status = E_FAIL;
  }

  return status;
}

status_t Connection_handler::get(const pool_t       pool,
                                 const std::string& key,
                                 void*&             value,
                                 size_t&            value_len)
{
  API_LOCK();

  const auto iobs = std::unique_ptr<buffer_t, iob_free>(allocate(), this);
  const auto iobr = std::unique_ptr<buffer_t, iob_free>(allocate(), this);
  assert(iobs);
  assert(iobr);

  status_t status;

  try {

    const auto msg = new (iobs->base()) Dawn::Protocol::Message_IO_request(iobs->length(),
                                                                          auth_id(),
                                                                          ++_request_id,
                                                                          pool,
                                                                          Dawn::Protocol::OP_GET,  // op
                                                                          key.c_str(),
                                                                          key.length(),
                                                                          0);

    /* indicate how much space has been allocated on this side. For
       get this is based on buffer size
    */
    msg->val_len = iobs->original_length - sizeof(Dawn::Protocol::Message_IO_response);

    if (_options.short_circuit_backend)
      msg->resvd |= Dawn::Protocol::MSG_RESVD_SCBE;

    iobs->set_length(msg->msg_len);

    post_recv(&*iobr);
    sync_inject_send(&*iobs);
    wait_for_completion(&*iobr); /* TODO; could we issue the recv and send together? */

    const auto response_msg =
      response_ptr<const Dawn::Protocol::Message_IO_response>(iobr->base());

    if (option_DEBUG)
      PLOG("got response from GET operation: status=%d request_id=%lu data_len=%lu",
           response_msg->status, response_msg->request_id,
           response_msg->data_length());

    if (response_msg->status != S_OK)
      return response_msg->status;

    if (option_DEBUG) PLOG("message value:(%s)", response_msg->data);

    if (response_msg->is_set_twostage_bit()) {
      /* two-stage get */
      const auto data_len = response_msg->data_length() + 1;
      value               = ::aligned_alloc(MiB(2), data_len);
      madvise(value, data_len, MADV_HUGEPAGE);

      auto region = register_memory(value, data_len); /* we could have some pre-registered? */
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

    status = response_msg->status;
  }
  catch(...) {
    status = E_FAIL;
  }

  return status;
}

status_t Connection_handler::get_direct(const pool_t                         pool,
                                        const std::string&                   key,
                                        void*                                value,
                                        size_t&                              out_value_len,
                                        Component::IKVStore::memory_handle_t handle)
{
  API_LOCK();

  if (!value || out_value_len == 0)
    return E_BAD_PARAM;

  buffer_t* value_iob = reinterpret_cast<buffer_t*>(handle);
  if (!value_iob->check_magic()) {
    PWRN("bad handle parameter to get_direct");
    return E_BAD_PARAM;
  }

  /* check value is not too large for underlying transport */
  if(out_value_len > _max_message_size)
    return IKVStore::E_TOO_LARGE;

  const auto iobs = std::unique_ptr<buffer_t, iob_free>(allocate(), this);
  const auto iobr = std::unique_ptr<buffer_t, iob_free>(allocate(), this);
  assert(iobs);
  assert(iobr);

  status_t status;
  try {
    const auto msg = new (iobs->base()) Dawn::Protocol::Message_IO_request(iobs->length(),
                                                                          auth_id(),
                                                                          ++_request_id,
                                                                          pool,
                                                                          Dawn::Protocol::OP_GET,
                                                                          key.c_str(),
                                                                          key.length(),
                                                                          0);

    /* indicate that this is a direct request and register
       how much space has been allocated on this side. For
       get_direct this is allocated by the client */
    msg->resvd = Protocol::MSG_RESVD_DIRECT;
    msg->val_len = out_value_len;

    iobs->set_length(msg->msg_len);

    post_recv(&*iobr);
    sync_inject_send(&*iobs);
    wait_for_completion(&*iobr); /* get response */

    const auto response_msg =
      response_ptr<const Dawn::Protocol::Message_IO_response>(iobr->base());

    if(option_DEBUG)
      PLOG("get_direct: got initial response (two_stage=%s)",
           response_msg->is_set_twostage_bit() ?  "true" : "false");

    /* insufficent space should have been dealt with already */
    assert(out_value_len >= response_msg->data_length());

    status = response_msg->status;

    /* if response not S_OK, do not do anything else */
    if (status != S_OK) {
      return status;
    }

    /* set out_value_len to receiving length */
    out_value_len = response_msg->data_length();

    if (response_msg->is_set_twostage_bit()) {
      /* two-stage get */
      post_recv(value_iob);

      /* synchronously wait for receive to complete */
      wait_for_completion(value_iob);
    }
    else {
      memcpy(value, response_msg->data, response_msg->data_len);
    }

    status = response_msg->status;

  }
  catch(...) {
    status = E_FAIL;
  }

  return status;
}

status_t Connection_handler::erase(const pool_t pool,
                                   const std::string& key)
{
  API_LOCK();

  const auto iobs = std::unique_ptr<buffer_t, iob_free>(allocate(), this);
  const auto iobr = std::unique_ptr<buffer_t, iob_free>(allocate(), this);
  assert(iobs);
  assert(iobr);

  status_t status;

  try {
    const auto msg = new (iobs->base()) Dawn::Protocol::Message_IO_request(iobs->length(),
                                                                          auth_id(),
                                                                          ++_request_id,
                                                                          pool,
                                                                          Dawn::Protocol::OP_ERASE,
                                                                          key.c_str(),
                                                                          key.length(),
                                                                          0);

    iobs->set_length(msg->msg_len);

    post_recv(&*iobr);
    sync_inject_send(&*iobs);
    wait_for_completion(&*iobr);

    const auto response_msg =
      response_ptr<const Dawn::Protocol::Message_IO_response>(iobr->base());

    if (option_DEBUG)
      PLOG("got response from ERASE operation: status=%d request_id=%lu data_len=%lu",
           response_msg->status, response_msg->request_id,
           response_msg->data_length());
    status = response_msg->status;
  }
  catch(...) {
    status = E_FAIL;
  }

  return status;
}

size_t Connection_handler::count(const pool_t pool)
{
  API_LOCK();

  const auto iobs = std::unique_ptr<buffer_t, iob_free>(allocate(), this);
  const auto iobr = std::unique_ptr<buffer_t, iob_free>(allocate(), this);
  assert(iobs);
  assert(iobr);

  try {
    const auto msg = new (iobs->base()) Dawn::Protocol::Message_INFO_request(auth_id());
    msg->pool_id = pool;
    msg->type = Component::IKVStore::Attribute::COUNT;
    iobs->set_length(msg->base_message_size());

    post_recv(&*iobr);
    sync_inject_send(&*iobs);
    wait_for_completion(&*iobr);

    const auto response_msg =
      response_ptr<const Dawn::Protocol::Message_INFO_response>(iobr->base());

    return response_msg->value;
  }
  catch ( ... )
  {
    return 0;
  }
}

status_t Connection_handler::get_attribute(const IKVStore::pool_t pool,
                                           const IKVStore::Attribute attr,
                                           std::vector<uint64_t>& out_attr,
                                           const std::string* key)
{

  API_LOCK();

  const auto iobs = std::unique_ptr<buffer_t, iob_free>(allocate(), this);
  const auto iobr = std::unique_ptr<buffer_t, iob_free>(allocate(), this);
  assert(iobs);
  assert(iobr);

  status_t status;

  try {
    const auto msg = new (iobs->base()) Dawn::Protocol::Message_INFO_request(auth_id());
    msg->pool_id = pool;

    msg->type = attr;
    msg->set_key(iobs->length(), *key);
    iobs->set_length(msg->message_size());

    post_recv(&*iobr);
    sync_inject_send(&*iobs);
    wait_for_completion(&*iobr);

    const auto response_msg =
      response_ptr<const Dawn::Protocol::Message_INFO_response>(iobr->base());

    out_attr.clear();
    out_attr.push_back(response_msg->value);
    status = response_msg->status;
  }
  catch(...) {
    status = E_FAIL;
  }

  return status;
}


status_t Connection_handler::get_statistics(Component::IDawn::Shard_stats& out_stats)
{
  API_LOCK();

  const auto iobs = std::unique_ptr<buffer_t, iob_free>(allocate(), this);
  const auto iobr = std::unique_ptr<buffer_t, iob_free>(allocate(), this);
  assert(iobs);
  assert(iobr);

  status_t status;

  try {
    const auto msg = new (iobs->base()) Dawn::Protocol::Message_INFO_request(auth_id());
    msg->pool_id = 0;
    msg->type = Dawn::Protocol::INFO_TYPE_GET_STATS;
    iobs->set_length(msg->message_size());

    post_recv(&*iobr);
    sync_inject_send(&*iobs);
    wait_for_completion(&*iobr);

    const auto response_msg =
      response_ptr<const Dawn::Protocol::Message_stats>(iobr->base());

    status = response_msg->status;
    out_stats = response_msg->stats;
  }
  catch(...) {
    status = E_FAIL;
  }

  return status;
}

status_t Connection_handler::find(const IKVStore::pool_t pool,
                                  const std::string& key_expression,
                                  const offset_t offset,
                                  offset_t& out_matched_offset,
                                  std::string& out_matched_key)
{
  API_LOCK();

  const auto iobs = std::unique_ptr<buffer_t, iob_free>(allocate(), this);
  const auto iobr = std::unique_ptr<buffer_t, iob_free>(allocate(), this);
  assert(iobs);
  assert(iobr);

  status_t status;

  try {
    const auto msg = new (iobs->base()) Dawn::Protocol::Message_INFO_request(auth_id());
    msg->pool_id = pool;
    msg->type = Dawn::Protocol::INFO_TYPE_FIND_KEY;
    msg->offset = offset;

    msg->set_key(iobs->length(), key_expression);
    iobs->set_length(msg->message_size());

    post_recv(&*iobr);
    sync_inject_send(&*iobs);
    wait_for_completion(&*iobr);

    const auto response_msg =
      response_ptr<const Dawn::Protocol::Message_INFO_response>(iobr->base());

    status = response_msg->status;

    if(status == S_OK) {
      out_matched_key = response_msg->c_str();
      out_matched_offset = response_msg->offset;
    }
  }
  catch(...) {
    status = E_FAIL;
  }

  return status;
}


status_t Connection_handler::invoke_ado(const IKVStore::pool_t pool,
                                        const std::string& key,
                                        const std::string& command,
                                        const uint32_t flags,                              
                                        std::string& out_response,
                                        const size_t value_size)
{
  API_LOCK();

  const auto iobs = std::unique_ptr<buffer_t, iob_free>(allocate(), this);
  const auto iobr = std::unique_ptr<buffer_t, iob_free>(allocate(), this);
  assert(iobs);
  assert(iobr);

  status_t status;

  try {
    const auto msg = new (iobs->base()) Dawn::Protocol::Message_ado_request(iobs->length(),
                                                                            auth_id(),
                                                                            ++_request_id,
                                                                            pool,
                                                                            key,
                                                                            command,
                                                                            value_size);
    iobs->set_length(msg->message_size());

    post_recv(&*iobr);
    sync_send(&*iobs);
    wait_for_completion(&*iobr);

    const auto response_msg =
      response_ptr<const Dawn::Protocol::Message_ado_response>(iobr->base());

    status = response_msg->status;

    if(status == S_OK) {
      out_response = response_msg->response();
      PLOG("!!!! GOT ADO response (%s)", out_response.c_str());
    }
  }
  catch(...) {
    status = E_FAIL;
  }

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
    const auto iob = std::unique_ptr<buffer_t, iob_free>(allocate(), this);
    auto msg = new (iob->base()) Dawn::Protocol::Message_handshake(0, 1);

    iob->set_length(msg->msg_len);
    post_send(iob->iov, iob->iov + 1, &iob->desc, &*iob);

    wait_for_completion(&*iob);

    set_state(HANDSHAKE_GET_RESPONSE);
    break;
  }
  case HANDSHAKE_GET_RESPONSE: {
    const auto iob = std::unique_ptr<buffer_t, iob_free>(allocate(), this);
    post_recv(iob->iov, iob->iov + 1, &iob->desc, &*iob);

    wait_for_completion(&*iob);

    const auto response_msg =
      response_ptr<const Dawn::Protocol::Message_handshake_reply>(iob->base());

    set_state(READY);

    PMAJOR("client : HANDSHAKE_GET_RESPONSE");

    _max_message_size = max_message_size(); /* from fabric component */
    break;
  }
  case READY: {
    return 0;
    break;
  }
  case SHUTDOWN: {
    const auto iob = std::unique_ptr<buffer_t, iob_free>(allocate(), this);
    auto msg = new (iob->base())
      Dawn::Protocol::Message_close_session((uint64_t) this);

    iob->set_length(msg->msg_len);
    post_send(iob->iov, iob->iov + 1, &iob->desc, &*iob);

    wait_for_completion(&*iob);

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
