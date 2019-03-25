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
//#define PROFILE

#include <api/components.h>
#include <common/dump_utils.h>
#include <common/utils.h>

#ifdef PROFILE
#include <gperftools/profiler.h>
#endif

#include "shard.h"

using namespace Dawn;

namespace Dawn
{
/* global parameters */
namespace Global
{
unsigned debug_level = 0;
}

void Shard::initialize_components(const std::string& backend,
                                  const std::string& index,
                                  const std::string& pci_addr,
                                  const std::string& dax_config,
                                  unsigned           debug_level)
{
  using namespace Component;

  /* STORE */
  {
    IBase* comp;

    if (backend == "pmstore")
      comp = load_component("libcomanche-pmstore.so", pmstore_factory);
    else if (backend == "mapstore")
      comp = load_component("libcomanche-storemap.so", mapstore_factory);
    else if (backend == "hstore")
      comp = load_component("libcomanche-hstore.so", hstore_factory);
    else if (backend == "nvmestore")
      comp = load_component("libcomanche-nvmestore.so", nvmestore_factory);
    else if (backend == "filestore")
      comp = load_component("libcomanche-storefile.so", filestore_factory);
    else if (backend == "dummystore")
      comp = load_component("libcomanche-dummystore.so", dummystore_factory);
    else
      throw General_exception("invalid backend (%s)", backend.c_str());

    if (option_DEBUG > 2)
      PLOG("Shard: using store backend (%s)", backend.c_str());

    if (!comp)
      throw General_exception("unable to initialize Dawn backend component");

    IKVStore_factory* fact =
        (IKVStore_factory*) comp->query_interface(IKVStore_factory::iid());
    assert(fact);

    if (backend == "hstore") {
      if (dax_config.empty())
        throw General_exception("hstore backend requires dax configuration");

      _i_kvstore = fact->create("owner", "name", dax_config);
    }
    else if (backend == "nvmestore") {
      if (pci_addr.empty())
        throw General_exception(
            "nvmestore backend needs pci device configuration");

      _i_kvstore = fact->create("owner", "name", pci_addr);
    }
    else if (backend == "pmstore") { /* components that support debug level */
      _i_kvstore = fact->create(debug_level, "owner", "name", "");
    }
    else {
      _i_kvstore = fact->create("owner", "name");
    }
    fact->release_ref();
  }
  /* INDEX */
  {
    IBase* comp;
    if (index == "rbtree") {
      comp = load_component("libcomanche-indexrbtree.so", rbtreeindex_factory);
      if (!comp)
        throw General_exception("unable to load libcomanche-indexrbtree.so");
      _index_factory = static_cast<IKVIndex_factory*>(comp->query_interface(IKVIndex_factory::iid()));
      assert(_index_factory);
    }    
  }
}

void Shard::main_loop()
{
  using namespace Dawn::Protocol;

  assert(_i_kvstore);

#ifdef PROFILE
  ProfilerStart("shard_main_loop");
#endif

  uint64_t                  tick __attribute__((aligned(8))) = 0;
  static constexpr uint64_t CHECK_CONNECTION_INTERVAL        = 1000000;

  Connection_handler::action_t                            action;
  std::vector<std::vector<Connection_handler*>::iterator> pending_close;

  while (_thread_exit == false) {
    /* check for new connections - but not too often */
    if (tick % CHECK_CONNECTION_INTERVAL == 0)
      check_for_new_connections();

    /* if there are no sessions, sleep thread */
    if(_handlers.empty()) {
      usleep(500000);
      check_for_new_connections();
    }
    else {
    
      /* iterate connection handlers (each connection is a client session) */
      for (std::vector<Connection_handler*>::iterator handler_iter =
             _handlers.begin();
           handler_iter != _handlers.end(); handler_iter++) {
        const auto handler = *handler_iter;

        /* issue tick, unless we are stalling */
        uint64_t tick_response;
        if(handler->stall_tick() == 0)
          tick_response = handler->tick();
        else continue;
    
        /* close session */
        if (tick_response == Dawn::Connection_handler::TICK_RESPONSE_CLOSE) {
          if (option_DEBUG > 1) PMAJOR("Shard: closing connection %p", handler);
          pending_close.push_back(handler_iter);
        }

        /* process ALL deferred actions */
        while (handler->get_pending_action(action)) {
          switch (action.op) {
          case Connection_handler::ACTION_RELEASE_VALUE_LOCK:
            if (option_DEBUG > 2)
              PLOG("releasing value lock (%p)", action.parm);
            release_locked_value(action.parm);
            break;
          default:
            throw Logic_exception("unknown action type");
          }
        }

        /* collect ALL available messages */
        buffer_t*          iob;
        Protocol::Message* p_msg = nullptr;
        while ((iob = handler->get_pending_msg(p_msg)) != nullptr) {
          assert(p_msg);
          switch (p_msg->type_id) {
          case MSG_TYPE_IO_REQUEST:
            process_message_IO_request(handler, static_cast<Protocol::Message_IO_request*>(p_msg));
            break;
          case MSG_TYPE_POOL_REQUEST:
            process_message_pool_request(handler, static_cast<Protocol::Message_pool_request*>(p_msg));
            break;
          default:
            throw General_exception("unrecognizable message type");
          }
          handler->free_recv_buffer();
        }
      }  // handler iter

    
      /* handle pending close sessions */
      if (unlikely(!pending_close.empty())) {
        for (auto& h : pending_close) {
          if (option_DEBUG > 1) PLOG("Deleting handler (%p)", *h);
          delete *h;
          _handlers.erase(h);

          if (option_DEBUG > 1)
            PLOG("# remaining handlers (%lu)", _handlers.size());
        }
        pending_close.clear();
      }
    }

    tick++;
  }

  if (option_DEBUG > 1) PMAJOR("Shard (%p) exited", this);

#ifdef PROFILE
  ProfilerStop();
  ProfilerFlush();
#endif
}

void Shard::process_message_pool_request(Connection_handler* handler,
                                         Protocol::Message_pool_request* msg)
{
  // validate auth id
  assert(msg->op);

  /* allocate response buffer */
  auto response_iob = handler->allocate();
  assert(response_iob);
  assert(response_iob->base());
  memset(response_iob->iov->iov_base, 0, response_iob->iov->iov_len);

  Protocol::Message_pool_response* response = new (response_iob->base())
      Protocol::Message_pool_response(handler->auth_id());
  assert(response);
  response->status = S_OK;

  /* handle operation */
  if (msg->op == Dawn::Protocol::OP_CREATE) {
    if (option_DEBUG > 1)
      PMAJOR("POOL CREATE: op=%u name=%s size=%lu obj-count=%lu", msg->op,
             msg->pool_name(), msg->pool_size,
             msg->expected_object_count);

    const std::string pool_name = msg->pool_name();

    Component::IKVStore::pool_t pool;

    if (handler->check_for_open_pool(pool_name, pool)) {
      handler->add_reference(pool);
    }
    else {
      pool = _i_kvstore->create_pool(msg->pool_name(),
                                     msg->pool_size,
                                     msg->flags,
                                     msg->expected_object_count);

      if(pool == Component::IKVStore::POOL_ERROR) {
        response->pool_id = 0;
        response->status = Component::IKVStore::POOL_ERROR;
        PWRN("unable to create pool (%s)", pool_name.c_str());
      }
      else {
        /* register pool handle */
        handler->register_pool(pool_name, pool);
        response->pool_id = pool;
        response->status  = S_OK;
      }

      if (option_DEBUG > 2) PLOG("OP_CREATE: new pool id: %lx", pool);
    }
  }
  else if (msg->op == Dawn::Protocol::OP_OPEN) {
    if (option_DEBUG > 1)
      PMAJOR("POOL OPEN: name=%s", msg->pool_name());

    Component::IKVStore::pool_t pool;
    const std::string pool_name(msg->pool_name());
    
    /* check that pool is not already open */
    if (handler->check_for_open_pool(pool_name, pool)) {
      PLOG("reusing existing open pool (%p)", (void*) pool);
      /* pool exists, increment reference */
      handler->add_reference(pool);
      response->pool_id = pool;
    }
    else {
        /* pool does not exist yet */
      pool = _i_kvstore->open_pool(msg->pool_name());

      if(pool == Component::IKVStore::POOL_ERROR) {
        response->pool_id = 0;
        response->status = E_INVAL;
      }
      else {
        /* register pool handle */
        handler->register_pool(pool_name, pool);
        response->pool_id = pool;
      }
    }
    if (option_DEBUG > 2) PLOG("OP_OPEN: pool id: %lx", pool);    
  }
  else if (msg->op == Dawn::Protocol::OP_CLOSE) {
    if (option_DEBUG > 1) PMAJOR("POOL CLOSE: pool_id=%lx", msg->pool_id);

    /* release reference, if its zero, we can close pool for real */
    if(handler->release_pool_reference(msg->pool_id)) {
      PLOG("actually closing pool %p", (void*) msg->pool_id);
      response->status = _i_kvstore->close_pool(msg->pool_id);
      assert(response->status == S_OK);
    }
    else {
      response->status = S_OK;
    }
  }
  else if (msg->op == Dawn::Protocol::OP_DELETE) {

    if (option_DEBUG > 2) PMAJOR("POOL DELETE: pool_id=%lx name=%s", msg->pool_id, msg->pool_name());
    
    Component::IKVStore::pool_t pool;
    const std::string pool_name = msg->pool_name();

    /* check if pool is still open; return error if it is */
    if (handler->check_for_open_pool(pool_name, pool)) {
      response->pool_id = 0;
      response->status = Component::IKVStore::E_ALREADY_OPEN;
    }
    else {
      response->pool_id = 0;
      response->status = _i_kvstore->delete_pool(msg->pool_name());
    }
  }
  else
    throw Protocol_exception("process_message_pool_request - bad operation (msg->op = %d)", msg->op);

 send_response:
  /* trim response length */
  response_iob->set_length(response->msg_len);

  /* finally, send response */
  handler->post_response(response_iob);
}




void Shard::process_message_IO_request(Connection_handler*           handler,
                                       Protocol::Message_IO_request* msg)
{
  using namespace Component;
  int status = S_OK;

  /////////////////////////////////////////////////////////////////////////////
  //   PUT ADVANCE   //
  /////////////////////
  if (msg->op == Protocol::OP_PUT_ADVANCE) {
    if (option_DEBUG > 2)
      PLOG("PUT_ADVANCE: (%p) key=(%.*s) value_len=%lu request_id=%lu", this,
           (int) msg->key_len, msg->key(), msg->val_len, msg->request_id);

    const auto val_len = msg->val_len;

    /* open memory */
    void*  target     = nullptr;
    size_t target_len = msg->val_len;
    assert(target_len > 0);
    assert(msg->pool_id > 0);

    /* can't support dont stomp flag */
    if(msg->flags & IKVStore::FLAGS_DONT_STOMP) {
      status = E_INVAL;
      goto send_response;
    }

    std::string k(msg->key(), msg->key_len);
    /* create (if needed) and lock value */    
    auto key_handle = _i_kvstore->lock(msg->pool_id,
                                       k,
                                       IKVStore::STORE_LOCK_WRITE,
                                       target,
                                       target_len);

    if (key_handle == Component::IKVStore::KEY_NONE) {
      PWRN("PUT_ADVANCE failed to lock value (lock() returned KEY_NONE) ");
      status = E_INVAL;
      goto send_response;
    }

    if (target_len != msg->val_len)
      throw Logic_exception("locked value length mismatch");

    auto pool_id = msg->pool_id;
    add_locked_value(pool_id, key_handle, target);

    /* register memory unless pre-registered */
    Connection_base::memory_region_t region = nullptr;  // TODO handler->get_preregistered(pool_id);

    if (option_DEBUG > 2) {
      PLOG("registering locked value: %p %lu", target, target_len);
    }
  
      
    // if(!region) {
    //   if(option_DEBUG || 1)
    //     PLOG("using ondemand registration");
    //   region = ondemand_register(handler, target, target_len);
    // }
    // else {
    //   if(option_DEBUG > 2)
    //     PLOG("using pre-registered region (handle=%p)", region);
    // }
    // assert(region);
    // TODO
    region = handler->ondemand_register(target, target_len);

    handler->set_pending_value(target, target_len, region);

    return;
  }
  
 send_response:
  /* states that we require a response */
  const auto iob = handler->allocate();

  Protocol::Message_IO_response* response = new (iob->base())
      Protocol::Message_IO_response(iob->length(), handler->auth_id());

  /////////////////////////////////////////////////////////////////////////////
  //   PUT           //
  /////////////////////
  if (msg->op == Protocol::OP_PUT) {
    /* for basic 'puts' we have to do a memcpy - to support "in-place"
       puts for larger data, we use a two-stage operation
    */
    if (option_DEBUG > 2)
      PLOG("PUT: (%p) key=(%.*s) value=(%.*s)", this, (int) msg->key_len,
           msg->key(), (int) msg->val_len, msg->value());

    if (unlikely(msg->resvd & Dawn::Protocol::MSG_RESVD_SCBE)) {
      status = S_OK;  // short-circuit backend
      if (option_DEBUG > 2) PLOG("PUT: short-circuited backend");
    }
    else {
      const std::string k(msg->key(), msg->key_len);

      status = _i_kvstore->put(msg->pool_id,
                               k,
                               msg->value(),
                               msg->val_len,
                               msg->flags);

      if (option_DEBUG > 2) {
        if (status == Component::IKVStore::E_ALREADY_EXISTS)
          PLOG("kvstore->put returned E_ALREADY_EXISTS");
        else
          PLOG("kvstore->put returned %d", status);
      }
    }
  }
  /////////////////////////////////////////////////////////////////////////////
  //   GET           //
  /////////////////////
  else if (msg->op == Protocol::OP_GET) {
    if (option_DEBUG > 2)
      PMAJOR("GET: (%p) (request=%lu) key=(%.*s) ", this, msg->request_id,
             (int) msg->key_len, msg->key());

    if (msg->resvd & Dawn::Protocol::MSG_RESVD_SCBE) {
      if (option_DEBUG > 2) PLOG("GET: short-circuited backend");
      response->data_len   = 0;
      response->request_id = msg->request_id;
      response->status     = S_OK;
      iob->set_length(response->base_message_size());
      handler->post_response(iob);
    }
    else {
      void*  value_out     = nullptr;
      size_t value_out_len = 0;

      std::string k(msg->key(), msg->key_len);

      auto key_handle = _i_kvstore->lock(msg->pool_id,
                                         k,
                                         IKVStore::STORE_LOCK_READ,
                                         value_out,
                                         value_out_len);

      if (option_DEBUG > 2)
        PLOG("shard: locked OK: value_out=%p (%.*s) value_out_len=%lu",
             value_out, (int) value_out_len, (char*) value_out, value_out_len);

      if (key_handle == Component::IKVStore::KEY_NONE) { /* key not found */
        response->status = E_NOT_FOUND;
        iob->set_length(response->base_message_size());
        handler->post_response(iob, nullptr);
        return;
      }

      assert(value_out_len);
      assert(value_out);

      if (value_out_len <
          (iob->original_length - response->base_message_size())) {
        /* value can fit in message buffer, let's copy instead of
           performing two-part DMA */
        if (option_DEBUG > 2) PLOG("shard: performing memcpy for small get");

        iob->set_length(response->base_message_size() + value_out_len);
        memcpy(response->data, value_out, value_out_len);

        _i_kvstore->unlock(msg->pool_id, key_handle);

        response->data_len   = value_out_len;
        response->request_id = msg->request_id;
        response->status     = S_OK;

        handler->post_response(iob);
      }
      else {
        iob->set_length(response->base_message_size());

        /* register memory on-demand */
        auto region = handler->ondemand_register(value_out, value_out_len);
        assert(region);

        response->data_len   = value_out_len;
        response->request_id = msg->request_id;
        response->status     = S_OK;

        /* create value iob */
        buffer_t* value_buffer;

        if (response->status == S_OK) {
          value_buffer         = new buffer_t(value_out_len);
          value_buffer->iov    = new iovec{(void*) value_out, value_out_len};
          value_buffer->region = region;
          value_buffer->desc   = handler->get_memory_descriptor(region);

          /* register clean up task for value */
          add_locked_value(msg->pool_id, key_handle, value_out);
        }

        if (value_out_len <=
            (handler->IO_buffer_size() - response->base_message_size())) {
          if (option_DEBUG > 2)
            PLOG("posting response header and value together");

          /* post both buffers together in same response packet */
          handler->post_response(iob, value_buffer);
        }
        else {
          /* for large gets we use a two-stage protocol sending
             response message and value separately
          */
          response->set_twostage_bit();

          /* send two separate packets */
          handler->post_response(iob);

          /* the client is allocating the recv buffer only after
             receiving the response advance. this could timeout if
             this side issues before the remote buffer is ready */
          handler->post_send_value_buffer(value_buffer);
        }
      }
    }
    return;
  }
  /////////////////////////////////////////////////////////////////////////////
  //   ERASE         //
  /////////////////////
  else if (msg->op == Protocol::OP_ERASE) {
    std::string k(msg->key(), msg->key_len);
    
    status = _i_kvstore->erase(msg->pool_id, k);
  }
  else {
    throw Protocol_exception("operation not implemented");
  }

  response->request_id = msg->request_id;
  response->status     = status;

  iob->set_length(response->msg_len);
  handler->post_response(iob);  // issue IO request response
}

void Shard::check_for_new_connections()
{
  /* new connections are transferred from the connection handler
     to the shard thread */
  Connection_handler* handler;

  while ((handler = get_new_connection()) != nullptr) {
    if (option_DEBUG > 1)
      PMAJOR("Shard: processing new connection (%p)", handler);
    _handlers.push_back(handler);
  }
}

}  // namespace Dawn
