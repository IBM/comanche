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

/*< #included in shard.cpp */
#include <api/ado_itf.h>
#include <nupm/mcas_mod.h>

void Shard::process_ado_request(Connection_handler* handler,
                                Protocol::Message_ado_request* msg)
{
  auto response_iob = handler->allocate();
  assert(response_iob);
  Protocol::Message_ado_response* response;

  auto error_func = [&](const char * message) {
    response = new (response_iob->base())
    Protocol::Message_ado_response(response_iob->length(),
                                   handler->auth_id(),
                                   msg->request_id,
                                   (void*) message,
                                   strlen(message));
    response->status = E_INVAL;
    response_iob->set_length(response->message_size());
    handler->post_send_buffer(response_iob);
  };
  
  if(!_i_ado_mgr) {
    error_func("ADO!NOT_ENABLED");
    return;
  }

  std::string response_string = "ADO!OK";

  /* ADO processes are instantiated on a per-pool basis.  First
     check if an ADO process already exists.
  */
  Component::IADO_proxy * ado;
  bool bootstrap = true;
  auto i = _ado_map.find(msg->pool_id);
  if(i == _ado_map.end()) {
    /* need to launch new ADO process */
    std::vector<std::string> args;
    args.push_back("--plugin");
    args.push_back(_default_ado_plugin);

    if (option_DEBUG > 2)
      PLOG("Launching ADO path: (%s), plugin (%s)", _default_ado_path.c_str(), _default_ado_plugin.c_str());
    
    ado = _i_ado_mgr->create(_default_ado_path, args, 0);

    if (option_DEBUG > 2)
      PLOG("ADO process launched OK.");
    
    assert(ado);
    _ado_map[msg->pool_id] = std::make_pair(ado, handler);
  }
  else {
    ado = i->second.first;
    bootstrap = false;
  }

  /* conditionally bootstrap ADO */
  if(bootstrap) {
    if(ado->bootstrap_ado() != S_OK) {
      error_func("ADO!BOOTSTRAP=FAILED");
      return;
    }

    /* exchange memory mapping information */
    if(nupm::check_mcas_kernel_module()) {

      std::vector<::iovec> regions;
      if(_i_kvstore->get_pool_regions(msg->pool_id, regions) != S_OK) {
        error_func("ADO!GET_POOL_REGIONS=FAILED");
        return;
      }

      for(auto& r: regions) {

        /* expose memory - for the moment use the address as the token */
        uint64_t token = (uint64_t) r.iov_base;

        nupm::revoke_memory(token); /*remove any existing registrations */
        if(nupm::expose_memory(token,
                               r.iov_base,
                               r.iov_len) != S_OK) {
          PWRN("Shard: failed to expose memory to ADO");
          continue;
        }
        ado->send_memory_map(token, r.iov_len, r.iov_base);

        if (option_DEBUG > 2)
          PLOG("Shard: exposed region: %p %lu", r.iov_base, r.iov_len);
      }
    }
  } /* end of bootstrap */

  /* get key-value pair */ 
  void * value = nullptr;
  size_t value_len = msg->ondemand_val_len;

  assert(value_len > 0);
  
  Component::IKVStore::key_t key_handle;
  auto locktype = Component::IKVStore::STORE_LOCK_WRITE;  
  if(_i_kvstore->lock(msg->pool_id,
                      msg->key(),
                      locktype,
                      value,
                      value_len,
                      key_handle) != S_OK)                 
  {
    error_func("ADO!ALREADY_LOCKED");
    return;
  }

  if(key_handle == Component::IKVStore::KEY_NONE)
    throw Logic_exception("lock gave KEY_NONE");

  if (option_DEBUG > 2)
    PLOG("Shard_ado: locked KV pair (value=%p, value_len=%lu)", value, value_len);

  /* register outstanding work */
  auto wr = new work_request_t { msg->pool_id, key_handle, locktype, msg->request_id };
  auto wr_key = reinterpret_cast<work_request_key_t>(wr);
  _outstanding_work.insert(wr_key);

  /* now send the work request */
  ado->send_work_request(wr_key, msg->key(), value, value_len, msg->request(), msg->request_len());

  if (option_DEBUG > 2)
    PLOG("Shard_ado: sent work request (len=%lu, key=%lu)", msg->request_len(), wr_key);
 
  handler->free_buffer(response_iob);

  /* we don't send a response to the client until the work completion
     has been picked up.  Of course this gives synchronous semantics
     on the client side.  We may need to extend this to asynchronous
     semantics for longer ADO operations */
  
}
  

/** 
 * Handle messages coming back from the ADO process
 * 
 */
void Shard::process_messages_from_ado()
{
  for(auto record: _ado_map) {
    Component::IADO_proxy* ado = record.second.first;
    Connection_handler * handler = record.second.second;

    assert(ado);
    assert(handler);

    void * response = nullptr;
    size_t response_len = 0;
    work_request_key_t request_key = 0;
    while(ado->check_work_completions(request_key, response, response_len) > 0) {

      if(_outstanding_work.count(request_key) == 0)
        throw General_exception("bad record key from ADO (%lu)", request_key);

      auto request_record = request_key_to_record(request_key);
      PMAJOR("Shard: collected WORK completion (request_record=%p)", request_record);

      _outstanding_work.erase(request_key);

      /* unlock the KV pair */
      if( _i_kvstore->unlock(request_record->pool,
                             request_record->key_handle) != S_OK)
        throw Logic_exception("unlock for KV after ADO work completion failed");

      if (option_DEBUG > 2)
        PLOG("Unlocked KV pair (pool=%lx, key_handle=%p)", request_record->pool, request_record->key_handle);
      
      /* send response to client */
      {
        auto iob = handler->allocate();

        assert(iob);
        Protocol::Message_ado_response* response_msg;

        response_msg = new (iob->base())
          Protocol::Message_ado_response(iob->length(),
                                         handler->auth_id(),
                                         request_record->request_id, // TODOmsg->request_id,
                                         response,
                                         response_len);

        response_msg->status = S_OK;
        iob->set_length(response_msg->message_size());
        handler->post_send_buffer(iob);
      }
      
      ::free(response);
    }
  }
}            
