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
#include <nupm/mcas_mod.h>
static long MsgType_Proxy2Ado = 0xBEEF;
static long MsgType_Ado2Proxy = 0xFEED;


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
                                   message);
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
    PLOG("Launching ADO path: (%s)", _default_ado_path.c_str());
    ado = _i_ado_mgr->create(_default_ado_path, args, 0);
    PLOG("ADO process launched OK.");
    assert(ado);
    _ado_map[msg->pool_id] = ado;
  }
  else {
    ado = i->second;
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
        PLOG("Shard: exposed region: %p %lu", r.iov_base, r.iov_len);
      }
    }
  } /* end of bootstrap */

  /* get key-value pair */ 
  void * value = nullptr;
  size_t value_len = 0;
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
  PLOG("Shard_ado: locked KV pair (value=%p, value_len=%lu)", value, value_len);

  /* register outstanding work */
  auto wr = new work_request_t { msg->pool_id, key_handle, locktype };
  _outstanding_work.insert(wr);

  /* now send the work request */
  ado->send_work_request(wr, value, value_len, msg->command());

  /* finally, send a response.  We don't wait around for the
     work to finish.  This would drag the shard thread */
  response = new (response_iob->base())
    Protocol::Message_ado_response(response_iob->length(),
                                   handler->auth_id(),
                                   msg->request_id,
                                   response_string);

  response->status = S_OK;
  response_iob->set_length(response->message_size());
  handler->post_send_buffer(response_iob);
}
