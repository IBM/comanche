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

void Shard::process_ado_request(Connection_handler* handler,
                                Protocol::Message_ado_request* msg)
{
  auto response_iob = handler->allocate();
  assert(response_iob);
  Protocol::Message_ado_response* response;
  
  if(!_i_ado_mgr) {
    PWRN("ADO request ignored. Not enabled");
    response = new (response_iob->base())
    Protocol::Message_ado_response(response_iob->length(),
                                   handler->auth_id(),
                                   msg->request_id,
                                   "ADO!NOT_ENABLED");
    response->status = E_INVAL;
    response_iob->set_length(response->message_size());
    handler->post_send_buffer(response_iob);
    return;
  }

  std::string response_string = "ADO!OK";
 

  /* ADO processes are instantiated on a per-pool basis.  First
     check if an ADO process already exists.
  */
  Component::IADO_proxy * ado;
  auto i = _ado_map.find(msg->pool_id);
  if(i == _ado_map.end()) {
    /* need to launch new ADO process */
    std::vector<std::string> args;
    PLOG("Launching ADO path: (%s)", _default_ado_path.c_str());
    ado = _i_ado_mgr->create(_default_ado_path,args,0,0);
    assert(ado);
    _ado_map[msg->pool_id] = ado;
  }
  else {
    ado = i->second;
  }

  // if(msg->command().begins_with("ADO!POST")) {
  //   ado->post_work(IADO_proxy::Op_type::DOMAIN_OPERATION,
  //                  msg->command
  // }
  
  response = new (response_iob->base())
    Protocol::Message_ado_response(response_iob->length(),
                                   handler->auth_id(),
                                   msg->request_id,
                                   response_string);

  response->status = S_OK;
  response_iob->set_length(response->message_size());
  handler->post_send_buffer(response_iob);
  PLOG("ADO dummy response sent.");
}
