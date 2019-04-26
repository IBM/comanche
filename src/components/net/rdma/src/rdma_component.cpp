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
#include <iostream>
#include "rdma_component.h"

Rdma_component::Rdma_component(const std::string& device_name) : _device_name(device_name)
{
  _transport = new Rdma_transport;
}

Rdma_component::~Rdma_component()
{
}

status_t Rdma_component::connect(const std::string& peer_name, int port)
{
  if(_device_name == "any")
    return _transport->connect(NULL, peer_name.c_str(), port);
  else
    return _transport->connect(_device_name.c_str(), peer_name.c_str(), port);
}

status_t Rdma_component::wait_for_connect(int port)
{
  if(_device_name == "any")
    return _transport->wait_for_connect(NULL, port);
  else
    return _transport->wait_for_connect(_device_name.c_str(), port);
}

status_t Rdma_component::disconnect()
{
  PLOG("RDMA: recreating transport object");
  delete _transport;
  _transport = new Rdma_transport;
  return S_OK;
}


struct ibv_mr * Rdma_component::register_memory(void * contig_addr, size_t size)
{
  return _transport->register_memory(contig_addr,size);
}

void Rdma_component::post_send(uint64_t gwid, struct ibv_mr * mr0, struct ibv_mr * extra_mr)
{
  if(_transport->post_send(gwid, mr0, extra_mr) != S_OK)
    throw General_exception("rdma transport post_send failed");
}

void Rdma_component::post_recv(uint64_t gwid, struct ibv_mr * mr0)
{
  if(_transport->post_recv(gwid, mr0) != S_OK)
    throw General_exception("rdma transport post_recv failed");
}

int Rdma_component::poll_completions(std::function<void(uint64_t)> completion_func)
{
  return _transport->poll_completions(completion_func);
}

uint64_t Rdma_component::wait_for_next_completion(unsigned timeout_polls)
{
  return _transport->wait_for_next_completion(timeout_polls);
}

/** 
 * Factory entry point
 * 
 */
extern "C" void * factory_createInstance(Component::uuid_t& component_id)
{
  if(component_id == Rdma_component_factory::component_id()) {
    return static_cast<void*>(new Rdma_component_factory());
  }
  else return NULL;
}

