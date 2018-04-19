#include <iostream>
#include "rdma_component.h"

Rdma_component::Rdma_component(const std::string& device_name) : _device_name(device_name)
{
}

Rdma_component::~Rdma_component()
{
}

status_t Rdma_component::connect(const std::string& peer_name, int port)
{
  if(_device_name == "any")
    return _transport.connect(NULL, peer_name.c_str(), port);
  else
    return _transport.connect(_device_name.c_str(), peer_name.c_str(), port);
}

status_t Rdma_component::wait_for_connect(int port)
{
  if(_device_name == "any")
    return _transport.wait_for_connect(NULL, port);
  else
    return _transport.wait_for_connect(_device_name.c_str(), port);
}

status_t Rdma_component::disconnect()
{
  return E_FAIL;
}


struct ibv_mr * Rdma_component::register_memory(void * contig_addr, size_t size)
{
  return _transport.register_memory(contig_addr,size);
}

void Rdma_component::post_send(uint64_t gwid, struct ibv_mr * mr0, struct ibv_mr * extra_mr)
{
  if(_transport.post_send(gwid, mr0, extra_mr) != S_OK)
    throw General_exception("rdma transport post_send failed");
}

void Rdma_component::post_recv(uint64_t gwid, struct ibv_mr * mr0)
{
  if(_transport.post_recv(gwid, mr0) != S_OK)
    throw General_exception("rdma transport post_recv failed");
}

int Rdma_component::poll_completions(std::function<void(uint64_t)> completion_func)
{
  return _transport.poll_completions(completion_func);
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

