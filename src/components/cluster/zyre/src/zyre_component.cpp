#include <iostream>
#include "zyre_component.h"

Zyre_component::Zyre_component(const std::string& node_name, const std::string& end_point)
{
  uint64_t version = zyre_version();
  assert ((version / 10000) % 100 == ZYRE_VERSION_MAJOR);
  assert ((version / 100) % 100 == ZYRE_VERSION_MINOR);
  assert (version % 100 == ZYRE_VERSION_PATCH);

  _node = zyre_new(node_name.c_str());
  if(_node == nullptr)
    throw General_exception("zyre_new failed unexpectedly.");

  if(option_DEBUG) {
    PLOG("Zyre: version %f", version/10000.0f);
    zyre_set_verbose(_node);
  }

  if(!end_point.empty())
    if(zyre_set_endpoint(_node, "%s", end_point.c_str())!=0)
      throw General_exception("unable to set end point (%s)", end_point.c_str());
  
  zyre_set_interval(_node, HEARTBEAT_INTERVAL_MS);
  zyre_set_header(_node, "X-ZYRE-COMANCHE", "1");
}

Zyre_component::~Zyre_component()
{
  zyre_destroy(&_node);
}


void Zyre_component::start_node()
{
  zyre_start(_node);
}

void Zyre_component::stop_node()
{
  zyre_stop(_node);
}

void Zyre_component::group_join(const std::string& group)
{
  if(zyre_join(_node, group.c_str()))
    throw General_exception("zyre_join failed");
}

void Zyre_component::group_leave(const std::string& group)
{
  if(zyre_leave(_node, group.c_str()))
    throw General_exception("zyre_join failed");
}

void Zyre_component::shout(const std::string& group,
                           const std::string& type,
                           const std::string& message)
{
  zmsg_t * msg = zmsg_new();
  zmsg_addstr(msg, zyre_uuid(_node)); /* frame0 is uuid of sender */
  //  zmsg_addstr(msg, zyre_name(_node));
  zmsg_addstr(msg, type.c_str());
  zmsg_addstr(msg, message.c_str());
  
  if(zyre_shout(_node, group.c_str(), &msg)) /* destroy message after sending */
    throw General_exception("zyre_shout failed");
}

void Zyre_component::whisper(const std::string& peer_uuid,
                             const std::string& type,
                             const std::string& message)
{
  zmsg_t * msg = zmsg_new();
  zmsg_addstr(msg, zyre_uuid(_node)); /* frame0 is uuid of sender */
  //  zmsg_addstr(msg, zyre_name(_node));
  zmsg_addstr(msg, type.c_str());
  zmsg_addstr(msg, message.c_str());

  if(zyre_whisper(_node, peer_uuid.c_str(), &msg)) /* destroy message after sending */
    throw General_exception("zyre_whisper failed");
}

void Zyre_component::poll_recv(std::function<void(const std::string& sender_uuid,
                                                  const std::string& type,
                                                  const std::string& message)> callback)
{
  zmsg_t * msg = nullptr;
  auto socket = zyre_socket(_node);
  while((msg = zmsg_recv_nowait(socket))) {
    assert(zmsg_is(msg));
    
    /* expect same framing as sending routines */
    assert(zmsg_size(msg)==3);
    
    char * uuid = zmsg_popstr(msg);
    assert(uuid);
    std::string sender = uuid;
    free(uuid);

    char * type = zmsg_popstr(msg);
    assert(type);
    std::string msgtype = type;
    free(type);


    char * body = zmsg_popstr(msg);
    assert(body);
    std::string msgstr = body;    
    free(body);

    callback(sender, msgtype, msgstr); /* invoke callback */
    zmsg_destroy(&msg);
  }
  
}


std::string Zyre_component::uuid() const
{
  return std::string(zyre_uuid(_node));
}

std::string Zyre_component::node_name() const
{
  return std::string(zyre_name(_node));
}

void Zyre_component::dump_info() const
{
  zyre_dump(_node);
}

/** 
 * Factory entry point
 * 
 */
extern "C" void * factory_createInstance(Component::uuid_t& component_id)
{
  if(component_id == Zyre_component_factory::component_id()) {
    return static_cast<void*>(new Zyre_component_factory());
  }
  else return NULL;
}

