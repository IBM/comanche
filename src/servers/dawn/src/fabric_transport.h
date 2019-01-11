#ifndef __FABRIC_TRANSPORT_H__
#define __FABRIC_TRANSPORT_H__

#include <api/components.h>
#include <api/fabric_itf.h>

namespace Dawn
{
class Connection_handler;

class Fabric_transport {
  bool option_DEBUG;

 public:
  static constexpr unsigned INJECT_SIZE = 128;

  using memory_region_t = Component::IFabric_memory_region*;
  using buffer_t = Buffer_manager<Component::IFabric_server>::buffer_t;

  Fabric_transport(const std::string provider, const std::string device, unsigned port) {
    option_DEBUG = Dawn::Global::debug_level > 1;

    if (option_DEBUG) PLOG("fabric_transport: (provider=%s, device=%s, port=%u)", provider.c_str(), device.c_str(), port);

    init(provider, device, port);
  }

  Connection_handler* get_new_connection() {
    auto connection = _server_factory->get_new_connection();
    if (!connection) return nullptr;
    return new Connection_handler(_server_factory, connection);
  }

 private:
  void init(const std::string& provider, const std::string& device, unsigned port) {
    using namespace Component;

    if (option_DEBUG) PLOG("Fabric: bound to device (%s)", device.c_str());

    /* FABRIC */
    auto i_fabric_factory = static_cast<IFabric_factory*>(load_component("libcomanche-fabric.so", net_fabric_factory));

    if (!i_fabric_factory) throw General_exception("unable to load Fabric Comanche component");

    /* The libfabric 1.6 sockets provider requires a FI_MR_BASIC specfication
     * even though FI_MR_BASIC is supposedly obsolete after libfabric 1.4.
     */
    const std::string mr_mode = provider == "sockets" ? "[ \"FI_MR_BASIC\" ]"
                                                      : "[ \"FI_MR_LOCAL\", \"FI_MR_VIRT_ADDR\", "
                                                        "\"FI_MR_ALLOCATED\", \"FI_MR_PROV_KEY\" ]";

    const std::string fabric_spec{"{ \"fabric_attr\" : { \"prov_name\" : \"" + provider +
                                  "\" }"
                                  ","
                                  " \"domain_attr\" : { \"mr_mode\" : " +
                                  mr_mode + " , \"name\" : \"" + device +
                                  "\" }"
                                  ","
                                  " \"tx_attr\" : { \"inject_size\" : " +
                                  std::to_string(INJECT_SIZE) +
                                  " }"
                                  ","
                                  " \"ep_attr\" : { \"type\" : \"FI_EP_MSG\" }"
                                  "}"};

    _fabric = i_fabric_factory->make_fabric(fabric_spec);

    const std::string server_factory_spec{"{}"};
    /* ERROR: hard-coded port */
    _server_factory = _fabric->open_server_factory(server_factory_spec, port);

    i_fabric_factory->release_ref();
  }

  Component::IFabric* _fabric;
  Component::IFabric_server_factory* _server_factory;
};

}  // namespace Dawn

#endif  // __FABRIC_TRANSPORT_H__
