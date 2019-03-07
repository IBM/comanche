#include "experiment.h"

#define PMSTORE_PATH "libcomanche-pmstore.so"
#define FILESTORE_PATH "libcomanche-storefile.so"
#define NVMESTORE_PATH "libcomanche-nvmestore.so"
#define ROCKSTORE_PATH "libcomanche-rocksdb.so"
#define DAWN_PATH "libcomanche-dawn-client.so"

int Experiment::initialize_store(unsigned core)
{
  using namespace Component;

  IBase * comp;

  try
  {
    if( component_is( "pmstore" ) ) {
      comp = load_component(PMSTORE_PATH, pmstore_factory);
    }
    else if( component_is( "filestore" ) ) {
      comp = load_component(FILESTORE_PATH, filestore_factory);
    }
    else if( component_is( "nvmestore" ) ) {
      comp = load_component(NVMESTORE_PATH, nvmestore_factory);
    }
    else if( component_is( "rockstore" ) ) {
      comp = load_component(ROCKSTORE_PATH, rocksdb_factory);
    }
    else if( component_is( "dawn" ) ) {

      DECLARE_STATIC_COMPONENT_UUID(dawn_factory, 0xfac66078,0xcb8a,0x4724,0xa454,0xd1,0xd8,0x8d,0xe2,0xdb,0x87);  // TODO: find a better way to register arbitrary components to promote modular use
      comp = load_component(DAWN_PATH, dawn_factory);
    }
    else if ( component_is( "hstore" ) ) {
      comp = load_component("libcomanche-hstore.so", hstore_factory);
    }
    else if ( component_is( "mapstore" ) ) {
      comp = load_component("libcomanche-storemap.so", mapstore_factory);
    }
    else throw General_exception("unknown --component option (%s)", _component.c_str());
  }
  catch ( const Exception &e )
  {
    PERR("error during load_component: %s. Aborting experiment.", e.cause());
    throw;
  }
  catch ( const std::exception &e )
  {
    PERR("error during load_component: %s. Aborting experiment.", e.what());
    throw;
  }
  catch(...)
  {
    PERR("%s", "error during load_component.");
    return 1;
  }

  if (_verbose)
    {
      PINF("[%u] component address: %p", core, static_cast<const void *>(&comp));
    }

  if (!comp)
  {
    PERR("%s", "comp loaded, but returned invalid value");
    return 1;
  }

  try
  {
    IKVStore_factory * fact = static_cast<IKVStore_factory *>(comp->query_interface(IKVStore_factory::iid()));

    if( component_is( "nvmestore" ) ) {
      _store = fact->create("owner",_owner, _pci_address);
    }
    else if ( component_is( "dawn" ) ) {
      std::stringstream url;
      auto port = _port;
      if(_port_increment > 0) {
        port += (_get_core_index(core) / _port_increment);
      }
      url << _server_address << ":" << port;
      PLOG("(%d) server url: (%s)", _get_core_index(core), url.str().c_str());
      _store = fact->create(_debug_level, _owner, url.str(), _device_name);
      PMAJOR("dawn component instance: %p", static_cast<const void *>(_store));
    }
    else if ( component_is( "hstore" ) ) {
      auto device = core_to_device(core);
      std::size_t dax_base = 0x7000000000;
      /* at least the dax size, rounded for alignment */
      std::size_t dax_stride = round_up_to_pow2(dev_dax_max_size(_device_name));
      std::size_t dax_node_stride = round_up_to_pow2(dev_dax_max_count_per_node(_device_name)) * dax_stride;

      unsigned region_id = 0;
      std::ostringstream addr;
      /* stride ignores dax "major" number, so /dev/dax0.n and /dev/dax1.n map to the same memory */
      addr << std::showbase << std::hex << dax_base + dax_node_stride * device.first + dax_stride * device.second;
      std::ostringstream device_full_name;
      device_full_name << _device_name << (std::isdigit(_device_name.back()) ? "." : "") << device;
      std::ostringstream device_map;
      device_map <<
        "[ "
          " { "
            + json_map("region_id", std::to_string(region_id))
            /* actual device name is <idevice_name>.<device>, e.g. /dev/dax0.2 */
            + ", " + json_map("path", quote(device_full_name.str()))
            + ", " + json_map("addr", quote(addr.str()))
            + " }"
        " ]";
      _store = fact->create(_debug_level, "name", _owner, device_map.str());
    }
    else {
      _store = fact->create("owner", _owner);
    }

    if (_verbose)
    {
      PINF("factory: release_ref on %p", static_cast<const void *>(&fact));
    }
    fact->release_ref();
  }
  catch ( const Exception &e )
  {
    PERR("factory creation step failed: %s. Aborting experiment.", e.cause());
    throw;
  }
  catch ( const std::exception &e )
  {
    PERR("factory creation step failed: %s. Aborting experiment.", e.what());
    throw;
  }
  catch(...)
    {
      PERR("%s", "factory creation step failed");
      return 1;
    }

  return 0;
}
