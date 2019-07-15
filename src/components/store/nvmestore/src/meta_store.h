#ifndef META_STORE_H_
#define META_STORE_H_

#include <api/components.h>
#include <api/kvstore_itf.h>
#include "nvmestore_types.h"

using namespace Component;
namespace nvmestore
{
/** Metastore to store obj meta and also block allocation info.*/
class MetaStore {
 public:
  MetaStore(const std::string&   owner,
            const std::string&   name,
            const std::string&   pm_path,
            const persist_type_t persist_type)
      : _persist_type(persist_type)
  {
    IBase* comp;
    if (_persist_type == PERSIST_FILE) {
      comp = load_component("libcomanche-storefile.so", filestore_factory);
    }
    else if (_persist_type == PERSIST_HSTORE)
      comp = load_component("libcomanche-hstore.so", hstore_factory);
    else {
      throw API_exception("Option %d not supported", _persist_type);
    }
    if (!comp)
      throw General_exception("unable to initialize Dawn backend component");

    IKVStore_factory* fact =
        (IKVStore_factory*) comp->query_interface(IKVStore_factory::iid());
    assert(fact);

    unsigned debug_level = 0;
    if (_persist_type ==
        PERSIST_FILE) { /* components that support debug level */
      std::map<std::string, std::string> params;
      params["pm_path"] = pm_path + "/meta/";
      _store            = fact->create(debug_level, params);
    }
    else if (_persist_type == PERSIST_HSTORE) {
      std::string dax_config =
          "[ { \"region_id\": 0, \"path\" : \"/dev/dax0.1\", \"addr\" : "
          "\"0x9000000000\" } ]";

      _store = fact->create(owner, name, dax_config);
      if (!_store)
        throw General_exception(
            "hstore failed in initialization. you shall set export "
            "USE_DRAM=24; export NO_CLFLUSHOPT=1; export DAX_RESET=1");
    }
    else {
      // TODO refer to dawn
      throw API_exception("hstore init not implemented");
    }
    fact->release_ref();
  }
  ~MetaStore()
  {
    PLOG("deleting meta store");
    _store->release_ref();
  }

  persist_type_t get_type() { return _persist_type; }
  IKVStore*      get_store() { return _store; }

 private:
  persist_type_t _persist_type;
  IKVStore*      _store;
};

}  // namespace nvmestore

#endif
