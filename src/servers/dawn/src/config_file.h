#ifndef __DAWN_CONFIG_FILE_H__
#define __DAWN_CONFIG_FILE_H__


#include <string>
#include <assert.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <common/exceptions.h>
#include <rapidjson/document.h>
#include <rapidjson/filereadstream.h>

static const char* k_typenames[] = { "Null",
                                     "False",
                                     "True",
                                     "Object",
                                     "Array",
                                     "String",
                                     "Number" };

namespace Dawn
{

class Config_file
{
 private:
  static constexpr bool option_DEBUG = true;

 public:
  Config_file(const std::string& filename) {

    if(option_DEBUG)
      PLOG("config_file: (%s)", filename.c_str());
    
    using namespace rapidjson;

    /* use file stream instead of istreamwrapper because of older Ubuntu 16.04 */
    FILE * fp = fopen(filename.c_str(),"rb");
    if(fp == nullptr)
      throw General_exception("configuration file open/parse failed");

    struct stat st;
    stat(filename.c_str(), &st);
    char * buffer = (char *) malloc(st.st_size);

    try {
      FileReadStream is(fp, buffer, st.st_size);
      _doc.ParseStream(is);
    }
    catch(...) {
      throw General_exception("configuration file open/parse failed");
    }
    free(buffer);
    fclose(fp);

    _shards = _doc["shards"];
    
    PLOG("shards type:%s", k_typenames[_shards.GetType()]);
    if(!_shards.IsArray()) throw General_exception("bad JSON: shard should be array");

    for (auto& m : _shards.GetArray()) {
      if(!m["core"].IsInt()) throw General_exception("bad JSON: shards::core member not integer");
      if(!m["port"].IsInt()) throw General_exception("bad JSON: shards::port member not integer");
      if(!m["device"].IsString()) throw General_exception("bad JSON: shards::device member not string");

      if(!m["net"].IsNull())
        if(!m["net"].IsString()) throw General_exception("bad JSON: optional shards::net member not string");     
    }
    if(option_DEBUG) {
      for(unsigned i=0;i<shard_count();i++) {
        PLOG("shard: core(%d) port(%d) device(%s) net(%s)",
             get_shard_core(i),
             get_shard_port(i),
             get_shard_device(i),
             get_shard_net(i)
             );
      }
    }
    
  }

  rapidjson::SizeType shard_count() const {
    return _shards.Size();
  }

  auto get_shard(rapidjson::SizeType i) const {
    if(i > shard_count()) throw General_exception("get_shard out of bounds");
    assert(_shards[i].IsObject());    
    return _shards[i].GetObject();
  }

  unsigned int get_shard_core(rapidjson::SizeType i) const {
    if(i > shard_count()) throw General_exception("get_shard out of bounds");
    assert(_shards[i].IsObject());    
    auto shard = _shards[i].GetObject();
    return shard["core"].GetUint();
  }

  unsigned int get_shard_port(rapidjson::SizeType i) const {
    if(i > shard_count()) throw General_exception("get_shard out of bounds");
    assert(_shards[i].IsObject());    
    auto shard = _shards[i].GetObject();
    return shard["port"].GetUint();
  }

  const char * get_shard_device(rapidjson::SizeType i) const {
    if(i > shard_count()) throw General_exception("get_shard out of bounds");
    assert(_shards[i].IsObject());    
    auto shard = _shards[i].GetObject();
    return shard["device"].GetString();
  }

  const char * get_shard_net(rapidjson::SizeType i) const {
    if(i > shard_count()) throw General_exception("get_shard out of bounds");
    assert(_shards[i].IsObject());    
    auto shard = _shards[i].GetObject();
    if(shard["net"].IsNull()) return nullptr;
    return shard["net"].GetString();
  }

  unsigned int debug_level() const {
    if(_doc["debug_level"].IsNull()) return 0;
    return _doc["debug_level"].GetUint();
  }

 private:
  rapidjson::Document _doc;
  rapidjson::Value _shards;
};

}
#endif // __DAWN_CONFIG_FILE_H__
