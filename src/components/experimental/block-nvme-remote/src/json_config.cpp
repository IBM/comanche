/*
 * (C) Copyright IBM Corporation 2017. All rights reserved.
 *
 */

/* 
 * Authors: 
 * 
 * Daniel G. Waddington (daniel.waddington@ibm.com)
 *
 */

#include "json_config.h"

#include <sstream>
#include <common/exceptions.h>
#include <common/utils.h>
#include <common/logging.h>

#define SYM_VOLUME_CAPACITY "volume_capacity_mb"
#define SYM_NVME_DEVICE_IDX "nvme_device_index"


/** 
 * Constructor based on static configuration file
 * 
 * @param configuration_file 
 * 
 */
Json_configuration::
Json_configuration(const char * configuration_file)
{
  using namespace rapidjson;

  if(!configuration_file)
    throw API_exception("volume configuration file null");
  
  std::ifstream ifs(configuration_file);
  if(!ifs.is_open())
    throw Constructor_exception("unable to open volume configuration file");
  
  rapidjson::IStreamWrapper isw(ifs);
  _jdoc.ParseStream(isw);
  
  update_from_jdoc();
}

/** 
 * Update configuration state from JSON document
 * 
 */
void Json_configuration::update_from_jdoc()
{
  using namespace rapidjson;


  if(_jdoc.FindMember("volume_info") == _jdoc.MemberEnd()) return;
  
  if(_jdoc["volume_info"].IsNull()) {
    PLOG("no volume_info");
    return;
  }
  
  try {
    assert(!_jdoc["volume_info"].IsNull());
   
    const Value& members = _jdoc["volume_info"]["storage_nodes"];
    for (SizeType i = 0; i < members.Size(); i++) {
        if(!members[i].IsString())
          throw Constructor_exception("invalid volume configuration file (storage_nodes)");
        
        _initial_storage_nodes.push_back(std::string(members[i].GetString()));
    }

    _capacity_mb = _jdoc["volume_info"][SYM_VOLUME_CAPACITY].GetInt();
      
  }
  catch(...) {
    throw General_exception("bad configuration file");
  }
  
}


bool Json_configuration::is_initial_member(const char * member_id) {
  for(auto& m: _initial_storage_nodes) {
    if(m.compare(member_id)==0) return true;
  }
  return false;
}


