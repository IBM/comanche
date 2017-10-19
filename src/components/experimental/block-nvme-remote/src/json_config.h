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

#ifndef __VOL_CONFIGURATION_H__
#define __VOL_CONFIGURATION_H__

#include <vector>
#include <string>
#include <fstream>
#include <common/exceptions.h>
#include "types.h"

class Json_configuration
{
public:

  enum {
    POLICY_NONE = 0,
    POLICY_LOCAL,
    POLICY_MIRROR,
  };

  /** 
   * Constructor using configuration file.
   * 
   * @param configuration_file 
   */
  Json_configuration(const char * configuration_file);

  /** 
   * Determine if member is in the volume configuration
   * 
   * @param member_id Member identifier
   * 
   * @return True if is a member
   */
  bool is_initial_member(const char * member_id);

  inline const std::vector<std::string>& initial_storage_members() {
    return _initial_storage_nodes;
  }

  inline const char* volume_id() {

    if(_jdoc.FindMember("volume_info") == _jdoc.MemberEnd()) return nullptr;
    
    const char *vid;
    try {
      vid = _jdoc["volume_info"]["volume_id"].GetString();
    }
    catch(...) { throw General_exception("bad volume specification"); }
    return vid;
  }

  inline const char* initial_leader() {
    const char * result;
    try {
      result = _jdoc["volume_info"]["leader_pref"][0].GetString();
    }
    catch(...) {
      throw General_exception("bad volume specification");
      return NULL;
    }
    return result;
  }

  int policy() {
    const char * result;
    try {
      if(_jdoc.FindMember("policy") == _jdoc.MemberEnd()) return POLICY_NONE;
      
      result = _jdoc["policy"].GetString();
    }
    catch(...) {
      throw General_exception("bad volume specification");
    }
    if(result==nullptr) return POLICY_NONE;
    else if(strcmp(result,"local")==0) return POLICY_LOCAL;
    else if(strcmp(result,"mirror")==0) return POLICY_MIRROR;
    else return POLICY_NONE;
  }


  inline void clear_configuration() { _initial_storage_nodes.clear(); }
  inline bool valid_configuration() { return !_initial_storage_nodes.empty(); }

  /** 
   * Get JSON configuration string
   * 
   * 
   * @return 
   */
  std::string config_string() const {
    rapidjson::StringBuffer buffer;   
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    _jdoc["volume_info"].Accept(writer);
    return std::string(buffer.GetString());
  }

  /** 
   * Get PCI addresses
   * 
   * @param hostname 
   * @param index 
   * 
   * @return 
   */
  rapidjson::Value& device_pci_addr(const char * hostname) {
    assert(!_jdoc["cluster_info"].IsNull());
    assert(!_jdoc["cluster_info"][hostname].IsNull());
    return _jdoc["cluster_info"][hostname];
  }

  rapidjson::Value& session_cores(const char * hostname) {
    assert(!_jdoc["session_cores"].IsNull());
    assert(!_jdoc["session_cores"][hostname].IsNull());
    return _jdoc["session_cores"][hostname];
  }

  int get_channel_connect_port() {
    try {
      return _jdoc["connection_info"]["port"].GetInt();
    }
    catch(...) {
      throw General_exception("bad connection_info::port specification");      
    }
  }
  
private:

  void update_from_jdoc();
    
  json_document_t          _jdoc;
  std::vector<std::string> _initial_storage_nodes;
  unsigned int             _capacity_mb;
};


#endif
