/*
   Copyright [2018] [IBM Corporation]

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

#ifndef __ZYRE_COMPONENT_H__
#define __ZYRE_COMPONENT_H__

#include <zyre.h>
#include <component/base.h>
#include <api/cluster_itf.h>


class Zyre_component : public Component::ICluster
{  
private:
  static constexpr bool option_DEBUG = true;
  static constexpr unsigned HEARTBEAT_INTERVAL_MS = 1000;
  
public:
  /** 
   * Constructor
   * 
   * @param block_device Block device interface
   * 
   */
  Zyre_component(const std::string& node_name,
                 const std::string& end_point);

  /** 
   * Destructor
   * 
   */
  virtual ~Zyre_component();

  /** 
   * Component/interface management
   * 
   */
  DECLARE_VERSION(0.1);
  DECLARE_COMPONENT_UUID(0x5d19463b,0xa29d,0x4bc1,0x989c,0xbe,0x74,0x0a,0xc2,0x79,0x10);
  
  void * query_interface(Component::uuid_t& itf_uuid) override {
    if(itf_uuid == Component::ICluster::iid()) {
      return (void *) static_cast<Component::ICluster*>(this);
    }
    else return NULL; // we don't support this interface
  }

  void unload() override {
    delete this;
  }

public:
  
  /* ICluster interface */
  virtual void start_node() override;
  virtual void stop_node() override;
  virtual std::string uuid() const override;
  virtual std::string node_name() const override;
  virtual void dump_info() const override;
  virtual void group_join(const std::string& group) override;
  virtual void group_leave(const std::string& group) override;
  virtual void shout(const std::string& group, const std::string& type, const std::string& message) override;
  virtual void whisper(const std::string& peer_uuid, const std::string& type, const std::string& message) override;
  virtual void poll_recv(std::function<void(const std::string& sender, const std::string& type, const std::string& message)> callback) override;


private:
  zyre_t * _node;
};


class Zyre_component_factory : public Component::ICluster_factory
{  
public:

  /** 
   * Component/interface management
   * 
   */
  DECLARE_VERSION(0.1);
  DECLARE_COMPONENT_UUID(0xfac9463b,0xa29d,0x4bc1,0x989c,0xbe,0x74,0x0a,0xc2,0x79,0x10);
  
  void * query_interface(Component::uuid_t& itf_uuid) override {
    if(itf_uuid == Component::ICluster_factory::iid()) {
      return (void *) static_cast<Component::ICluster_factory*>(this);
    }
    else return NULL; // we don't support this interface
  }

  void unload() override {
    delete this;
  }

  virtual Component::ICluster * create(const std::string& node_name,
                                       const std::string& end_point) override
  {    
    Component::ICluster * obj = static_cast<Component::ICluster*>(new Zyre_component(node_name, end_point));    
    obj->add_ref();
    return obj;
  }

  virtual Component::ICluster * create(const std::string& node_name) override
  {    
    Component::ICluster * obj = static_cast<Component::ICluster*>(new Zyre_component(node_name, ""));    
    obj->add_ref();
    return obj;
  }

};



#endif
