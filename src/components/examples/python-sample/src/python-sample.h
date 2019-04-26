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


/* 
 * Authors: 
 * 
 * Daniel G. Waddington (daniel.waddington@ibm.com)
 *
 */

#ifndef __SAMPLE_COMPONENT_H__
#define __SAMPLE_COMPONENT_H__

#include <api/sample_itf.h>


class Sample : public Component::ISample
{  
private:
  static constexpr bool option_DEBUG = true;

public:
  /** 
   * Constructor
   * 
   * @param block_device Block device interface
   * 
   */
  Sample(std::string name);


  /** 
   * Component/interface management
   * 
   */
  DECLARE_VERSION(0.1);
  DECLARE_COMPONENT_UUID(0x59564581,0x9e1b,0x4811,0xbdb2,0x19,0x57,0xa0,0xa6,0x84,0x57);
  
  void * query_interface(Component::uuid_t& itf_uuid) override {
    if(itf_uuid == Component::ISample::iid()) {
      return (void *) static_cast<Component::ISample*>(this);
    }
    else return NULL; // we don't support this interface
  }

  void unload() override {
    delete this;
  }

public:

  /* ISample */
  void say_hello() {
    _pyfunc_sample_say_hello(); /* Python function pointer */
  }
  
private:
  std::string           _name;
  boost::python::object _main_module;
  boost::python::object _main_namespace;
  boost::python::object _pyobj_sample;
  boost::python::object _pyfunc_sample_say_hello;
};


class Sample_factory : public Component::ISample_factory
{  
public:

  /** 
   * Component/interface management
   * 
   */
  DECLARE_VERSION(0.1);
  DECLARE_COMPONENT_UUID(0xfac64581,0x9e1b,0x4811,0xbdb2,0x19,0x57,0xa0,0xa6,0x84,0x57);
  
  void * query_interface(Component::uuid_t& itf_uuid) override {
    if(itf_uuid == Component::ISample_factory::iid()) {
      return (void *) static_cast<Component::ISample_factory*>(this);
    }
    else return NULL; // we don't support this interface
  }

  void unload() override {
    delete this;
  }

  virtual Component::ISample * open(std::string name) override
  {    
    Component::ISample * obj = static_cast<Component::ISample*>(new Sample(name));    
    obj->add_ref();
    return obj;
  }

};



#endif
