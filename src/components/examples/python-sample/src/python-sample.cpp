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
#include <iostream>
#include <boost/python.hpp>
#include <common/exceptions.h>
#include "python-sample.h"

#include <config_comanche.h>

__attribute__((constructor)) static void _ctor() 
{
  std::string path = CONF_COMANCHE_HOME;
  path += "/src/components/examples/python-sample/";
  
  PLOG("setting PYTHONPATH=%s", path.c_str());
  setenv("PYTHONPATH", path.c_str(), true);
  
  Py_Initialize();
}


using namespace boost;

Sample::Sample(std::string name) : _name(name)
{
  try {
    _main_module = python::import("__main__");
    _main_namespace = _main_module.attr("__dict__");

    /* note we set PYTHONPATH in global constructor */
    python::exec("import pysample", _main_namespace);
  }
  catch(...) {
    throw General_exception("boost::python::import failed");
  }

  try {
    _pyobj_sample = python::eval("pysample.Sample()", _main_namespace);
    _pyfunc_sample_say_hello = _pyobj_sample.attr("say_hello");
  }
  catch(...) {
    throw General_exception("creating python Sample object failed");
  }
}


/** 
 * Factory entry point
 * 
 */
extern "C" void * factory_createInstance(Component::uuid_t& component_id)
{
  try {
    PLOG("Python initialized OK.");
  }
  catch(...) {
    PERR("Unable to initialized Python");
    return nullptr;
  }
  
  if(component_id == Sample_factory::component_id()) {
    return static_cast<void*>(new Sample_factory());
  }
  else return NULL;
}

#undef RESET_STATE
