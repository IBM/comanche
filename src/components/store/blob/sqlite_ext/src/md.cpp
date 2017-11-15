/*
   Copyright [2017] [IBM Corporation]

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

#include <string>
#include <string.h>
#include <common/exceptions.h>
#include <common/logging.h>
#include <sqlite3.h>
#include <api/components.h>
#include <api/block_itf.h>

#include "md.h"

/** 
 * Metadata residing on block device using SQLite3 as the front end.  We use the
 * SQLite3 virtual table mechanism.
 * 
 */


class Metadata
{
private:
  static constexpr bool option_DEBUG = true;
public:
  Metadata(const char * block_device_pci, int core);
  ~Metadata();
  void foo(void);
  char * get_schema();

private:
  
  Component::IBlock_device * _base_block;
  std::string                _schema;
};

using namespace Component;

    

/** 
 * Metadata class
 * 
 */

Metadata::Metadata(const char * block_device_pci, int core) : _schema("")
{
  if(option_DEBUG)
    PLOG("::Metadata(%s)", block_device_pci);

  /* create block device */
  {
    cpu_mask_t cpus;
    cpus.add_core(core);

    IBase * comp = load_component("libcomanche-blknvme.so",
                                  block_nvme_factory);

    IBlock_device_factory * fact = (IBlock_device_factory *)
      comp->query_interface(IBlock_device_factory::iid());

    _base_block = fact->create(block_device_pci, &cpus);
    assert(_base_block);
    
    fact->release_ref();
  }

  _schema = "CREATE TABLE x(TEXT foo)";
}

Metadata::~Metadata()
{
  _base_block->release_ref();
  TRACE();
}

char * Metadata::get_schema()
{
  char * schema = (char *) sqlite3_malloc(_schema.size() + 1);
  strcpy(schema, _schema.c_str());
  return schema;
}

void Metadata::foo()
{
  PLOG("foo invoked: this=%p", this);
}



/** 
 * C wrappers for Metadata class
 * 
 */
extern "C" {
  
  void * mddb_create_instance(const char * block_device_pci, int core) {
    return new Metadata(block_device_pci, core);
  }
  
  void mddb_free_instance(void * _this) {
    delete static_cast<Metadata *>(_this);
  }

  void mddb_foo(void * _this) {
    static_cast<Metadata *>(_this)->foo();
  }

  char * mddb_get_schema(void * _this) {
    return static_cast<Metadata *>(_this)->get_schema();
  }
}
