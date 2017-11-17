#include <iostream>
#include <api/metadata_itf.h>
#include "metadata.h"

using namespace Component;

Metadata::Metadata(std::string owner,
                   std::string name,
                   Component::IBlock_device * block_device,
                   int flags)
{
}

Metadata::~Metadata()
{
}


size_t Metadata::get_record_count()
{
  return 0;
}

IMetadata::iterator_t Metadata::open_iterator(std::string filter)
{
  return nullptr;
}

status_t Metadata::iterator_get(IMetadata::iterator_t iter,
                                std::string& out_metadata,
                                void *& allocator_handle,
                                uint64_t* lba,
                                uint64_t* lba_count)
{
  return E_FAIL;
}

size_t Metadata::iterator_record_count(iterator_t iter)
{
  return 0;
}

void Metadata::close_iterator(iterator_t iterator)
{
}


/** 
 * Factory entry point
 * 
 */
extern "C" void * factory_createInstance(Component::uuid_t& component_id)
{
  if(component_id == Metadata_factory::component_id()) {
    return static_cast<void*>(new Metadata_factory());
  }
  else return NULL;
}

