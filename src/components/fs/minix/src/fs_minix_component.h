#pragma once

#ifndef __FS_MINIX_COMPONENT_H__
#define __FS_MINIX_COMPONENT_H__

#include <api/block_itf.h>
#include <api/fs_itf.h>
#include "zerocopy_passthrough.h"

namespace comanche
{

class Minix_fs_component : public Zerocopy_passthrough_impl<Component::IFile_system>
{
private:
  static constexpr bool option_DEBUG = true;
public:

  /** 
   * Component/interface management
   * 
   */
  DECLARE_VERSION(0.1);
  DECLARE_COMPONENT_UUID(0xf53fd819,0xe157,0x4e69,0x9157,0xe8,0x49,0x51,0x77,0xa3,0x25);

  void * query_interface(Component::uuid_t& itf_uuid) override {
    if(itf_uuid == Component::IFile_system::iid()) {
      return (void *) static_cast<Component::IFile_system *>(this);
    }
    else 
      return NULL; // we don't support this interface
  }

  void unload() override {
    delete this;
  }
  
public:
  /** 
   * Constructor
   * 
   * 
   */
  Minix_fs_component();

  /** 
   * Destructor
   * 
   */
  virtual ~Minix_fs_component();
  
  /** 
   * IBase control methods
   * 
   */
  int bind(IBase * component) override;
  status_t start() override;
  status_t stop() override;
  virtual status_t shutdown() override { return E_NOT_IMPL; }
  virtual status_t reset() override { return E_NOT_IMPL; }

  
  Component::IFile* create_file(const std::string path, const std::string flags) override {
    return nullptr;
  }
  
  Component::IFile* open(const std::string path, const std::string flags) override {
    return nullptr;
  }
  
  void delete_file(const std::string path) override {
  }
  
  void close(Component::IFile* handle) override {
  }
              

};

} // comanche
#endif
