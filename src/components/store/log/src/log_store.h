#ifndef __LOG_STORE_H__
#define __LOG_STORE_H__

#include <core/zerocopy_passthrough.h>
#include <core/ring_buffer.h>
#include <api/log_itf.h>
#include <api/block_allocator_itf.h>

#include "header.h"
#include "buffer_manager.h"

/** 
 * Log store is single threaded so that ordering can be maintained.  Currently, it is memcpy
 * based, but this could be improved to use zero-copy IO buffers.
 * 
 */
class Log_store : public Core::Zerocopy_passthrough_impl<Component::ILog>
{  
private:
  static constexpr bool option_DEBUG = false;
  
public:
  /** 
   * Constructor
   * 
   * @param owner Owner
   * @param name Name identifier
   * @param block Block device
   * @param flags Flags
   * @param fixed_size Specify size if fixed
   * @param use_src True if a crc32 should be included
   * 
   */
  Log_store(std::string owner,
            std::string name,
            Component::IBlock_device* block,
            int flags,
            size_t fixed_size,
            bool use_crc);

  /** 
   * Destructor
   * 
   */
  virtual ~Log_store();
 

  /** 
   * Component/interface management
   * 
   */
  DECLARE_VERSION(0.1);
  DECLARE_COMPONENT_UUID(0xef92ad2a,0xaf0d,0x4834,0xbfc0,0x93,0x13,0x1a,0x3e,0xd7,0xb0);
  
  void * query_interface(Component::uuid_t& itf_uuid) override {
    if(itf_uuid == Component::ILog::iid()) {
      return (void *) static_cast<Component::ILog*>(this);
    }
    else return nullptr; // we don't support this interface
  }

  void unload() override {
    delete this;
  }

  /* ILog */
  
  /** 
   * Synchronously write data (copy-based)
   * 
   * @param data Pointer to data
   * @param data_len Length of data in bytes
   * @param queue_id Queue identifier
   * 
   * @return Byte offset of item (actually of crc32,len header)
   */
  virtual index_t write(const void * data, const size_t data_len, unsigned queued_id) override;

  /** 
   * Read data from a given offset (copy-based)
   * 
   * @param index Index of item
   * @param iob Target IO buffer
   * @param queue_id Queue identifier
   * 
   * @return Pointer to record in iob
   */
  virtual byte * read(const index_t index, Component::io_buffer_t iob, unsigned queue_id) override;
  
  /** 
   * Flush queued IO and wait for completion
   * 
   * 
   * @return S_OK on success
   */
  virtual status_t flush(unsigned queue_id) override;

  /** 
   * Get last point of used storage
   * 
   * 
   * @return Index (byte offset)
   */
  virtual index_t get_tail() override {
    std::lock_guard<std::mutex> g(_lock);
    return _hdr.get_tail();
  }

private:

  inline size_t header_size() {
    if(_use_crc) return 8;
    else return 4;
  }
  
private:

  size_t _max_io_blocks;
  size_t _max_io_bytes;
  size_t _num_io_queues;
  size_t _fixed_size;
  bool   _use_crc;
  Header _hdr;

  std::mutex             _lock;
  Component::VOLUME_INFO _vi;
  Buffer_manager         _bm;
};


class Log_store_factory : public Component::ILog_factory
{  
public:

  /** 
   * Component/interface management
   * 
   */
  DECLARE_VERSION(0.1);
  DECLARE_COMPONENT_UUID(0xfac2ad2a,0xaf0d,0x4834,0xbfc0,0x93,0x13,0x1a,0x3e,0xd7,0xb0);
  
  void * query_interface(Component::uuid_t& itf_uuid) override {
    if(itf_uuid == Component::ILog_factory::iid()) {
      return (void *) static_cast<Component::ILog_factory*>(this);
    }
    else return NULL; // we don't support this interface
  }

  void unload() override {
    delete this;
  }
  
  virtual Component::ILog * create(std::string owner,
                                   std::string name,
                                   Component::IBlock_device * block,
                                   int flags,
                                   size_t fixed_size,
                                   bool use_crc32) override
  {
    using namespace Component;
    
    if(block == nullptr)
      throw Constructor_exception("%s: bad block interface param", __PRETTY_FUNCTION__);
        
    ILog * obj = static_cast<ILog *> (new Log_store(owner,
                                                    name,
                                                    block,
                                                    flags,
                                                    fixed_size,
                                                    use_crc32));    
    obj->add_ref();
    return obj;
  }
};



#endif // __LOG_STORE_H__
