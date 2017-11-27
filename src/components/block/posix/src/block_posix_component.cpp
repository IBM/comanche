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


/* 
 * Authors: 
 * 
 * Daniel G. Waddington (daniel.waddington@ibm.com)
 *
 */

#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/ioctl.h>
#include <linux/fs.h>
#include <common/exceptions.h>
#include <common/utils.h>
#include <core/dpdk.h>
#include <api/block_itf.h>
#include <rapidjson/document.h>

#include "block_posix_component.h"

using namespace Component;

enum {
  IOCTL_CMD_GETBITMAP = 9,
  IOCTL_CMD_GETPHYS = 10,
};

typedef struct
{
  addr_t vaddr;
  addr_t out_paddr;
} __attribute__((packed)) IOCTL_GETPHYS_param;


Block_posix::Block_posix(std::string config) : _size_in_blocks(0), _work_id(1)
{
  using namespace rapidjson;
  Document document;
  document.Parse(config.c_str());

  if(!document.HasMember("path") ||
     !document["path"].IsString())
    throw Constructor_exception("Block_posix: bad JSON config string (%s)",config.c_str());
  
  _file_path = document["path"].GetString();

  /* size_in_blocks is not specified for raw block devices */
  if(document.HasMember("size_in_blocks") &&
     document["size_in_blocks"].IsInt64()) {
    auto val = document["size_in_blocks"].GetInt64();
    assert(val > 0);
    _size_in_blocks = val;
  }  

  if(_file_path.substr(0,4) == "/dev") {
    /* raw block device based */
    _fd = ::open(_file_path.c_str(), O_RDWR, O_DIRECT);

    if(_fd == -1) {
      perror("Block_posix::");
      throw Constructor_exception("Block_posix:: unable to open raw block device");
    }

    /* get size of device */
    uint64_t device_size;
    int rc = ioctl(_fd, BLKGETSIZE64, &device_size);
    if(rc)
      throw General_exception("ioctl failed");

    {
      /* verify sector size if 4K */
      uint64_t sector_size;
      ioctl(_fd, BLKSSZGET, &sector_size);
      if(sector_size != 4096)
        throw Constructor_exception("unsupported sector size (%ld)", sector_size);
    }
    
    assert(device_size > 0);
    _size_in_blocks = device_size / 4096;
  }
  else {
    /* file based */
    _fd = ::open(_file_path.c_str(),
                 O_CREAT | O_RDWR | O_DIRECT,
                 S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
    
    if(_fd == -1) {
      perror("Block_posix::");
      throw Constructor_exception("Block_posix:: unable to open file");
    }

    int rc = ::ftruncate(_fd, _size_in_blocks * KB(4));
    if(rc)
      throw Constructor_exception("Block_posix:: truncate failed");
  
  }

  PLOG("Block_posix: path=(%s) size=(%ld)", _file_path.c_str(), _size_in_blocks);
  
  assert(_fd);
  assert(_size_in_blocks);

  /* initialize AIO */
  struct aioinit init = {0};
  init.aio_threads = AIO_THREADS;
  init.aio_num = AIO_SIMUL;

  aio_init(&init);

  /* allocate AIO descriptors */
  for(unsigned i=0;i<AIO_DESCRIPTOR_POOL_SIZE;i++)
    _aiob_vector.push_back(new struct aiocb);

  /* open XMS if possible */
  _fd_xms = open("/dev/xms", O_RDWR);
  if(_fd_xms == -1) {
    PLOG("Block_posix: XMS not available");
  }
  else {
    PLOG("Block_posix: XMS available!");
  }

  /* initialize DPDK */
  DPDK::eal_init(256);
}
 
Block_posix::~Block_posix()
{
  std::lock_guard<std::mutex> g(_aiob_vector_lock);
  for(auto& d: _aiob_vector)
    delete d;

  fsync(_fd);
  close(_fd);

  if(_fd_xms != -1) close(_fd_xms);
}

struct aiocb *
Block_posix::allocate_descriptor()
{
  std::lock_guard<std::mutex> g(_aiob_vector_lock);
  assert(!_aiob_vector.empty());
  struct aiocb * aiocb = _aiob_vector.back();
  _aiob_vector.pop_back();
  memset(aiocb,0,sizeof(struct aiocb)); // not needed?
  return aiocb;
}

void
Block_posix::free_descriptor(struct aiocb * desc)
{
  std::lock_guard<std::mutex> g(_aiob_vector_lock);
  _aiob_vector.push_back(desc);
}


uint64_t
Block_posix::add_outstanding(struct aiocb * desc)
{
  std::lock_guard<std::mutex> g(_work_lock);
  _work_id++;
  _outstanding.push_front({desc,_work_id,0xF00D});
  return _work_id;
}

bool
Block_posix::check_complete(uint64_t workid)
{
  if(workid > _work_id)
    throw API_exception("invalid workid parameter");

  std::lock_guard<std::mutex> g(_work_lock);

  while(!_outstanding.empty()) {

    int rc;
    work_desc_t desc = _outstanding.back();
    assert(desc.magic == 0xF00D);

    // rc = aio_fsync(O_DSYNC, desc.aiocb);
    // assert(rc==0);

    
    rc = aio_error(desc.aiocb);
    if(rc == EINPROGRESS) {
      return false;
    }

    if(rc==ECANCELED)
      throw General_exception("AIO cancelled");

    if(rc) {
      throw General_exception("aio_error unexpected status %d desc.aiocb=%p offset=%ld tag=%lu aio_buf=%p",
                              rc, desc.aiocb, desc.aiocb->aio_offset, desc.tag, desc.aiocb->aio_buf);
    }
    
    //    bool status = (aio_return(desc.aiocb) == desc.aiocb->aio_nbytes);
    // if(!status)
    //   throw Logic_exception("aio returned unexpected nbytes (%ld)",
    //                         aio_return(desc.aiocb));
    
    free_descriptor(desc.aiocb); /* free descriptor */
    
    _outstanding.pop_back();

    if(option_DEBUG) {
      PINF("[block-posix]: processed completion %ld", desc.tag);
    }
    if(desc.tag >= workid) return true;
  }
  fsync(_fd);
  
  return true;
  //  throw Logic_exception("outstanding list is empty");
}


/** 
 * Factory 
 * 
 */
Component::IBlock_device *
Block_posix_factory::
create(std::string config_string, cpu_mask_t * cpuset, Core::Poller * poller)
{
  if(cpuset || poller)
    throw API_exception("unsupported");
  
  IBlock_device *blk = static_cast<IBlock_device*>(new Block_posix(config_string));
  blk->add_ref();
  return blk;
}


/** 
 * IBlock_device
 * 
 */

workid_t
Block_posix::
async_read(io_buffer_t buffer,
           uint64_t buffer_offset,
           uint64_t lba,
           uint64_t lba_count,
           int queue_id,
           io_callback_t cb,
           void * cb_arg0,
           void * cb_arg1)
{
  if(option_DEBUG)
    PINF("[+] block-posix: async_read(buffer=%p, offset=%lu, lba=%lu, lba_count=%lu",
         (void*) buffer, buffer_offset, lba, lba_count);

  if(cb || cb_arg0 || cb_arg1)
    throw API_exception("posix-block device does not yet support callbacks");
  
  if(queue_id > 0)
    throw API_exception("queue parameter not supported");

  struct aiocb * desc = allocate_descriptor();
  memset(desc, 0, sizeof(struct aiocb));
  desc->aio_buf = reinterpret_cast<char*>(buffer) + buffer_offset;
  desc->aio_fildes = _fd;   
  desc->aio_offset = lba * IO_BLOCK_SIZE;
  desc->aio_lio_opcode = LIO_READ; // ignored according to man page
  desc->aio_nbytes = lba_count * IO_BLOCK_SIZE;
  desc->aio_sigevent.sigev_notify = SIGEV_NONE;
  
  if(aio_read(desc)!=0)
    throw General_exception("aio_read failed");

  uint64_t gwid = add_outstanding(desc);
  if(option_DEBUG)
    PLOG("block-posix: async-read submitted %ld", gwid);
  
  return gwid;
}
 
workid_t
Block_posix::
async_write(io_buffer_t buffer,
            uint64_t buffer_offset,
            uint64_t lba,
            uint64_t lba_count,
            int queue_id,
            io_callback_t cb,
            void * cb_arg0,
            void * cb_arg1)
{
  if(option_DEBUG)
    PINF("[+] block-posix: async_write(buffer=%p, offset=%lu, lba=%lu, lba_count=%lu",
         (void*)buffer, buffer_offset, lba, lba_count);

  if(cb || cb_arg0 || cb_arg1)
    throw API_exception("posix-block device does not yet support callbacks");
  
  if(queue_id > 0)
    throw API_exception("queue parameter not supported");

  struct aiocb * desc = allocate_descriptor();
  memset(desc, 0, sizeof(struct aiocb));
  desc->aio_buf = reinterpret_cast<char*>(buffer) + buffer_offset;
  desc->aio_fildes = _fd;   
  desc->aio_offset = lba * IO_BLOCK_SIZE;
  desc->aio_lio_opcode = LIO_WRITE; // ignored according to man page
  desc->aio_nbytes = lba_count * IO_BLOCK_SIZE;
  desc->aio_sigevent.sigev_notify = SIGEV_NONE;
  
  if(aio_write(desc)!=0)
    throw General_exception("aio_write failed");

  uint64_t gwid = add_outstanding(desc);
  if(option_DEBUG)
    PLOG("block-posix: async-write submitted %ld", gwid);
  
  return gwid;
}

void
Block_posix::
write(Component::io_buffer_t buffer,
      uint64_t buffer_offset,
      uint64_t lba,
      uint64_t lba_count,
      int queue_id)
{
  if(option_DEBUG)
    PINF("[+] block-posix: write(buffer=%p, offset=%lu, lba=%lu, lba_count=%lu",
         (void*)buffer, buffer_offset, lba, lba_count);

  if(queue_id > 0)
    throw API_exception("queue parameter not supported");

  void * ptr = (void*) (reinterpret_cast<char*>(buffer) + buffer_offset);
  size_t nbytes = lba_count * IO_BLOCK_SIZE;
  ssize_t rc = pwrite(_fd, ptr, nbytes, lba * IO_BLOCK_SIZE);
  if(rc != nbytes)
    throw General_exception("%s: pwrite failed", __PRETTY_FUNCTION__);

  if(fsync(_fd))
    throw General_exception("%s: fsync failed", __PRETTY_FUNCTION__);

  if(option_DEBUG)
    PINF("[block-posix] write: %p lba=%lu lba_count=%lu", ptr, lba, lba_count);
}

void
Block_posix::
read(Component::io_buffer_t buffer,
     uint64_t buffer_offset,
     uint64_t lba,
     uint64_t lba_count,
     int queue_id)
{
  if(option_DEBUG)
    PINF("[+] block-posix: read(buffer=%p, offset=%lu, lba=%lu, lba_count=%lu",
         (void*)buffer, buffer_offset, lba, lba_count);

  if(queue_id > 0)
    throw API_exception("queue parameter not supported");

  void * ptr = (void*) (reinterpret_cast<char*>(buffer) + buffer_offset);
  size_t nbytes = lba_count * IO_BLOCK_SIZE;
  ssize_t rc = pread(_fd, ptr, nbytes, lba * IO_BLOCK_SIZE);
  if(rc != nbytes)
    throw General_exception("%s: pread failed", __PRETTY_FUNCTION__);

  if(option_DEBUG)
    PINF("[block-posix] read: %p lba=%lu lba_count=%lu", ptr, lba, lba_count);
}


/** 
 * Check for completion of a work request. This API is thread-safe.
 * 
 * @param gwid Work request identifier
 * 
 * @return True if completed.
 */
bool
Block_posix::
check_completion(workid_t gwid, int queue_id)
{
  if(queue_id > 0)
    throw API_exception("queue parameter not supported");

  return check_complete(gwid);
}

/** 
 * Get device information
 * 
 * @param devinfo pointer to VOLUME_INFO struct
 * 
 * @return S_OK on success
 */
void
Block_posix::
get_volume_info(VOLUME_INFO& devinfo)
{
  devinfo.block_size = IO_BLOCK_SIZE;
  devinfo.distributed = false;
  devinfo.hash_id = 0;
  devinfo.max_lba = _size_in_blocks - 1;
  strncpy(devinfo.volume_name,_file_path.c_str(),VOLUME_INFO_MAX_NAME);
}



/** 
 * Factory entry point
 * 
 */
extern "C" void * factory_createInstance(Component::uuid_t& component_id)
{
  if(component_id == Block_posix_factory::component_id()) {
    PLOG("Creating 'Block_posix' factory.");
    return static_cast<void*>(new Block_posix_factory());
  }
  else
    return NULL;
}

