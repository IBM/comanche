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

#ifndef __CORE_UIPC_H__
#define __CORE_UIPC_H__

#include <sys/stat.h>
#include <string>
#include <mutex>

/* expose as C because we want other programming languages to interface
   onto this API */
extern "C"
{
  void * uipc_create_shared_memory(const char * path_name, size_t size_in_bytes, int* fd);
  void * uipc_connect_shared_memory(const char * path_name, int* fd);
  void   uipc_close_shared_memory(int fd);
}

// namespace Core {
// namespace UIPC {

// class Base
// {
// private:
//   static constexpr bool option_DEBUG = false;
  
// public:
//   Base(std::string id) : _session_id(id)
//   {
//     PLOG("UIPC session:%s", _session_id.c_str());
//   }
  
//   virtual ~Base()
//   {
//   }
  
//   virtual std::string session_id() const
//   {
//     return _session_id;
//   }

// protected:
//   std::string _session_id;
// };

// class Bootstrap
// {
// public:
//   Bootstrap(std::string path_name) {
//   }
// };

// #if 0
// /** 
//  * Fast point-to-point communications through shared memory.
//  * 
//  */
// class Session : public Base
// {
// public:

//   /** 
//    * Constructor
//    * 
//    */
//   Session(size_t size_in_bytes) : Base(create_id())
//   {
//     assert(!_session_id.empty());

//     // auto fn = std::string("/var/run/shm/") + _session_id;
//     // remove(fn.c_str());


//     _shared_vaddr = next_free_vaddr();

//     /* create or open shared memory segment */
//     {
//       /* TODO: improve security around precise permissions 
//          may have to change file in permissions for /var/run/shm/xxx file
//       */
//       umask(0);
        
//       int fd = shm_open(_session_id.c_str(),
//                         O_CREAT | O_TRUNC | O_RDWR,
//                         S_IRUSR | S_IWUSR | S_IROTH | S_IWOTH | S_IRGRP | S_IWGRP);
//       if(fd == -1)
//         throw Constructor_exception("shm_open failed to open/create %s", _session_id.c_str());

//       if(ftruncate(fd,PAGE_SIZE*2)) /* allocate 8K */
//         throw General_exception("unable to allocate 4K page for shared memory IPC");

//       _shared_segment = mmap((void*) _shared_vaddr,
//                              PAGE_SIZE*2,
//                              PROT_READ | PROT_WRITE,
//                              MAP_SHARED | MAP_FIXED, fd, 0);

//       if(_shared_segment == (void*) -1)
//         throw Constructor_exception("mmap on shared segment at fixed address (0x%lx) failed", _shared_vaddr);
//       assert(_shared_segment == (void*)_shared_vaddr);
        
//       close(fd);
//     }
      
//     assert(_shared_segment);

//     /* perform constructor on this side */
//     static_assert(sizeof(_tx_queue) <= PAGE_SIZE, "insuffient queue memory");
//     _tx_queue = (command_queue_t *) new (_shared_segment) command_queue_t();
//     _rx_queue = (command_queue_t *) new (((char*) _shared_segment) + PAGE_SIZE) command_queue_t();

//     PLOG("_tx_queue base:%p", _tx_queue->buffer_base());
//     PLOG("_rx_queue base:%p", _rx_queue->buffer_base());
    
//   }

//   /** 
//    * Destructor
//    * 
//    * 
//    */
//   virtual ~Session() {

//     /* clean up shared memory */
//     assert(_shared_segment);
//     if(munmap(_shared_segment, PAGE_SIZE*2))
//       throw General_exception("unmap of shared segment failed");

//     shm_unlink(_session_id.c_str());

//     /* release virtual address */
//     release_vaddr(_shared_vaddr);
//   }
    
//   addr_t shared_vaddr() const {
//     return (addr_t) _shared_segment;
//   }

// private:
//   /* manage virtual address space (static) */
//   static std::mutex        _vspace_lock;
//   static std::list<addr_t> _vspace_list;
//   static addr_t            _vspace_base;

//   static addr_t initialize_vspace() {
//     {
//       std::lock_guard<std::mutex> g(_vspace_lock);
//       addr_t vaddr = 0x800000000UL;
//       for(addr_t i=0;i<64;i++) {
//         auto addr = vaddr + (i * (PAGE_SIZE*2));
//         _vspace_list.push_back(addr);
//       }
//     }
//     return _vspace_base;
//   }
    
//   static addr_t next_free_vaddr() {
//     std::lock_guard<std::mutex> g(_vspace_lock);
//     addr_t result = _vspace_list.front();
//     assert(_vspace_list.empty()==false);
//     _vspace_list.pop_front();
//     return result;
//   }

//   static void release_vaddr(addr_t addr) {
//     std::lock_guard<std::mutex> g(_vspace_lock);
//     _vspace_list.push_front(addr);
//   }

// private:

//   static addr_t SEGMENT_START_ADDR;
  
//   static std::string create_id() {
//     std::string r = "Dataplane_";
//     r += Common::random_string(8);
//     return r;
//   }
  
//   void *             _shared_segment;
//   command_queue_t *  _rx_queue;
//   command_queue_t *  _tx_queue;  
//   addr_t             _shared_vaddr;

// private:

//   /** 
//    * Worker thread entry point
//    * 
//    */
//   int thread_entry();
    
// };

// #endif

// }} // Core::UIPC

#endif
