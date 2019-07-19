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
#include <assert.h>
#include <common/exceptions.h>
#include <common/utils.h>
#include <core/slab.h>
#include <fcntl.h>
#include <libgen.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <mutex>
#include <string>

#include "uipc.h"

namespace Core
{
namespace UIPC
{
struct addr_size_pair {
  addr_t addr;
  size_t size;
};

// typedef Common::Spsc_bounded_lfq_sleeping<Dataplane::Command_t, 32 /* queue
// size */> command_queue_t;

Channel::Channel(std::string name, size_t message_size, size_t queue_size)
    : _master(true) {
  size_t queue_footprint = queue_t::memory_footprint(queue_size);
  unsigned pages_per_queue = round_up(queue_footprint, PAGE_SIZE) / PAGE_SIZE;

  assert((queue_size != 0) && ((queue_size & (~queue_size + 1)) ==
                               queue_size));  // queue len is a power of 2
  assert(message_size % 8 == 0);

  PLOG("pages per FIFO queue: %u", pages_per_queue);

  size_t slab_queue_pages =
      round_up(queue_t::memory_footprint(queue_size * 2), PAGE_SIZE) /
      PAGE_SIZE;

  PLOG("slab_queue_pages: %ld", slab_queue_pages);

  size_t slab_pages =
      round_up(message_size * queue_size * 2, PAGE_SIZE) / PAGE_SIZE;
  PLOG("slab_pages: %ld", slab_pages);

  size_t total_pages = ((pages_per_queue * 2) + slab_queue_pages + slab_pages);
  PLOG("total_pages: %ld", total_pages);

  _shmem_fifo_m2s = new Shared_memory(name + "-m2s", pages_per_queue);
  _shmem_fifo_s2m = new Shared_memory(name + "-s2m", pages_per_queue);
  _shmem_slab_ring = new Shared_memory(name + "-slabring", slab_queue_pages);
  _shmem_slab = new Shared_memory(name + "-slab", slab_pages);

  _out_queue = new (_shmem_fifo_m2s->get_addr()) queue_t(
      queue_size, ((char*) _shmem_fifo_m2s->get_addr()) + sizeof(queue_t));

  _in_queue = new (_shmem_fifo_s2m->get_addr()) queue_t(
      queue_size, ((char*) _shmem_fifo_s2m->get_addr()) + sizeof(queue_t));

  unsigned slab_slots = queue_size * 2;
  _slab_ring = new (_shmem_slab_ring->get_addr()) queue_t(
      slab_slots, ((char*) _shmem_slab_ring->get_addr()) + sizeof(queue_t));
  byte* slot_addr = (byte*) _shmem_slab->get_addr();
  for (unsigned i = 0; i < slab_slots; i++) {
    _slab_ring->enqueue(slot_addr);
    slot_addr += message_size;
  }
}

Channel::Channel(std::string name) : _master(false) {
  _shmem_fifo_m2s = new Shared_memory(name + "-m2s");
  PMAJOR("got fifo (m2s) @ %p - %lu bytes", _shmem_fifo_m2s->get_addr(),
         _shmem_fifo_m2s->get_size());

  _shmem_fifo_s2m = new Shared_memory(name + "-s2m");
  PMAJOR("got fifo (s2m) @ %p - %lu bytes", _shmem_fifo_s2m->get_addr(),
         _shmem_fifo_s2m->get_size());

  _shmem_slab_ring = new Shared_memory(name + "-slabring");
  PMAJOR("got slab ring @ %p - %lu bytes", _shmem_slab_ring->get_addr(),
         _shmem_slab_ring->get_size());

  _shmem_slab = new Shared_memory(name + "-slab");
  PMAJOR("got slab @ %p - %lu bytes", _shmem_slab->get_addr(),
         _shmem_slab->get_size());

  _in_queue = reinterpret_cast<queue_t*>(_shmem_fifo_m2s->get_addr());
  _out_queue = reinterpret_cast<queue_t*>(_shmem_fifo_s2m->get_addr());
  _slab_ring = reinterpret_cast<queue_t*>(_shmem_slab_ring->get_addr());

  usleep(500000); /* hack to let master get ready - could improve with state in
                     shared memory */
}

Channel::~Channel() {
  /* don't delete queues since they were constructed on shared memory */

  delete _shmem_fifo_m2s;
  delete _shmem_fifo_s2m;
  delete _shmem_slab_ring;
  delete _shmem_slab;
}

status_t Channel::send(void* msg) {
  assert(_out_queue);
  if (_out_queue->enqueue(msg)) {
    return S_OK;
  }
  else {
    return E_FULL;
  }
}

status_t Channel::recv(void*& recvd_msg) {
  assert(_in_queue);
  if (_in_queue->dequeue(recvd_msg))
    return S_OK;
  else
    return E_EMPTY;
}

void Channel::unblock_threads() { _in_queue->exit_threads(); }

void* Channel::alloc_msg() {
  assert(_slab_ring);
  void* msg = nullptr;
  _slab_ring->dequeue(msg);
  if (!msg) throw General_exception("channel: out of message slots");
  return msg;
}

status_t Channel::free_msg(void* msg) {
  assert(_slab_ring);
  assert(msg);
  /* currently no checks for validity */
  _slab_ring->enqueue(msg);
  return S_OK;
}

/**
 * Shared_memory class
 *
 * @param name
 * @param n_pages
 */

Shared_memory::Shared_memory(std::string name, size_t n_pages)
    : _master(true), _name(name) {
  std::string fifo_name = "fifo." + name;
  if (option_DEBUG) PLOG("try to negotiate_shared_memory(%s): master", name.c_str());
  _vaddr = negotiate_addr_create(fifo_name.c_str(), n_pages * PAGE_SIZE);
  assert(_vaddr);
  _size_in_pages = n_pages;

  if (option_DEBUG) PLOG("open_shared_memory(%s): master", name.c_str());

  open_shared_memory(name.c_str(), true);
}

Shared_memory::Shared_memory(std::string name)
    : _master(false), _name(name), _size_in_pages(0) {
  std::string fifo_name = "fifo." + name;
  if (option_DEBUG) PLOG("try to negotiate_shared_memory(%s): slave", name.c_str());
  _vaddr = negotiate_addr_connect(fifo_name.c_str(), &_size_in_pages);
  assert(_vaddr);

  if (option_DEBUG) PLOG("open_shared_memory(%s): slave", name.c_str());

  open_shared_memory(name.c_str(), false);
}

Shared_memory::~Shared_memory() noexcept(false) {
  PLOG("unmapping shared memory: %p", _vaddr);

  if (munmap(_vaddr, _size_in_pages * PAGE_SIZE) != 0)
    throw General_exception("unmap failed");

  if (_master) {
    int rc = shm_unlink(_name.c_str());
    if (rc != 0)
      throw General_exception("shared memory failed to unlink (%s)",
                              _name.c_str());

    for (auto& n : _fifo_names) {
      PLOG("removing fifo (%s)", n.c_str());
      rc = unlink(n.c_str());
      if (rc != 0)
        throw General_exception("shared memory failed to remove fifo (%s)",
                                n.c_str());
    }
  }
}

void* Shared_memory::get_addr(size_t offset) {
  if (offset >> (_size_in_pages * PAGE_SIZE))
    throw API_exception("invalid offset parameter");

  return reinterpret_cast<void*>(((addr_t) _vaddr) + offset);
}

void Shared_memory::open_shared_memory(std::string name, bool master) {
  PLOG("open shared memory:%s %d", name.c_str(), master);

  umask(0);
  int fd = -1;

  if (master) {
    fd = shm_open(name.c_str(), O_CREAT | O_TRUNC | O_RDWR,
                  S_IRUSR | S_IWUSR | S_IROTH | S_IWOTH | S_IRGRP | S_IWGRP);
  }
  else {
    while (fd == -1) {
      fd = shm_open(name.c_str(), O_RDWR,
                    S_IRUSR | S_IWUSR | S_IROTH | S_IWOTH | S_IRGRP | S_IWGRP);
      usleep(100000);
    }
  }

  if (fd == -1)
    throw Constructor_exception("shm_open failed to open/create %s",
                                name.c_str());

  if (ftruncate(fd, _size_in_pages * PAGE_SIZE))
    throw General_exception("unable to allocate shared memory IPC");

  void* ptr = mmap(_vaddr, _size_in_pages * PAGE_SIZE, PROT_READ | PROT_WRITE,
                   MAP_SHARED | MAP_FIXED, fd, 0);
  if (ptr != _vaddr)
    throw Constructor_exception("mmap failed in Shared_memory");

  memset(ptr, 0, _size_in_pages * PAGE_SIZE);
  close(fd);
}

static addr_t VADDR_BASE = 0x8000000000;

static void wait_for_read(int fd, size_t s) {
  assert(s > 0);

  int count = 0;
  do {
    ioctl(fd, FIONREAD, &count);
  } while (count < s);
}

void* Shared_memory::negotiate_addr_create(std::string name,
                                           size_t size_in_bytes) {
  /* create FIFO - used to negotiate memory address */
  umask(0);

  std::string name_s2c = name + ".s2c";
  std::string name_c2s = name + ".c2s";

  if (option_DEBUG) {
    PLOG("mkfifo %s", name_s2c.c_str());
    PLOG("mkfifo %s", name_c2s.c_str());
  }

  assert(_master);

  unlink(name_c2s.c_str());
  unlink(name_s2c.c_str());

  if (mkfifo(name_c2s.c_str(), 0666) || mkfifo(name_s2c.c_str(), 0666)) {
    perror("mkfifo:");
    throw General_exception("mkfifo failed in negotiate_addr_create");
  }

  int fd_s2c = open(name_s2c.c_str(), O_WRONLY);
  int fd_c2s = open(name_c2s.c_str(), O_RDONLY);

  assert(fd_c2s >= 0 && fd_s2c >= 0);

  PLOG("saving fifo name: %s", name_c2s.c_str());
  PLOG("saving fifo name: %s", name_s2c.c_str());
  _fifo_names.push_back(name_c2s);
  _fifo_names.push_back(name_s2c);

  addr_t vaddr = VADDR_BASE;
  void* ptr = nullptr;

  do {
    ptr = mmap((void*) vaddr, size_in_bytes, PROT_NONE,
               MAP_SHARED | MAP_ANONYMOUS, 0, 0);

    if ((ptr == (void*) -1) || (ptr != (void*) vaddr)) {
      munmap(ptr, size_in_bytes);
      vaddr += size_in_bytes;
      continue; /* slide and retry */
    }

    /* send proposal */
    if (option_DEBUG)
      PLOG("sending vaddr proposal: %p - %ld bytes", ptr, size_in_bytes);

    addr_size_pair offer = {vaddr, size_in_bytes};
    if (write(fd_s2c, &offer, sizeof(addr_size_pair)) != sizeof(addr_size_pair))
      throw General_exception(
          "write failed in uipc_accept_shared_memory (offer)");

    if (option_DEBUG) PLOG("waiting for response..");

    wait_for_read(fd_c2s, 1);

    char response = 0;
    if (read(fd_c2s, &response, 1) != 1)
      throw General_exception(
          "read failed in uipc_accept_shared_memory (response)");
    assert(response);

    if (response == 'Y') { /* 'Y' signals agreement */
      break;
    }

    /* remove previous mapping and slide along */
    munmap(ptr, size_in_bytes);
    vaddr += size_in_bytes;
  } while (1);

  close(fd_c2s);
  close(fd_s2c);

  PLOG("master: negotiated %p", ptr);
  return ptr;
}

void* Shared_memory::negotiate_addr_connect(std::string name,
                                            size_t* size_in_pages_out) {
  umask(0);

  std::string name_s2c = name + ".s2c";
  std::string name_c2s = name + ".c2s";

  // Wait till master creates the fifo.
  int fd_s2c = -1, fd_c2s = -1;
  while(fd_s2c <=0){
    fd_s2c = open(name_s2c.c_str(), O_RDONLY);
    usleep(100);
  }
  while(fd_c2s <=0){
    fd_c2s = open(name_c2s.c_str(), O_WRONLY);
    usleep(100);
  }
  assert(fd_c2s != ENXIO);

  _fifo_names.push_back(name_c2s);
  _fifo_names.push_back(name_s2c);

  void* ptr = nullptr;
  addr_size_pair offer = {0};

  do {
    wait_for_read(fd_s2c,
                  sizeof(addr_size_pair));  // sizeof(void*) + sizeof(size_t));
    if (read(fd_s2c, &offer, sizeof(addr_size_pair)) != sizeof(addr_size_pair))
      throw General_exception(
          "fread failed (offer) in uipc_connect_shared_memory");

    if (option_DEBUG) PLOG("got offer %lx - %ld bytes", offer.addr, offer.size);

    assert(offer.size > 0);

    ptr = mmap((void*) offer.addr, offer.size, PROT_NONE, MAP_SHARED | MAP_ANON,
               0, 0);

    char answer;
    if (ptr != (void*) offer.addr) {
      munmap(ptr, offer.size);
      answer = 'N';
      if (write(fd_c2s, &answer, sizeof(answer)) != sizeof(answer))
        throw General_exception("write failed");
    }
    else {
      answer = 'Y';
      if (write(fd_c2s, &answer, sizeof(answer)) != sizeof(answer))
        throw General_exception("write failed");
      break;
    }
  } while (1);

  close(fd_s2c);
  close(fd_c2s);

  *size_in_pages_out = offer.size / PAGE_SIZE;

  PLOG("slave: negotiated %p", ptr);
  return ptr;
}

}  // namespace UIPC
}  // namespace Core

/**
 * 'C' exported functions
 *
 */
extern "C" channel_t uipc_create_channel(const char* path_name,
                                         size_t message_size,
                                         size_t queue_size) {
  return new Core::UIPC::Channel(path_name, message_size, queue_size);
}

extern "C" channel_t uipc_connect_channel(const char* path_name) {
  return new Core::UIPC::Channel(path_name);
}

extern "C" status_t uipc_close_channel(channel_t channel) {
  try {
    delete reinterpret_cast<Core::UIPC::Channel*>(channel);
  } catch (...) {
    return E_FAIL;
  }
  return S_OK;
}

extern "C" void* uipc_alloc_message(channel_t channel) {
  auto ch = static_cast<Core::UIPC::Channel*>(channel);
  assert(ch);
  return ch->alloc_msg();
}

extern "C" status_t uipc_free_message(channel_t channel, void* message) {
  auto ch = static_cast<Core::UIPC::Channel*>(channel);
  assert(ch);
  return ch->free_msg(message);
}

extern "C" status_t uipc_send(channel_t channel, void* data) {
  auto ch = static_cast<Core::UIPC::Channel*>(channel);
  assert(ch);
  return ch->send(data);
}

extern "C" status_t uipc_recv(channel_t channel, void*& data_out) {
  auto ch = static_cast<Core::UIPC::Channel*>(channel);
  assert(ch);
  return ch->recv(data_out);
}
