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

#ifndef __CORE_IPC__
#define __CORE_IPC__

#if defined(__cplusplus)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>

#include <common/chksum.h>
#include <common/exceptions.h>
#include <common/utils.h>
#include <common/dump_utils.h>
#include <common/str_utils.h>

#include <nanomsg/nn.h>
#include <nanomsg/reqrep.h>

namespace Core
{
/** 
   * IPC server helper class
   * 
   * @param url URL of nanomsg endpoint (e.g. ipc:///tmp/foobar)
   */
class IPC_server
{
 private:
  static const auto     MAX_MSG_SIZE = 1024;
  static constexpr bool option_DEBUG = true;
  std::string           _url;

 public:
  /** 
     * Constructor
     * 
     * @param url Nanomsg URL to listen on
     */
  IPC_server(const std::string url) : _url(url)
  {
    if (_fd < 0) throw new Constructor_exception("IPC_server constructor nn_socket: %s\n", nn_strerror(nn_errno()));

    if (nn_bind(_fd, _url.c_str()) < 0)
      throw new Constructor_exception("IPC_server constructor nn_bind: %s\n", nn_strerror(nn_errno()));

    if (option_DEBUG) PLOG("IPC server endpoint (url=%s)", _url.c_str());

    std::string unix_file = _url.substr(5);
    Common::string_replace(unix_file, "//", "/");

    if (option_DEBUG) PLOG("UNIX domain socket = (%s)", unix_file.c_str());

    /* modify permission so that server can run as root if needed and 
         clients need not. Sticky bit should be set.
      */
    chmod(unix_file.c_str(), 0777);
  }

  /** 
     * Destructor
     * 
     */
  virtual ~IPC_server()
  {
    assert(_fd >= 0);

    nn_close(_fd);
  }


  /** 
     * Called to start the message loop
     * 
     */
  void ipc_start()
  {
    if (option_DEBUG) PLOG("IPC server starting on ... (url=%s)", _url.c_str());
    message_loop();
  }


  /** 
     * Set access control 
     * 
     */
  void set_acl()
  {
    // TODO
  }

  /** 
     * Helper used in process_message implementation to allocate new message
     * 
     * @param size Size of message to allocate in bytes
     * 
     * @return Pointer to new message
     */
  void* alloc_reply(size_t size)
  {
    return nn_allocmsg(size, 0);
  }

 private:
  /** 
     * Callback to handle each message (must be implemented)
     * 
     * @param msg Pointer to incoming message
     * @param msg_len Length of incoming message data
     * @param reply [out] Reply message (allocated in this function)
     * @param reply_len [out] Length of reply message
     * 
     * @return -1 to exit message loop
     */
  virtual int process_message(void* msg, size_t msg_len, void* reply, size_t reply_len) = 0;


 private:
  void message_loop()
  {
    void* msg = nn_allocmsg(MAX_MSG_SIZE, 0);
    assert(msg);

    while (!_exit) {
      assert(_fd >= 0);

      // wait for request message
      int rc = nn_recv(_fd, msg, MAX_MSG_SIZE, 0);
      assert(rc > 0);
      if (option_DEBUG) {
        PNOTICE("nn_recv(request): chksum=%x", Common::chksum32(msg, rc));
        hexdump(msg, 128);
      }

      // allocate memory for reply; this will be freed after nn_send
      void*  reply_msg     = nn_allocmsg(MAX_MSG_SIZE, 0);
      size_t reply_msg_len = MAX_MSG_SIZE;
      if (process_message(msg, rc, reply_msg, reply_msg_len) == -1) {
        PWRN("process_message fault:");
        _exit = true;
      }

      // send reply
      if (option_DEBUG) {
        PNOTICE("nn_send (reply): chksum=%x", Common::chksum32(reply_msg, reply_msg_len));
        hexdump(msg, 128);
      }

      rc = nn_send(_fd, reply_msg, reply_msg_len, 0);
      assert(rc == (int) reply_msg_len);
      if (rc == 0) throw General_exception("nn_send failed.");
      assert(rc > 0);
    }

    nn_freemsg(msg);

    if (option_DEBUG) PDBG("message loop exited ok.");
  }

 private:
  const int _fd{nn_socket(AF_SP, NN_REP)};
  bool      _exit = false;
};



/** 
   * IPC client helper class
   * 
   * @param url URL of nanomsg endpoint (e.g. ipc:///tmp/foobar)
   */
class IPC_client
{
 private:
  static constexpr auto MAX_REPLY_SIZE = 1024;
  static constexpr bool option_DEBUG   = false;

 public:
  /** 
     * Constructor
     * 
     * @param url Nanomsg URL to connect to
     */
  IPC_client(const std::string url)
  {
    if (_fd < 0) throw new Constructor_exception("IPC_client constructor nn_socket: %s\n", nn_strerror(nn_errno()));

    if (nn_connect(_fd, url.c_str()) < 0)
      throw new Constructor_exception("IPC_client constructor nn_connect: %s\n", nn_strerror(nn_errno()));

    PDBG("IPC (nanogmsg) ctor (url=%s)", url.c_str());
  }

  /** 
     * Destructor
     * 
     */
  ~IPC_client()
  {
    assert(_fd >= 0);
    nn_close(_fd);
  }

  /** 
     * Free message from send_and_wait
     * 
     * @param msg Pointer to message to free.
     * 
     * @return EFAULT if pointer is invalid
     */
  static int free_msg(void* msg)
  {
    return nn_freemsg(msg);
  }

  /** 
     * Send string data
     * 
     * @param msg Pointer to string to send
     * 
     * @return Pointer to reply message. Must be freed by client through
     */
  void* send_and_wait(const char* msg, size_t msg_len, size_t* reply_len)
  {
    int rc;

    if (option_DEBUG) {
      PNOTICE("nn_send (request): chksum=%x", Common::chksum32((void*) msg, msg_len));
      hexdump((void*) msg, 128);
    }

    rc = nn_send(_fd, msg, msg_len, 0);
    if (rc < 0) PWRN("nn_send failed unexpectedly");

    void* reply_msg = nn_allocmsg(MAX_REPLY_SIZE, 0);
    assert(reply_msg);
    rc = nn_recv(_fd, reply_msg, MAX_REPLY_SIZE, 0);
    assert(rc >= 0);
    *reply_len = rc;

    if (option_DEBUG) {
      PNOTICE("nn_recv(reply): chksum=%x", Common::chksum32(reply_msg, rc));
      hexdump(reply_msg, 128);
    }

    return reply_msg;
  }

 private:
  const int _fd{nn_socket(AF_SP, NN_REQ)};
};
}

#endif

#endif  // __COMANCHE_IPC__
