/*
   eXokernel Development Kit (XDK)

   Samsung Research America Copyright (C) 2013

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.
   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <http://www.gnu.org/licenses/>.

   As a special exception, if you link the code in this file with
   files compiled with a GNU compiler to produce an executable,
   that does not cause the resulting executable to be covered by
   the GNU Lesser General Public License.  This exception does not
   however invalidate any other reasons why the executable file
   might be covered by the GNU Lesser General Public License.
   This exception applies to code released by its copyright holders
   in files containing the exception.
*/

/*
  Authors:
  Copyright (C) 2015, Daniel G. Waddington <daniel.waddington@acm.org>
  Copyright (C) 2017, Daniel G. Waddington <daniel.waddington@ibm.com>
*/

#ifndef __COMPONENT_BASE_H__
#define __COMPONENT_BASE_H__

#include <assert.h>
#include <common/errors.h>
#include <common/types.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <atomic>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#define DECLARE_INTERFACE_UUID(f1, f2, f3, f4, f5, f6, f7, f8, f9, f10) \
  static Component::uuid_t &iid() {                                     \
    static Component::uuid_t itf_uuid = {                               \
        f1, f2, f3, f4, {f5, f6, f7, f8, f9, f10}};                     \
    return itf_uuid;                                                    \
  }

#define DECLARE_COMPONENT_UUID(f1, f2, f3, f4, f5, f6, f7, f8, f9, f10) \
  static Component::uuid_t &component_id() {                            \
    static Component::uuid_t comp_uuid = {                              \
        f1, f2, f3, f4, {f5, f6, f7, f8, f9, f10}};                     \
    return comp_uuid;                                                   \
  }

#define DECLARE_VERSION(X) \
  float version() override { return X; }

#define DECLARE_STATIC_COMPONENT_UUID(NAME, f1, f2, f3, f4, f5, f6, f7, f8, \
                                      f9, f10)                              \
  static Component::uuid_t NAME                                             \
      __attribute__((unused)) = {f1, f2, f3, f4, {f5, f6, f7, f8, f9, f10}}

#define DUMMY_IBASE_CONTROL                           \
  status_t start() override { return E_NOT_IMPL; }    \
  status_t stop() override { return E_NOT_IMPL; }     \
  status_t shutdown() override { return E_NOT_IMPL; } \
  status_t reset() override { return E_NOT_IMPL; }

namespace Component
{
/**
 * Standard UUID (Universal Unique Identifier) structure
 *
 */
struct uuid_t {
  uint32_t uuid0;
  uint16_t uuid1;
  uint16_t uuid2;
  uint16_t uuid3;
  uint8_t uuid4[6];

  std::string toString() {
    std::stringstream ss;
    ss << std::hex << std::setfill('0') << std::setw(8) << uuid0 << "-"
       << std::setfill('0') << std::setw(4) << uuid1 << "-" << std::setfill('0')
       << std::setw(4) << uuid2 << "-" << std::setfill('0') << std::setw(4)
       << uuid3 << "-" << std::setfill('0') << std::setw(2) << int(uuid4[0])
       << std::setfill('0') << std::setw(2) << int(uuid4[1])
       << std::setfill('0') << std::setw(2) << int(uuid4[2])
       << std::setfill('0') << std::setw(2) << int(uuid4[3])
       << std::setfill('0') << std::setw(2) << int(uuid4[4])
       << std::setfill('0') << std::setw(2) << int(uuid4[5]);

    return ss.str();
  }

  status_t fromString(const std::string &str) {
    unsigned long p0;
    unsigned int p1, p2, p3;
    unsigned int q[6];

    int err =
        sscanf(str.c_str(), "%08lx-%04x-%04x-%04x-%02x%02x%02x%02x%02x%02x",
               &p0, &p1, &p2, &p3, &q[0], &q[1], &q[2], &q[3], &q[4], &q[5]);

    if (err != 10) return E_FAIL;

    uuid0 = uint32_t(p0);
    uuid1 = uint16_t(p1);
    uuid2 = uint16_t(p2);
    uuid3 = uint16_t(p3);
    for (unsigned i = 0; i < 6; i++) uuid4[i] = uint8_t(q[i]);

    return S_OK;
  }
};

bool operator==(const Component::uuid_t &lhs, const Component::uuid_t &rhs);

/**
 * Base class. All components must inherit from this class.
 *
 */
class IBase {
 private:
  std::atomic<unsigned> _ref_count; /* component level reference counting */
  void *_dll_handle;

  IBase(IBase &) = delete;
  IBase &operator=(const IBase &) = delete;

 public:
  IBase() : _ref_count(0), _dll_handle(NULL) {}

  virtual ~IBase() {}

  /**
   * Pure virtual functions that should be implemented by all
   * components
   *
   */
  virtual void *query_interface(Component::uuid_t &itf) = 0;

  /* optional unload */
  virtual void unload() {}

  /**
   * [optional] Connect to another component.  Used for third-party binding. If
   * used
   * with release_bindings, the implementation should increment reference count.
   *
   * @param component Component which is offering its interface
   *
   * @return Number of connections remaining to be made. Returns -1
   * on error and 0 when all bindings are complete.
   */
  virtual int bind(IBase * /* component */) {
    return 0; /* by default, no bindings to perform */
  }

  /**
   * [optional] Called to connect to another component.
   *
   * @param component Component which is offering its interface
   * @param id Identifies the role of the binding
   *
   * @return Number of connections remaining to be made. Returns -1
   * on error and 0 when all bindings are complete.
   */
  virtual int specified_bind(IBase * /*component */, int /* id */) {
    return 0; /* by default, no bindings to perform */
  }

  /**
   * [optional] Release as many bindings as possible.
   *
   * @param component Component to unbind from.
   *
   * @return Return number of bindings remaining.
   */
  virtual int release_bindings() { return 0; /* default implementation */ }

  /**
   * Increment to reference count
   *
   */
  virtual void add_ref() { _ref_count++; }

  /**
   * Decrement reference count
   *
   */
  virtual void release_ref();

  /**
   * Get reference count
   *
   * @return Reference count
   */
  virtual unsigned ref_count() { return _ref_count.load(); }

  /**
   * Used as a dynamic invocation interface
   *
   * @param operation_string String version of operation to invoke
   * @param out_result Result of operation in string form
   *
   * @return S_OK on success.
   */
  virtual status_t invoke(std::string /*operation_string*/,
                          std::string & /*out_result*/) {
    return E_NOT_IMPL;
  }

  /**
   * Return the component version
   *
   *
   * @return 0.0;
   */
  virtual float version() { return 0.0; }

  /***************************/
  /** Control plane methods **/
  /***************************/

  /**
     Start component operation.  May be stopped through subsequent
     call to stop();
  */
  virtual status_t start() { return E_NOT_IMPL; }

  /**
     Stop the component operation.  May be restarted through a
     subsequent call to start();
  */
  virtual status_t stop() { return E_NOT_IMPL; }

  /**
     Shutdown component. Stop and exit threads.  Component
     cannot be restarted after shutdown.  Once shutdown, the
     component should be ready for destruction (delete).
  */
  virtual status_t shutdown() { return E_NOT_IMPL; }

  /**
     Flush buffers and state.
   */
  virtual status_t reset() { return E_NOT_IMPL; }

  virtual void set_dll_handle(void *dll) {
    assert(dll);
    _dll_handle = dll;
  }
};

/**
 * Called by the client to load the component from a DLL file
 *
 * @param dllname Name of dynamic library file
 * @param component_id Component UUID to load
 *
 * @return Pointer to IBase interface
 */
IBase *load_component(const char *dllname, Component::uuid_t component_id, bool quiet);
IBase *load_component(const char *dllname, Component::uuid_t component_id);

inline IBase *load_component(std::string &dllname,
                             Component::uuid_t component_id) {
  return load_component(dllname.c_str(), component_id);
}

/**
 * Third party binding of two or more components
 *
 * @param left
 * @param right
 *
 * @return
 */
status_t bind(std::vector<IBase *> components);
}  // namespace Component

#endif
