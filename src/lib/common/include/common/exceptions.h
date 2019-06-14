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
  Copyright (C) 2017, Daniel G. Waddington <daniel.waddington@ibm.com>
*/

#ifndef __COMMON_EXCEPTIONS_H__
#define __COMMON_EXCEPTIONS_H__

#include <assert.h>
#include <common/types.h>
#include <cstdarg>
#include <string>
#include "errors.h"
#include "logging.h"

#ifndef STRINGIFY
#define STRINGIFY(x) #x
#endif

#define TOSTRING(x) STRINGIFY(x)
#define ADD_LOC(X) X __FILE__ ":" TOSTRING(__LINE__)

class Exception {
 public:
  Exception() {}

  Exception(const char *cause) {
    __builtin_strncpy(_cause, cause, 256);
    PEXCEP("%s", cause);
    asm("int3");
  }

  const char *cause() const { return _cause; }

  void set_cause(const char *cause) {
    __builtin_strncpy(_cause, cause, 256);
    PEXCEP("%s", cause);
  }

 private:
  char _cause[256];
};

class Constructor_exception : public Exception {
 public:
  Constructor_exception(int err)
      : Exception("Constructor failed"), _err_code(err) {}

  Constructor_exception()
      : Constructor_exception(E_FAIL) {}

  __attribute__((__format__(__printf__, 2, 0)))
  Constructor_exception(const char *fmt, ...)
      : Constructor_exception() {
    va_list args;
    va_start(args, fmt);
    char msg[255] = {0};
    vsnprintf(msg, 254, fmt, args);
    set_cause(msg);
  }

  status_t error_code() const { return _err_code; }

 private:
  status_t _err_code;
};

class General_exception : public Exception {
 public:
  General_exception(int err) : Exception("General exception"), _err_code(err) {}

  General_exception() : General_exception(E_FAIL) {}

  __attribute__((__format__(__printf__, 2, 0)))
  General_exception(const char *fmt, ...)
      : General_exception() {
    va_list args;
    va_start(args, fmt);
    char msg[255] = {0};
    vsnprintf(msg, 254, fmt, args);
    set_cause(msg);
  }
  status_t error_code() const { return _err_code; }

 private:
  status_t _err_code;
};

class API_exception : public Exception {
 public:
  API_exception(int err) : Exception("API error"), _err_code(err) {}

  API_exception() : API_exception(E_FAIL) {}

  __attribute__((__format__(__printf__, 2, 0)))
  API_exception(const char *fmt, ...)
      : API_exception() {
    va_list args;
    va_start(args, fmt);
    char msg[255] = {0};
    vsnprintf(msg, 254, fmt, args);
    set_cause(msg);
  }

  status_t error_code() const { return _err_code; }

 private:
  status_t _err_code;
};

class Logic_exception : public Exception {
 public:
  Logic_exception(int err) : Exception("Logic error"), _err_code(err) {}

  Logic_exception() : Logic_exception(E_FAIL) {}

  __attribute__((__format__(__printf__, 2, 0)))
  Logic_exception(const char *fmt, ...)
      : Logic_exception() {
    va_list args;
    va_start(args, fmt);
    char msg[255] = {0};
    vsnprintf(msg, 254, fmt, args);
    set_cause(msg);
  }

  status_t error_code() const { return _err_code; }

 private:
  status_t _err_code;
};

class IO_exception : public Exception {
 public:
  IO_exception(int err) : Exception("IO error"), _err_code(err) {}

  IO_exception() : IO_exception(E_FAIL) {}

  __attribute__((__format__(__printf__, 2, 0)))
  IO_exception(const char *fmt, ...)
      : IO_exception() {
    va_list args;
    va_start(args, fmt);
    char msg[255] = {0};
    vsnprintf(msg, 254, fmt, args);
    set_cause(msg);
  }

  status_t error_code() const { return _err_code; }

 private:
  status_t _err_code;
};

class Program_exception : public Exception {
 public:
  Program_exception(int err) : Exception("Program error"), _err_code(err) {}

  Program_exception() : Program_exception(E_FAIL) {}

  __attribute__((__format__(__printf__, 2, 0)))
  Program_exception(const char *fmt, ...)
      : Program_exception() {
    va_list args;
    va_start(args, fmt);
    char msg[255] = {0};
    vsnprintf(msg, 254, fmt, args);
    set_cause(msg);
  }

  status_t error_code() const { return _err_code; }

 private:
  status_t _err_code;
};

class Data_exception : public Exception {
 public:
  Data_exception(int err) : Exception("Data error"), _err_code(err) {}

  Data_exception() : Data_exception(E_FAIL) {}

  __attribute__((__format__(__printf__, 2, 0)))
  Data_exception(const char *fmt, ...)
      : Data_exception() {
    va_list args;
    va_start(args, fmt);
    char msg[255] = {0};
    vsnprintf(msg, 254, fmt, args);
    set_cause(msg);
  }

  status_t error_code() const { return _err_code; }

 private:
  status_t _err_code;
};

class Protocol_exception : public Exception {
 public:
  Protocol_exception(int err) : Exception("Protocol error"), _err_code(err) {}

  Protocol_exception() : Protocol_exception(E_FAIL) {}

  __attribute__((__format__(__printf__, 2, 0)))
  Protocol_exception(const char *fmt, ...)
      : Protocol_exception() {
    va_list args;
    va_start(args, fmt);
    char msg[255] = {0};
    vsnprintf(msg, 254, fmt, args);
    set_cause(msg);
  }

  status_t error_code() const { return _err_code; }

 private:
  status_t _err_code;
};

#endif
