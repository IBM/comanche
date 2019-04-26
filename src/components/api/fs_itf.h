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

#ifndef __COMANCHE_API_FS_ITF__
#define __COMANCHE_API_FS_ITF__

#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus

#include <cstdint>
#include <component/base.h>
#include "memory_itf.h"

#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>
#include <sys/stat.h>

namespace Component
{

class IFile
{
public:

  DECLARE_INTERFACE_UUID(0x5e385221,0xf3db,0x4c02,0xabab,0xea,0x7c,0x91,0x65,0x24,0xbe);
  
  virtual off_t lseek(const off_t offset, const int whence) = 0;
  virtual ssize_t read(const void * buf, const size_t nbyte) = 0;
  virtual ssize_t write(const void * buf, const size_t nbyte) = 0;
  virtual size_t fsync() = 0;
};


/** 
 * Block device interface
 * 
 */
class IFile_system : public IZerocopy_memory
{
public:

  DECLARE_INTERFACE_UUID(0x2a7f74eb,0xd35f,0x44ff,0x92d7,0x21,0xeb,0x6b,0xba,0xb2,0x94);
  
  virtual IFile* create_file(const std::string path, const std::string flags) = 0;
  virtual IFile* open(const std::string path, const std::string flags) = 0;
  virtual void delete_file(const std::string path) = 0;
  virtual void close(IFile* handle) = 0;
};

} //< namespace Component


#else
#error C API not implemented.
#endif

#endif 
