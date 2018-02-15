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
  Authors:

  Copyright (C) 2016, Daniel G. Waddington <daniel.waddington@ibm.com>
*/

#ifndef __COMANCHE_LF_BITMAP_H__
#define __COMANCHE_LF_BITMAP_H__

#include <assert.h>
#include <stdint.h>

namespace Core
{

/* Relocatable and lock-free bitmap - use inplace new operator ? */

class Relocatable_LF_bitmap
{
public:

  Relocatable_LF_bitmap(void * arena,
                        size_t arena_size,
                        size_t n_elements,
                        bool reconstruct = false) {

    if(n_elements % 8)
      throw API_exception("n_elements must be modulo 8");
    if(arena_size * 8 < n_elements)
      throw API_exception("insuffient arena for requested elements");

    _bitmap = (uint64_t *) arena;
    
    if(!reconstruct)
      memset(arena, 0, arena_size);
  }

  size_t get_element_range() const {
    return _n_elements;
  }

  size_t expand_elements(size_t new_size) {
    if(new_size % 8)
      throw API_exception("new_size must be modulo 8");
    
    return _n_elements;
  }

  unsigned long allocate() {
  }

  void free(unsigned long slot) {
  }
  
private:
  struct {
    uint32_t magic;
    uint64_t bitmap;
  } * _hdr;
  size_t     _arena_size;
  size_t     _n_elements;
};
  
}

#endif
