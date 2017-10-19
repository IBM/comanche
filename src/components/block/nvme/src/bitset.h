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

#ifndef __COMANCHE_BITSET_H__
#define __COMANCHE_BITSET_H__

#include <stdint.h>

namespace comanche
{

template <unsigned NUM_BITS = 128>
class Bitset
{
private:
  uint64_t _qwords[(NUM_BITS/64)+1] __attribute__((aligned(8)));
  unsigned _rolling_qword_iter = 0;
  const unsigned NUM_QWORDS;
  
public:
  Bitset() : NUM_QWORDS(NUM_BITS/64) {
    static_assert(NUM_BITS % 64 == 0, "must be modulo 64");
    memset(_qwords,0,NUM_QWORDS*8);
  }

  inline bool operator[](unsigned pos) const
  {
    assert(pos < (NUM_QWORDS * 64));
    unsigned qword = pos / 64;
    unsigned bitpos = pos % 64;
    return (_qwords[qword] & (1ULL << bitpos));
  }

  inline bool get(unsigned pos) const
  {
    assert(pos < (NUM_QWORDS * 64));
    unsigned qword = pos / 64;
    unsigned bitpos = pos % 64;
    return (_qwords[qword] & (1ULL << bitpos));
  }

  inline void set(unsigned pos)
  {
    assert(pos < (NUM_QWORDS * 64));
    unsigned qword = pos / 64;
    unsigned bitpos = pos % 64;
    _qwords[qword] |= (1ULL << bitpos);
  }

  inline void clear(unsigned pos)
  {
    assert(pos < (NUM_QWORDS * 64));
    unsigned qword = pos / 64;
    unsigned bitpos = pos % 64;
    _qwords[qword] &= ~(1ULL << bitpos);
  }


  inline unsigned count() const {
    unsigned c = 0;
    for(unsigned i=0;i<NUM_QWORDS;i++) {
      c += __builtin_popcountll(_qwords[i]);
    }
    return c;
  }

  inline unsigned size() const {
    return NUM_BITS;
  }

  

  /** 
   * Before calling this, none() or count() should be used to
   * ascertain that a bit is set.
   * 
   * 
   * @return Position of bit that was set and is now cleared.
   */
  inline unsigned get_and_clear_next() {

#if 1
    while(_qwords[_rolling_qword_iter]==0) {
      _rolling_qword_iter++;
      if(_rolling_qword_iter == NUM_QWORDS) _rolling_qword_iter=0;
    }

    unsigned bitpos = __builtin_ctzll(_qwords[_rolling_qword_iter]);
    _qwords[_rolling_qword_iter] &= ~(1ULL << bitpos); /* clear bit */

    return ((_rolling_qword_iter*64)+bitpos);
#else
    /* temp */
    for(unsigned pos =0;pos < (NUM_QWORDS * 64); pos++) {
      if(get(pos)) {
        clear(pos);
        return pos;
      }
    }
    assert(0);
    return 0;
#endif
  }

  inline bool none() const
  {
    for(unsigned i=0;i<NUM_QWORDS;i++) {
      if(_qwords[i] > 0) return false;
    }
    return true;
  }
  
};



} // comanche
#endif // __COMANCHE_BITSET_H__
