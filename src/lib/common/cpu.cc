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
#include <common/cpu.h>
#include <common/errors.h>
#include <common/logging.h>
#include <stdint.h>
#include <string.h>
#include <boost/tokenizer.hpp>
#include <iostream>
#include <sstream>
#include <string>

int set_cpu_affinity_mask(cpu_mask_t& mask) {
#if defined(unix)
  return sched_setaffinity(0, mask.size(), mask.cpu_set());
#else
  PWRN("set_cpu_affinity_mask: not implemented");
  return -1;
#endif
}

/**
 * Convert comma separated list to cpu mask
 *
 * @param def
 * @param mask
 *
 * @return
 */
status_t string_to_mask(std::string def, cpu_mask_t &mask) {
  using namespace std;
  using namespace boost;

  if (def.find(",") == std::string::npos) {
    try {
      mask.add_core(stoi(def));
      return S_OK;
    } catch (std::invalid_argument e) {
      return E_INVAL;
    }
  }

  boost::char_separator<char> sep(",");
  vector<string> v;
  boost::tokenizer<boost::char_separator<char>> tok(def, sep);

  try {
    for_each(tok.begin(), tok.end(), [&](const string &s) {
      try {
        mask.add_core(stoi(s));
      } catch (std::invalid_argument e) {
        PWRN("invalid token in cpu mask string version.");
      }
    });
  } catch (...) {
    return E_FAIL;
  }

  return S_OK;
}
