/*
   Copyright [2018] [IBM Corporation]

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
 */

#include "fabric_check.h"

#include "fabric_runtime_error.h"

/* The man pages do not say where FI_SUCCESS is defined, but it turns out to be fi_errno.h */
#include <rdma/fi_errno.h> /* FI_SUCCESS */

/**
 * Fabric/RDMA-based network component
 *
 */

/* fi_fabric, fi_close (when called on a fabric) and most fi_poll functions FI_SUCCESS; others return 0 */
static_assert(FI_SUCCESS == 0, "FI_SUCCESS not zero");

/* most (all?) fabric functions return negative on error and 0 or positive on success */
unsigned check_ge_zero(int r, const char *file, int line)
{
  if ( r < 0 )
  {
    throw fabric_runtime_error(unsigned(-r), file, line);
  }
  return unsigned(r);
}

std::size_t check_ge_zero(ssize_t r, const char *file, int line)
{
  if ( r < 0 )
  {
    throw fabric_runtime_error(unsigned(-r), file, line);
  }
  return std::size_t(r);
}

std::size_t check_eq(ssize_t r, ssize_t exp, const char *file, int line)
{
  if ( r != exp )
  {
    throw fabric_runtime_error(unsigned(-r), file, line);
  }
  return std::size_t(r);
}
