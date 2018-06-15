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

#ifndef _FABRIC_CHECK_H_
#define _FABRIC_CHECK_H_

/*
 * Authors:
 *
 */

#include <unistd.h> /* ssize_t */

#include <cstddef> /* size_t */

/**
 * Fabric/RDMA-based network component
 *
 */

/* fi_fabric, fi_close (when called on a fabric) and most fi_poll functions FI_SUCCESS; others return 0 */
unsigned (check_ge_zero)(int r, const char *file, int line);
std::size_t (check_ge_zero)(ssize_t r, const char *file, int line);

#define CHECK_FI_ERR(V) (check_ge_zero)((V), __FILE__, __LINE__)

#endif
