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


#ifndef _COMANCHE_HSTORE_CONFIG_H_
#define _COMANCHE_HSTORE_CONFIG_H_

/*
 * USE_PMEM 1
 *   USE_CC_HEAP 0: allocation from pmemobj pool
 *   USE_CC_HEAP 1: simple allocation using actual addresses from a large region obtained from pmemobj
 *   USE_CC_HEAP 2: simple allocation using offsets from a large region obtained from pmemobj
 *   USE_CC_HEAP 3: AVL-based allocation using actual addresses from a large region obtained from pmemobj
 * USE_PMEM 0
 *   USE_CC_HEAP 1: simple allocation using actual addresses from a large region obtained from dax_map
 *   USE_CC_HEAP 2: simple allocation using offsets from a large region obtained from dax_map (NOT TESTED)
 *   USE_CC_HEAP 3: AVL-based allocation using actual addresses from a large region obtained from dax_map
 *
 */

#define USE_PMEM 0
#define USE_CC_HEAP 3
#define THREAD_SAFE_HASH 0

#endif
