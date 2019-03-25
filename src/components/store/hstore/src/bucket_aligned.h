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


#ifndef _COMANCHE_HSTORE_BUCKET_ALIGNED_H
#define _COMANCHE_HSTORE_BUCKET_ALIGNED_H

/*
 * A 64-byte aligned holder for data.
 */

namespace impl
{
	template <typename Data>
		struct alignas(64) bucket_aligned
			: public Data
		{
			using value_type = Data;
			bucket_aligned()
				: Data()
			{}
		};
}

#endif
