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


#ifndef COMANCHE_NUPM_ALLOCATOR_RA_H
#define COMANCHE_NUPM_ALLOCATOR_RA_H

#include "deallocator_ra.h"

#include "bad_alloc.h"

#include <cstddef> /* size_t, ptrdiff_t */

namespace nupm
{
	template <typename T>
		class allocator_ra;

	template <>
		class allocator_ra<void>
			: public deallocator_ra<void>
		{
		public:
			using deallocator_type = deallocator_ra<void>;
			using typename deallocator_type::pointer;
			using typename deallocator_type::const_pointer;
			using typename deallocator_type::value_type;

			template <typename U>
				struct rebind
				{
					using other = allocator_ra<U>;
				};
		};

	template <typename T>
		class allocator_ra
			: public deallocator_ra<T>
		{
		public:
			using deallocator_type = deallocator_ra<T>;
			using size_type = std::size_t;
			using difference_type = std::ptrdiff_t;
			using typename deallocator_type::pointer;
			using typename deallocator_type::const_pointer;
			using reference = T &;
			using const_reference = const T &;
			using typename deallocator_type::value_type; // = T;

			template <typename U>
				struct rebind
				{
					using other = allocator_ra<U>;
				};

			allocator_ra(Core::AVL_range_allocator &ra_) noexcept
				: deallocator_ra<T>(ra_)
			{}

			allocator_ra(const allocator_ra &a_) noexcept = default;

			template <typename U>
				allocator_ra(const allocator_ra<U> &a_) noexcept
					: allocator_ra(a_.ra())
				{}

			allocator_ra &operator=(const allocator_ra &a_) = delete;

#if 0
			/* deprecated in C++20 */
			pointer address(reference x) const noexcept
			{
				return pointer(&x);
			}
			const_pointer address(const_reference x) const noexcept
			{
				return pointer(&x);
			}
#endif
			auto allocate(
				size_type s
				, typename allocator_ra<void>::const_pointer /* hint */ =
					typename allocator_ra<void>::const_pointer{}
				, const char * = nullptr
			) -> pointer
			{
				auto ptr = this->ra()->alloc(s * sizeof(T), alignof(T))->addr();
				if ( ptr == 0 )
				{
					throw bad_alloc(0, s, sizeof(T));
				}
				return reinterpret_cast<pointer>(ptr);
			}
		};
}
#endif
