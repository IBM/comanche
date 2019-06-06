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


#ifndef COMANCHE_NUPM_DEALLOCATOR_RA_H
#define COMANCHE_NUPM_DEALLOCATOR_RA_H

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Weffc++"
#pragma GCC diagnostic ignored "-Wformat="
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wredundant-decls"
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <core/avl_malloc.h>
#pragma GCC diagnostic pop

#include <cstddef> /* size_t, ptrdiff_t */

namespace nupm
{
	template <typename T>
		class deallocator_ra;

	template <>
		class deallocator_ra<void>
		{
		public:
			using pointer = void *;
			using const_pointer = const void *;
			using value_type = void;
			template <typename U>
				struct rebind
				{
					using other = deallocator_ra<U>;
				};
		};

	template <typename T>
		class deallocator_ra
		{
			Core::AVL_range_allocator *_ra;
		public:
			using size_type = std::size_t;
			using difference_type = std::ptrdiff_t;
			using pointer = T*;
			using const_pointer = const T*;
			using reference = T &;
			using const_reference = const T &;
			using value_type = T;

		template <typename U>
			struct rebind
			{
				using other = deallocator_ra<U>;
			};

			explicit deallocator_ra(Core::AVL_range_allocator &ra_) noexcept
				: _ra(&ra_)
			{}

			explicit deallocator_ra(const deallocator_ra &) noexcept = default;

			template <typename U>
				explicit deallocator_ra(const deallocator_ra<U> &d_) noexcept
					: deallocator_ra(d_.ra())
				{}

			deallocator_ra &operator=(const deallocator_ra &e_) = delete;

			pointer address(reference x) const noexcept
			{
				return pointer(&x);
			}
			const_pointer address(const_reference x) const noexcept
			{
				return pointer(&x);
			}

			void deallocate(
				pointer p
				, size_type
			)
			{
				_ra->free(reinterpret_cast<addr_t>(p)
					/* size is offered by dealllocate but not accepted by free */
#if 0
					, sizeof(T) * sz_
#endif
				);
			}

			auto max_size() const
			{
				return 8; /* reminder to provide a proper max size value */
			}

			auto ra() const
			{
				return _ra;
			}
		};
}

#endif
