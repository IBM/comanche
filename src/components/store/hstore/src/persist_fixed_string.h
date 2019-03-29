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


#ifndef COMANCHE_HSTORE_PERSIST_FIXED_STRING_H
#define COMANCHE_HSTORE_PERSIST_FIXED_STRING_H

#include "persistent.h"

#include <algorithm>
#include <cstddef> /* size_t */
#include <tuple>

template <typename T, typename Allocator>
	class persist_fixed_string;

template <typename T, typename Allocator>
	union rep;

class fixed_string_access
{
	fixed_string_access() {}
public:
	template <typename T, typename Allocator>
		friend union rep;
};

/*
 * - fixed_string
 * - ref_count: because the object may be referenced twice as the table expands
 * - size: length of data (data immediately follows the fixed_string object)
 */
template <typename T>
	class fixed_string
	{
		unsigned _ref_count;
		uint64_t _size;
		uint64_t size() const { return _size; }
	public:
		using access = fixed_string_access;
		template <typename IT>
			fixed_string(IT first_, IT last_, access a_)
				: fixed_string(static_cast<std::size_t>(last_-first_), a_)
			{
				/* for small lengths we copy */
				std::copy(first_, last_, static_cast<T *>(static_cast<void *>(this+1)));
			}
		fixed_string(std::size_t data_len_, access)
			: _ref_count(1U)
			, _size(data_len_)
		{
		}
		template <typename Allocator>
			void persist(Allocator al_)
			{
				al_.persist(this, sizeof *this + size());
			}
		uint64_t size(access) const noexcept { return size(); }
		unsigned inc_ref(access) noexcept { return _ref_count++; }
		unsigned dec_ref(access) noexcept { return --_ref_count; }
	};

template <typename T, typename Allocator>
	union rep
	{
		using element_type = fixed_string<T>;
		using allocator_type = typename Allocator::template rebind<element_type>::other;
		using allocator_char_type = typename allocator_type::template rebind<char>::other;
		using allocator_void_type = typename allocator_type::template rebind<void>::other;
		using ptr_t = persistent_t<typename allocator_type::pointer>;
		using access = fixed_string_access;

		struct small_t
		{
			char value[23];
			std::uint8_t _size; /* discriminant */
			static constexpr uint8_t large_kind = sizeof value + 1;
			/* note: as of C++17, can use std::clamp */
			small_t(std::size_t size_)
				: _size(size_ <= sizeof value ? uint8_t(size_) : large_kind)
			{}
		} small;

		struct large_t
			: public allocator_char_type
		{
			allocator_char_type &al() { return static_cast<allocator_char_type &>(*this); }
			const allocator_char_type &al() const { return static_cast<const allocator_char_type &>(*this); }
			ptr_t ptr;
		} large;

		static_assert(
			sizeof large <= sizeof small.value
			, "large_t overlays with small.size"
		);

		/* ERROR: need to persist */
		rep()
			: small(0)
		{
		}

		template <typename IT, typename AL>
			rep(
				IT first_
				, IT last_
				, AL al_
			)
				: small(static_cast<std::size_t>(last_ - first_) * sizeof(T))
			{
				if ( is_small() )
				{
					std::copy(
						first_
						, last_
						, static_cast<T *>(static_cast<void *>(&small.value[0]))
					);
				}
				else
				{
					auto data_size =
						static_cast<std::size_t>(last_ - first_) * sizeof(T);
					using local_allocator_char_type = typename AL::template rebind<char>::other;
					new (&large.al()) allocator_char_type(al_);
					new (&large.ptr)
						ptr_t(
							static_cast<typename allocator_type::pointer>(
								typename allocator_void_type::pointer(
                                                      local_allocator_char_type(al_).allocate(sizeof(element_type) + data_size)
								)
							)
						);
					new (&*large.ptr) element_type(first_, last_, access{});
					large.ptr->persist(al_);
				}
			}

		template <typename AL>
			rep(
				std::size_t data_len_
				, AL al_
			)
				: small(data_len_ * sizeof(T))
			{
				/* large data sizes: data not copied, but space is reserved for RDMA */
				if ( is_small() )
				{
				}
				else
				{
					auto data_size = data_len_ * sizeof(T);
					new (&large.al()) allocator_char_type(al_);
					new (&large.ptr)
						ptr_t(
								static_cast<typename allocator_type::pointer>(
								typename allocator_void_type::pointer(
									al_.allocate(sizeof(element_type) + data_size)
								)
							)
						);
					new (&*large.ptr) element_type(data_size, access{});
				}
			}

		rep(const rep &other)
			: small(other.small._size)
		{
			if ( is_small() )
			{
				small = other.small;
			}
			else
			{
				small = other.small;
				new (&large.al()) allocator_char_type(other.large.al());
				new (&large.ptr) ptr_t(other.large.ptr);
				if ( large.ptr )
				{
					large.ptr->inc_ref(access{});
				}
			}
		}

		rep(rep &&other)
			: small(other.small.size)
		{
			if ( is_small() )
			{
				small = other.small;
			}
			else
			{
				small = other.small;
				large.al = other.large.al;
				new (&large.ptr) ptr_t(other.large.ptr);
				other.large.ptr = typename allocator_type::pointer{};
			}
		}

		rep &operator=(const rep &other)
		{
			if ( is_small() )
			{
				if ( other.is_small() )
				{
					/* small <- small */
					small = other.small;
				}
				else
				{
					/* small <- large */
					small = other.small; /* for "large" flag */
					new (&large.al()) allocator_type(other.large.al());
					new (&large.ptr) ptr_t(other.large.ptr);
					large.ptr->inc_ref(access{});
				}
			}
			else
			{
				/* large <- ? */
				if ( large.ptr && large.ptr->dec_ref(access{}) == 0 )
				{
					auto sz = sizeof *large.ptr + large.ptr->size(access());
					large.ptr->~element_type();
					large.al().deallocate(static_cast<typename allocator_char_type::pointer>(static_cast<typename allocator_void_type::pointer>(large.ptr)), sz);
				}
				large.al().~allocator_char_type();

				small = other.small; /* for "large" flag */

				if ( other.is_small() )
				{
					/* large <- small */
				}
				else
				{
					/* large <- large */
					large.ptr = other.large.ptr;
					large.ptr->inc_ref(access{});
					new (&large.al()) allocator_type(other.large.al());
				}
			}
			return *this;
		}

		rep &operator=(rep &&other)
		{
			if ( is_small() )
			{
				if ( other.is_small() )
				{
					/* small <- small */
					small = std::move(other.small);
				}
				else
				{
					/* small <- large */
					small = std::move(other.small); /* for "large" flag */
					new (&large.al()) allocator_type(std::move(other.large.al()));
					new (&large.ptr) ptr_t(std::move(other.large.ptr));
					other.large.ptr = ptr_t();
				}
			}
			else
			{
				if ( large.ptr && large.ptr->dec_ref(access{}) == 0 )
				{
					auto sz = sizeof *large.ptr + large.ptr->size(access());
					large.ptr->~element_type();
					large.al().deallocate(static_cast<typename allocator_char_type::pointer>(static_cast<typename allocator_void_type::pointer>(large.ptr)), sz);
				}
				large.al().~allocator_char_type();

				small = std::move(other.small); /* for "large" flag */

				if ( other.is_small() )
				{
					/* large <- small */
				}
				else
				{
					/* large <- large */
					large.ptr = other.large.ptr;
					other.large.ptr = ptr_t();
					new (&large.al()) allocator_type(other.large.al());
				}
			}
			return *this;
		}

		~rep()
		{
			if ( ! is_small() && large.ptr )
			{
				if ( large.ptr->dec_ref(access{}) == 0 )
				{
					auto sz = sizeof *large.ptr + large.ptr->size(access());
					large.ptr->~element_type();
					large.al().deallocate(static_cast<typename allocator_char_type::pointer>(static_cast<typename allocator_void_type::pointer>(large.ptr)), sz);
				}
			}
		}

		static constexpr uint8_t large_kind = sizeof small.value + 1;

		template <typename AL>
			void reconstitute(AL al_) const
			{
				using reallocator_char_type = typename AL::template rebind<char>::other;
				if ( ! is_small() )
				{
					auto alr = reallocator_char_type(al_);
					if ( alr.is_reconstituted(large.ptr) )
					{
						/* The data has already been reconstituted. Increase the reference count. */
						large.ptr->inc_ref(access());
					}
					else
					{
						/* The data is not yet reconstituted. Reconstitute it.
						 * Although the original may have had a refcount
						 * greater than one, we have not yet seene the
						 * second reference, so the recount must be set to one.
						 */
						alr.reconstitute(sizeof *large.ptr + size(), large.ptr);
						new (large.ptr) element_type( size(), access{} );
					}
				}
			}

		bool is_small() const
		{
			assert(small._size <= large_kind);
			return small._size < large_kind;
		}

		std::size_t size() const
		{
			return is_small() ? small._size : large.ptr->size(access{});
		}

		const T *data() const
		{
			if ( is_small() )
			{
				return static_cast<const T *>(&small.value[0]);
			}
			auto pt = static_cast<element_type *>(&*large.ptr);
			return static_cast<const T *>(static_cast<void *>(pt+1));
		}

		T *data()
		{
			if ( is_small() )
			{
				return static_cast<T *>(&small.value[0]);
			}
			auto pt = static_cast<element_type *>(&*large.ptr);
			return static_cast<T *>(static_cast<void *>(pt+1));
		}

	};

template <typename T, typename Allocator>
	class persist_fixed_string
	{
		using access = fixed_string_access;
		using element_type = fixed_string<T>;
		using EA = typename Allocator::template rebind<element_type>::other;
		using ptr_t = persistent_t<typename EA::pointer>;
		/* "rep" is most of persist_fixed_string; it is conceptually its base class
		 * It does not directly replace persist_fixed_string only to preserve the
		 * declaration of persist_fixed_string as a class, not a union
		 */
		rep<T, Allocator> _rep;
		/* NOTE: allocating the data string adjacent to the header of a fixed_string
		 * precludes use of a standard allocator
		 */

	public:
		using allocator_type = Allocator;
		persist_fixed_string()
			: _rep()
		{
		}

		template <typename IT, typename AL>
			persist_fixed_string(
				IT first_
				, IT last_
				, AL al_
			)
				: _rep(first_, last_, al_)
		{
		}

		template <typename AL>
			persist_fixed_string(
				std::size_t data_len_
				, AL al_
			)
				: _rep(data_len_, al_)
			{
			}

		persist_fixed_string(const persist_fixed_string &other)
			: _rep(other._rep)
		{
		}

		persist_fixed_string(persist_fixed_string &&other)
			: _rep(other._rep)
		{
		}

		persist_fixed_string &operator=(const persist_fixed_string &other) = default;
		persist_fixed_string &operator=(persist_fixed_string &&other) = default;

		~persist_fixed_string()
		{
		}

		std::size_t size() const { return _rep.size(); }

		const T *data() const { return _rep.data(); }

		T *data() { return _rep.data(); }

		template <typename AL>
			void reconstitute(AL al_) const { return _rep.reconstitute(al_); }
	};

template <typename T, typename Allocator>
	bool operator==(
		const persist_fixed_string<T, Allocator> &a
		, const persist_fixed_string<T, Allocator> &b
	)
	{
		return
			a.size() == b.size()
			&&
			std::equal(a.data(), a.data() + a.size(), b.data())
		;
	}

#endif
