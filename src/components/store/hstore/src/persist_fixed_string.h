#ifndef DAWN_HSTORE_PERSIST_FIXED_STRING_H
#define DAWN_HSTORE_PERSIST_FIXED_STRING_H

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
		uint64_t size(access) const noexcept { return _size; }
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

		ptr_t ptr;
		static_assert(
			sizeof ptr <= sizeof small.value
			, "_ptr conflicts with small.size"
		);
		rep()
			: small(0)
		{
		}

		template <typename IT>
			rep(
				IT first_
				, IT last_
				, Allocator al_
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
					ptr =
						ptr_t(
							typename allocator_void_type::pointer(
								allocator_char_type(al_).allocate(sizeof(element_type) + data_size)
							)
						);
					new (&*ptr) element_type(first_, last_, access{});
				}
			}

		rep(
			std::size_t data_len_
			, Allocator al_
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
				ptr =
					ptr_t(
						typename allocator_void_type::pointer(
							allocator_char_type(al_).allocate(sizeof(element_type) + data_size)
						)
					);
				new (&*ptr) element_type(data_size, access{});
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
				new (&ptr) ptr_t(other.ptr);
				if ( ptr )
				{
					ptr->inc_ref(access{});
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
				new (&ptr) ptr_t(other.ptr);
				other.ptr = typename allocator_type::pointer{};
			}
		}

		rep &operator=(const rep &other)
		{
			if ( other.is_small() )
			{
				small = other.small;
			}
			else
			{
				auto p = ptr;
				small = other.small;
				ptr = other.ptr;
				if ( ptr )
				{
					ptr->inc_ref(access{});
				}

				if ( p )
				{
					if ( p->dec_ref(access{}) == 0 )
					{
						/* Note: in this application (hashed key-value) we never
						 * expect to delete a value through operator=, but only
						 * through ~persist_fixed_string
						 */
						zfree(p, "data operator=");
					}
				}
			}
			return *this;
		}

		rep &operator=(rep &&other)
		{
			if ( other.is_small() )
			{
				small = other.small;
			}
			else
			{
				small = other.small;
				ptr = other.ptr;
				other.ptr = ptr_t();
			}
			return *this;
		}

		~rep()
		{
			if ( ! is_small() && ptr )
			{
				if ( ptr->dec_ref(access{}) == 0 )
				{
					ptr->~element_type();
					zfree(ptr, "data dtor");
				}
			}
		}

		static constexpr uint8_t large_kind = sizeof small.value + 1;
		bool is_small() const
	{
		assert(small._size <= large_kind);
		return small._size < large_kind;
	}

		std::size_t size() const
		{
			return is_small() ? small._size : ptr->size(access{});
		}

		const T *data() const
		{
			if ( is_small() )
			{
				return static_cast<const T *>(&small.value[0]);
			}
			auto pt = static_cast<element_type *>(&*ptr);
			return static_cast<const T *>(static_cast<void *>(pt+1));
		}

		T *data()
		{
			if ( is_small() )
			{
				return static_cast<T *>(&small.value[0]);
			}
			auto pt = static_cast<element_type *>(&*ptr);
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

		template <typename IT>
			persist_fixed_string(
				IT first_
				, IT last_
				, Allocator al_
			)
				: _rep(first_, last_, al_)
		{
		}

		persist_fixed_string(
			std::size_t data_len_
			, Allocator al_
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
