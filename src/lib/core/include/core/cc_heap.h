#ifndef CORE_ALLOCATOR
#define CORE_ALLOCATOR

#include <array>
#include <cstddef> /* size_t, ptrdiff_t */
#include <sstream> /* ostringstream */
#include <new> /* bad_alloc */
#include <string>

namespace Core
{
	class sbrk_alloc
	{
		struct bound
		{
			char *_end;
			void set(char *e) noexcept { _end = e; }
			char *end() const noexcept { return _end; }
		};
		struct state /* persists */
		{
			static const std::uint64_t magic_value = 0x47b3a47358294161;
			std::uint64_t _magic; /* persists. Initially not magic (we hope). */
			void *_location; /* persists. contains its own expected address */
			unsigned _sw; /* persists. Initially 0. Toggles between 0 and 1 */
			char *_limit; /* persists */
			std::array<bound, 2U> _bounds; /* persists, depends on _sw */
			bound &current() { return _bounds[_sw]; }
			bound &other() { return _bounds[1U-_sw]; }
			char *begin() { return static_cast<char *>(static_cast<void *>(this+1)); }
			void swap() { _sw = 1U - _sw; }
			void *limit() const { return _limit; }
		};
		bound &current() { return _state->current(); }
		bound &other() { return _state->other(); }
		void swap() { _state->swap(); }
		state *_state;
		template <typename T>
			void persist(const T &) {}
		void restore() const
		{
			if ( _state->_location != &_state->_location )
			{
				std::ostringstream s;
				s << "cc_heap region mapped at " << &_state->_location << " but required to be at " << _state->_location;
				throw std::runtime_error{s.str()};
			}
			assert(_state->_sw < _state->_bounds.size());
		}
	public:
		explicit sbrk_alloc(void *area, std::size_t sz)
			: _state(static_cast<state *>(area))
		{
			if ( _state->_magic != state::magic_value )
			{
				/* one-time initialization; assumes that initial bytes in area are zeros/nullptr */
				_state->_limit = static_cast<char *>(area) + sz;
				_state->_bounds[0]._end = _state->begin();
				_state->_sw = 0;
				_state->_location = &_state->_location;
				_state->_magic = state::magic_value;
				persist(_state);
			}
			else
			{
				restore();
			}
		}
		explicit sbrk_alloc(void *area)
			: _state(static_cast<state *>(area))
		{
			restore();
		}
		void *malloc(std::size_t sz)
		{
			/* round to double word */
			sz = (sz + 7UL) & ~7UL;
			if ( static_cast<std::size_t>(_state->_limit - current()._end) < sz ) { return nullptr; }
			auto p = current().end();
			auto q = p + sz;
			other().set(q);
			persist(other());
			swap();
			persist(_state->_sw);
			return p;
		}
		void free(const void *) {}
		void *area() const { return _state; }
	};

	class cc_alloc
		: public sbrk_alloc
	{
	public:
		explicit cc_alloc(void *area, std::size_t sz)
			: sbrk_alloc(area, sz)
		{}
		explicit cc_alloc(void *area)
			: sbrk_alloc(area)
		{}
	};

	class bad_alloc_cc
		: public std::bad_alloc
	{
		std::string _what;
	public:
		bad_alloc_cc(std::size_t pad, std::size_t count, std::size_t size)
			: _what(std::string("bad_alloc_cc: ") + std::to_string(pad) + "+" + std::to_string(count) + "*" + std::to_string(size))
		{}
		const char *what() const noexcept override
		{
			return _what.c_str();
		}
	};

	class persister
	{
	public:
		void persist(const void *, std::size_t) {}
	};

	template <typename T, typename Persister>
		class CC_allocator;

	template <>
		class CC_allocator<void, persister>
		{
		public:
			using pointer = void *;
			using const_pointer = const void *;
			using value_type = void;
			template <typename U>
				struct rebind
				{
					using other = CC_allocator<U, persister>;
				};
		};

	template <typename Persister>
		class CC_allocator<void, Persister>
		{
		public:
			using pointer = void *;
			using const_pointer = const void *;
			using value_type = void;
			template <typename U>
				struct rebind
				{
					using other = CC_allocator<U, Persister>;
				};
		};

	template <typename T, typename Persister = persister>
		class CC_allocator
			: public Persister
		{
			cc_alloc _pool;
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
					using other = CC_allocator<U, Persister>;
				};

			CC_allocator(void *area_, std::size_t size_, Persister p_ = Persister())
				: Persister(p_)
				, _pool(area_, size_)
			{}

			CC_allocator(void *area_, Persister p_ = Persister())
				: Persister(p_)
				, _pool(area_)
			{}

			CC_allocator(const cc_alloc &pool_, Persister p_ = Persister()) noexcept
				: Persister(p_)
				, _pool(pool_)
			{}

			CC_allocator(const CC_allocator &a_) noexcept = default;

			template <typename U, typename P>
				CC_allocator(const CC_allocator<U, P> &a_) noexcept
					: CC_allocator(a_.pool())
				{}

			CC_allocator &operator=(const CC_allocator &a_) = delete;

			pointer address(reference x) const noexcept
			{
				return pointer(&x);
			}
			const_pointer address(const_reference x) const noexcept
			{
				return pointer(&x);
			}

			auto allocate(
				size_type s
				, typename CC_allocator<void, Persister>::const_pointer /* hint */ =
						typename CC_allocator<void, Persister>::const_pointer{}
				, const char * = nullptr
			) -> pointer
			{
				auto ptr = _pool.malloc(s * sizeof(T));
				if ( ptr == 0 )
				{
					throw bad_alloc_cc(0, s, sizeof(T));
				}
				return static_cast<pointer>(ptr);
			}
			void deallocate(
				pointer p
				, size_type
			)
			{
				_pool.free(p);
			}
			auto max_size() const
			{
				return 8; /* reminder to provide a proper max size value */
			}
			void persist(const void *ptr, size_type len, const char * = nullptr)
			{
				Persister::persist(ptr, len);
			}
			auto pool() const
			{
				return _pool;
			}
		};
}

#endif
