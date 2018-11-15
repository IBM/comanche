#ifndef _DAWN_COND_PRINT_H_
#define _DAWN_COND_PRINT_H_

/*
 * cond_print: Convert a value to a printable string, if possible,
 * or to an excuse if not.
 */

#include <sstream> /* ostream, ostringstream */
#include <string> /* string */
#include <type_traits> /* false_type, true_type */
#include <utility> /* declval */

namespace impl
{
	/* determine whether a type is printable */
	template <typename ...> using void_t = void;
	template <typename, typename = void>
		struct is_printable
			: std::false_type
		{};
	template <typename T>
		struct is_printable<
			T
			, void_t<
				decltype(
					std::declval<std::ostream &>() << std::declval<const T &>()
				)
			>
		>
			: std::true_type
		{};

	/* selection to print or not */
	template <bool printable, typename T>
		struct print_rep;

	template <typename T>
		struct print_rep<false, T>
		{
			static std::string print(const T&, const std::string &s)
			{
				return s;
			}
		};

	template <typename T>
		struct print_rep<true, T>
		{
			static std::string print(const T &t, const std::string &)
			{
				std::ostringstream s;
				s << t;
				return s.str();
			}
		};

	template <typename T>
		std::string cond_print(const T& t, const std::string &s)
		{
			return print_rep<is_printable<T>::value, T>::print(t, s);
		}
}

#endif
