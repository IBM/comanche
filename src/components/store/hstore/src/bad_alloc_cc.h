/*
 * (C) Copyright IBM Corporation 2018, 2019. All rights reserved.
 * US Government Users Restricted Rights - Use, duplication or disclosure restricted by GSA ADP Schedule Contract with IBM Corp.
 */

#ifndef COMANCHE_HSTORE_BAD_ALLOC_H
#define COMANCHE_HSTORE_BAD_ALLOC_H

#include <cstddef> /* size_t */
#include <new> /* bad_alloc */
#include <string>

class bad_alloc_cc
	: public std::bad_alloc
{
	std::string _what;
public:
	bad_alloc_cc(std::size_t pad, std::size_t count, std::size_t size)
		: _what(std::string(__func__) + ": " + std::to_string(pad) + "+" + std::to_string(count) + "*" + std::to_string(size))
	{}
	const char *what() const noexcept override
	{
		return _what.c_str();
	}
};

#endif
