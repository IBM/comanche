/*
 * (C) Copyright IBM Corporation 2018. All rights reserved.
 * US Government Users Restricted Rights - Use, duplication or disclosure restricted by GSA ADP Schedule Contract with IBM Corp.
 */

#ifndef _COMANCHE_HSTORE_POBJ_BAD_ALLOC_H
#define _COMANCHE_HSTORE_POBJ_BAD_ALLOC_H

#include <cstring>
#include <stdexcept>
#include <string>

class pobj_bad_alloc
	: public std::bad_alloc
{
	int _err;
	std::string _what;
public:
	/* Note: it would make more senst for count, not size, to vary */
	pobj_bad_alloc(std::size_t pad, std::size_t count, std::size_t size_min, std::size_t size_max, int err_)
		: _err(err_)
		, _what(std::string("pobj_bad_alloc: ") + std::to_string(pad) + "+" + std::to_string(count) + "*" + std::to_string(size_min) + ".." + std::to_string(size_max) + " " + strerror(_err))
	{}
	const char *what() const noexcept override
	{
		return _what.c_str();
	}
};

#endif
