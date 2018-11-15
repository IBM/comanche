#ifndef _DAWN_POBJ_BAD_ALLOC_H
#define _DAWN_POBJ_BAD_ALLOC_H

#include <cstring>
#include <stdexcept>
#include <string>

class pobj_bad_alloc
	: public std::bad_alloc
{
	int _err;
	std::string _what;
public:
	pobj_bad_alloc(int err_)
		: _err(err_)
		, _what(std::string("pobj_bad_alloc: ") + strerror(_err))
	{}
	const char *what() const noexcept override
	{
		return _what.c_str();
	}
};

#endif
