#ifndef _DANW_PERSISTER_H_
#define _DANW_PERSISTER_H_

#include <cstddef> /* size_t */

class persister
{
public:
	virtual void persist(const void *, std::size_t, const char *) const = 0;
	virtual ~persister() {}
};

#endif
