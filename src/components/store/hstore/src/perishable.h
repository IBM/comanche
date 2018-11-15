#ifndef _DAWN_HSTORE_PERISHABLE_H
#define _DAWN_HSTORE_PERISHABLE_H

#include <cstdint>

class perishable
{
	static bool _enabled;
	static std::uint64_t _initial;
	static std::uint64_t _time_to_live;
public:
	static void reset(std::uint64_t n);
	static void enable(bool e);
	static void tick();
	static void report();
};

#endif
