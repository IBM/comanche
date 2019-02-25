#include "perishable.h"
#include "perishable_expiry.h"

#include <iostream>

bool perishable::_enabled = false;
std::uint64_t perishable::_initial = 0;
std::uint64_t perishable::_time_to_live = 0;

void perishable::tick()
{
	if ( _enabled )
	{
		if ( _time_to_live == 0 )
		{
			throw perishable_expiry{};
		}
		--_time_to_live;
	}
}

void perishable::reset(std::uint64_t n)
{
	_initial = n;
	_time_to_live = n;
}

void perishable::enable(bool e)
{
	_enabled = e;
}

void perishable::test()
{
	if ( _enabled )
	{
		if ( _time_to_live == 0 )
		{
			throw perishable_expiry{};
		}
	}

}

void perishable::report()
{
	if ( _initial != 0 )
	{
		std::cerr << "perishable: " << _time_to_live
			<< " of " << _initial << " ticks left\n";
	}
}
