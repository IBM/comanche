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
