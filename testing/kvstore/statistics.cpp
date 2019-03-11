#include "statistics.h"

#include <iosfwd> /* ostream */

std::ostream &BinStatistics::print_highest_count_bin(std::ostream &o_, unsigned core) const
{
  unsigned count_highest = std::numeric_limits<unsigned>::max();  // arbitrary placeholder value
  unsigned count_highest_index = std::numeric_limits<unsigned>::max();  // arbitrary placeholder value

  // find bin with highest count
  for ( unsigned i = 0; i < getBinCount(); ++i )
  {
    if ( getBin(i).getCount() > count_highest )
    {
      count_highest = getBin(i).getCount();
      count_highest_index = i;
    }
  }

  if ( count_highest != std::numeric_limits<unsigned>::max() )
  {
    const RunningStatistics bin = getBin(count_highest_index);

    // print information about that bin
    o_
      << "SUMMARY: core " << core << std::endl
      << "\tmean:\t" << bin.getMean() << std::endl
      << "\tmin:\t" << bin.getMin() << std::endl
      << "\tmax:\t" << bin.getMax() << std::endl
      << "\tstd:\t" << bin.getMax() << std::endl
      << "\tcount:\t" << bin.getCount() << std::endl;
  }
  return o_;
}
