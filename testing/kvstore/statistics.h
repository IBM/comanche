#ifndef __STATISTICS_H__
#define __STATISTICS_H__

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#include <common/logging.h>
#pragma GCC diagnostic pop

#include <algorithm>
#include <cmath>
#include <iosfwd>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <limits>
#include <vector>

class RunningStatistics
{
public:
    RunningStatistics()
      : count(0)
      , min(0)
      , max(0)
      , mean(0)
      , variance(0)
      , mean_last(0)
      , variance_last(0)
    {
    }

    void add_value(double value)
    {
        update_count();
        update_min(value);
        update_max(value);
        update_mean(value);
        update_variance(value);
    }

    unsigned getCount() const { return count; }
    double getMin() const { return min; }
    double getMax() const { return max; }
    double getMean() const { return mean; }
    double getVariance() const { return variance; }
    double getStd() const { return (count > 1 ? std::sqrt( variance / ( count - 1 ) ) : 0.0); }

private:
    void update_count()
    {
        count += 1;
    }

    void update_min(double value)
    {
        if (count == 1) // first time we've seen a value
        {
            min = value;
        }
        else
        {
            if (value < min)
            {
                min = value;
            }
        }
    }

    void update_max(double value)
    {
        if (count == 1) // first time we've seen a value
        {
            max = value;
        }
        else
        {
            if (value > max)
            {
                max = value;
            }
        }
    }

    void update_mean(double value)
    {
        if (count == 1)
        {
            mean = value;
            mean_last = value;
        }
        else
        {
            mean_last = mean;
            mean = mean_last + ((value - mean_last)/count);
        }
    }

    void update_variance(double value)
    {
        if (count == 1)
        {
            variance = 0.0;
            variance_last = 0.0;
        }
        else
        {
            variance_last = variance;
            variance = (variance_last + (value - mean_last)*(value - mean));
        }
    }

    unsigned count;
    double min;
    double max;
    double mean;
    double variance;

    double mean_last;
    double variance_last;
};

class BinStatistics
{
public:
    BinStatistics()
      : BinStatistics(1, std::numeric_limits<double>::min(), std::numeric_limits<double>::max())
    {
    }

    BinStatistics(unsigned bins, double threshold_min, double threshold_max)
      : _bin_count(bins)
      , _increment((threshold_max - threshold_min) / std::max(1U, bins))
      , _min(threshold_min)
      , _max(threshold_max)
      , _bins(bins)
    {
      if (bins == 0)
      {
        std::string e = "BinStatistics: bin count is 0, should be at least 1";
        PERR("%s.", e.c_str());
        throw std::runtime_error(e);
      }

      if (threshold_min > threshold_max)
      {
        std::ostringstream e;
        e << "BinStatistics: threshold_min " << threshold_min << " exceeds threshold_max " << threshold_max;
        PERR("%s.", e.str().c_str());
        throw std::runtime_error(e.str());
      }
    }

  public:

    void update(double value)
    {
        int bin = find_bin_from_value(value);

        _bins[bin].add_value(value);
    }

    void update_value_for_bin(double value, double bin_calc_value)
    {
        int bin = find_bin_from_value(bin_calc_value);

        _bins[bin].add_value(value);
    }

    double getIncrement() const { return _increment; }
    double getMinThreshold() const { return _min; }
    double getMaxThreshold() const { return _max; }
    unsigned getBinCount() const { return _bin_count; }

    const RunningStatistics &getBin(unsigned bin) const
    {
        if ( _bin_count <= bin )
        {
            throw std::domain_error("bin " + std::to_string(bin) + "not in range [0.." + std::to_string(_bin_count) + ")" );
        }

        return _bins[bin];
    }

    std::ostream &print_highest_count_bin(std::ostream &, unsigned core) const;

private:
    unsigned _bin_count;
    double _increment;
    double _min;
    double _max;
    std::vector<RunningStatistics> _bins;

    unsigned find_bin_from_value(double value) const
    {
        int bin = int((value - _min) / _increment);

        // clamp value to desired range
        if ( int(_bin_count) <= bin )
        {
            bin = int(_bin_count) - 1;
        }
        else if(bin < 0)
        {
            bin = 0;
        }

        return unsigned(bin);
    }
};

#endif // __STATISTICS_H__
