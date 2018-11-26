#ifndef __STATISTICS_H__
#define __STATISTICS_H__

#include <cmath>
#include <common/logging.h>
#include <iostream>
#include <limits>
#include <vector>

class RunningStatistics
{
public:
    RunningStatistics()
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

    int getCount() { return count; }
    double getMin() { return min; }
    double getMax() { return max; }
    double getMean() { return mean; }
    double getVariance() { return variance; }
    double getStd() { return (count > 1 ? sqrt( variance / ( count - 1 ) ) : 0.0); }

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

    int count = 0;
    double min = 0;  
    double max = 0;  
    double mean = 0;
    double variance = 0;

    double mean_last = 0;
    double variance_last = 0;
};

class BinStatistics
{
public:
    BinStatistics()
    {
       // default: everything goes into one bin with range at top and bottom of double range
       init(1, std::numeric_limits<double>::min(), std::numeric_limits<double>::max());
    }

    BinStatistics(int bins, double threshold_min, double threshold_max)
    {
        init(bins, threshold_min, threshold_max);
    } 

    void init(int bins, double threshold_min, double threshold_max)
    {
        if (bins < 0)
        {
            PERR("BinStatistics.init: bins can't be negative");
            throw std::exception();
        }
        else if (bins == 0)
        {
            PERR("BinStatistics.init: bin count should be at least 1");
            throw std::exception();
        }

        if (threshold_min > threshold_max)
        {
            PERR("BinStatistics.init: threshold_max should be larger than threshold_min. ");
            std::cerr << "min: " << threshold_min << ", max = " << threshold_max << std::endl;
            throw std::exception();
        }

        _bin_count = bins;
        _min = threshold_min;
        _max = threshold_max;

        _increment = (threshold_max - threshold_min) / bins;
        _bins.resize(bins);
    }


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

    double getIncrement() { return _increment; }
    double getMinThreshold() { return _min; }
    double getMaxThreshold() { return _max; }
    double getBinCount() { return _bin_count; }

    RunningStatistics getBin(int bin)
    {
        if (bin < 0)
        {
            throw std::exception();
        }
        else if (bin > _bin_count)
        {
            throw std::exception();
        }

        return _bins[bin];
    }

private:
    int _bin_count;
    double _increment;
    double _min;
    double _max;
    std::vector<RunningStatistics> _bins;

    int find_bin_from_value(double value)
    {
        int bin = (value - _min) / _increment;

        // clamp value to desired range
        if (bin >= (_bin_count - 1))
        {
            bin = _bin_count - 1;
        }
        else if(bin < 0)
        {
            bin = 0;
        }

        return bin;
    }
};

#endif // __STATISTICS_H__
