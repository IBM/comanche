#ifndef __STATISTICS_H__
#define __STATISTICS_H__

#include <cmath>
#include <iostream>
#include <vector>

class RunningStatistics
{
public:
    RunningStatistics()
    {

    }

    void add_value(double value)
    {
        std::cout << "RunningStatistics.add_value called" << std::endl;
        update_count();
        update_min(value);
        update_max(value);
        update_mean(value);
        update_variance(value);
    }

    double getCount() { return count; }
    double getMin() { return min; }
    double getMax() { return max; }
    double getMean() { return mean; }
    double getVariance() { return variance; }
    double getStd() { return sqrt(variance); }

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
            mean = mean = value;
            mean_last = 0.0;
        }
        else
        {
            mean = mean_last + (value - mean_last)/count; 
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
            variance = variance_last + (value - mean_last)*(value - mean);

            variance_last = variance;
        }
    }

    unsigned int count = 0;
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
    int _bin_count;
    double _increment;
    double _min;
    double _max;
    std::vector<RunningStatistics> _bins;

    BinStatistics()
    {
       // default
    }

    BinStatistics(int bins, double threshold_min, double threshold_max)
    {
        init(bins, threshold_min, threshold_max);
    } 

    void init(int bins, double threshold_min, double threshold_max)
    {
        _bin_count = bins;
        _min = threshold_min;
        _max = threshold_max;

        _increment = (threshold_max - threshold_min) / bins;
        _bins.resize(bins);

        std::cout << "BinStatistics.init: min = " << _min << ", max = " << _max << ". Bins = " << _bin_count << std::endl;
    }

    void update(double value)
    {
        std::cout << "BinnedStatistics.update called" << std::endl;
        int bin = get_latency_bin(value);

        std::cout << "\tvalue " << value << " got bin " << bin << std::endl;

        _bins[bin].add_value(value);
    }

    int get_latency_bin(double value)
    {
        int bin = (value - _min) / _increment;

        // clamp value to desired range
        if (bin >= (_bin_count - 1))
        {
            bin = _bin_count - 1;
        }

        return bin;
    }
};

#endif // __STATISTICS_H__
