#include <gtest/gtest.h>

#include "../statistics.h"

// test fixture: Running Statistics
TEST(RunningStatisticsTest, Init)
{
    RunningStatistics bin;
    
    int count = bin.getCount();
    double min = bin.getMin();
    double max = bin.getMax();
    double mean = bin.getMean();
    double std = bin.getStd();

    ASSERT_EQ(count, 0);
    ASSERT_EQ(min, 0.0);
    ASSERT_EQ(max, 0.0);
    ASSERT_EQ(mean, 0.0);
    ASSERT_EQ(std, 0.0);
}

TEST(RunningStatisticsTest, SingleValue)
{
    RunningStatistics bin;

    double value = 2.0;
    bin.add_value(value);
    
    int count = bin.getCount();
    double min = bin.getMin();
    double max = bin.getMax();
    double mean = bin.getMean();
    double std = bin.getStd();

    ASSERT_EQ(count, 1);
    ASSERT_EQ(min, value);
    ASSERT_EQ(max, value);
    ASSERT_EQ(mean, value);
    ASSERT_EQ(std, 0.0);
}

TEST(RunningStatisticsTest, TwoValues)
{
    RunningStatistics bin;

    bin.add_value(2.0);
    bin.add_value(6.0);
    
    int count = bin.getCount();
    double min = bin.getMin();
    double max = bin.getMax();
    double mean = bin.getMean();
    double std = bin.getStd();

    ASSERT_EQ(count, 2);
    ASSERT_EQ(min, 2.0);
    ASSERT_EQ(max, 6.0);
    ASSERT_EQ(mean, 4.0);
    ASSERT_NEAR(std, 2.82843, 0.0001);
}

TEST(RunningStatisticsTest, TenValues)
{
    RunningStatistics bin;

    for (int i = 1; i < 11; i++)
    {
        bin.add_value((double)i);
    }
    
    int count = bin.getCount();
    double min = bin.getMin();
    double max = bin.getMax();
    double mean = bin.getMean();
    double std = bin.getStd();
    double variance = bin.getVariance();

    ASSERT_EQ(count, 10);
    ASSERT_EQ(min, 1.0);
    ASSERT_EQ(max, 10.0);
    ASSERT_FLOAT_EQ(mean, 5.5);
    ASSERT_NEAR(std, 3.02765, 0.0001);
}

TEST(BinStatisticsTests, default_init)
{
    BinStatistics stats;

    int count = stats.getBin(0).getCount();

    ASSERT_EQ(count, 0);
}

TEST(BinStatisticsTests, default_single)
{
    BinStatistics stats;

    double value = 1.0;

    stats.update(value);

    int count = stats.getBin(0).getCount();
    double mean = stats.getBin(0).getMean();

    ASSERT_EQ(count, 1);
    ASSERT_NEAR(mean, value, 0.0001);
}

TEST(BinStatisticsTests, default_two_large)
{
    BinStatistics stats;

    double value_positive = 9999999999;
    double value_negative = -value_positive;

    stats.update(value_positive);
    stats.update(value_negative);

    int count = stats.getBin(0).getCount();
    double max = stats.getBin(0).getMax();
    double min = stats.getBin(0).getMin();
    double mean = stats.getBin(0).getMean();

    ASSERT_EQ(count, 2);
    ASSERT_EQ(max, value_positive);
    ASSERT_EQ(min, value_negative);
    ASSERT_NEAR(mean, 0.0, 0.0001);
}

TEST(BinStatisticsTests, two_bins_one_value)
{
    BinStatistics stats(2, 0, 10);

    double value = 4.0;  // should go into bin 0

    stats.update(value);

    // check bin 0
    int count_0 = stats.getBin(0).getCount();
    double max_0 = stats.getBin(0).getMax();
    double min_0 = stats.getBin(0).getMin();
    double mean_0 = stats.getBin(0).getMean();
    double std_0 = stats.getBin(0).getStd();

    ASSERT_EQ(count_0, 1);
    ASSERT_EQ(max_0, value);
    ASSERT_EQ(min_0, value);
    ASSERT_NEAR(mean_0, value, 0.0001);
    ASSERT_NEAR(std_0, 0.0, 0.0001);

    // check bin 1
    int count_1 = stats.getBin(1).getCount();
    double max_1 = stats.getBin(1).getMax();
    double min_1 = stats.getBin(1).getMin();
    double mean_1 = stats.getBin(1).getMean();
    double std_1 = stats.getBin(1).getStd();

    ASSERT_EQ(count_1, 0);
    ASSERT_EQ(max_1, 0.0);
    ASSERT_EQ(min_1, 0.0);
    ASSERT_NEAR(mean_1, 0.0, 0.0001);
    ASSERT_NEAR(std_1, 0.0, 0.0001);
}

TEST(BinStatisticsTests, ten_bins_two_values)
{
    BinStatistics stats(10, 1, 10);

    double value_1 = 4.0;  // should go into bin 4
    double value_2 = 20.0;  // should go into bin 9

    stats.update(value_1);
    stats.update(value_2);

    // check bin 3
    int count_3 = stats.getBin(3).getCount();
    double max_3 = stats.getBin(3).getMax();
    double min_3 = stats.getBin(3).getMin();
    double mean_3 = stats.getBin(3).getMean();
    double std_3 = stats.getBin(3).getStd();

    ASSERT_EQ(count_3, 1);
    ASSERT_EQ(max_3, value_1);
    ASSERT_EQ(min_3, value_1);
    ASSERT_NEAR(mean_3, value_1, 0.0001);
    ASSERT_NEAR(std_3, 0.0, 0.0001);

    // check bin 9
    int count_9 = stats.getBin(9).getCount();
    double max_9 = stats.getBin(9).getMax();
    double min_9 = stats.getBin(9).getMin();
    double mean_9 = stats.getBin(9).getMean();
    double std_9 = stats.getBin(9).getStd();

    ASSERT_EQ(count_9, 1);
    ASSERT_EQ(max_9, value_2);
    ASSERT_EQ(min_9, value_2);
    ASSERT_NEAR(mean_9, value_2, 0.0001);
    ASSERT_NEAR(std_9, 0.0, 0.0001);

    // check remaining bins
    int count;
    double max, min, mean, std;

    for (int i = 0; i < 10; i++)
    {
        if (i == 3 || i == 9)
        {
            continue;  // these bins are populated
        }

        count = stats.getBin(i).getCount();
        max = stats.getBin(i).getMax();
        min = stats.getBin(i).getMin();
        mean = stats.getBin(i).getMean();
        std = stats.getBin(i).getStd();

        ASSERT_EQ(count, 0);
        ASSERT_EQ(max, 0.0);
        ASSERT_EQ(min, 0.0);
        ASSERT_NEAR(mean, 0.0, 0.0001);
        ASSERT_NEAR(std, 0.0, 0.0001);
    }
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
