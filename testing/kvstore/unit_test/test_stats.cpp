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

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
