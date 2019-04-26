#include <gtest/gtest.h>

#include "../stopwatch.h"

#define TIME_TOLERANCE 0.01

// test fixture: Stopwatch 
TEST(StopwatchTest, Init)
{
  Stopwatch timer;

  double time = timer.get_time_in_seconds();
  ASSERT_EQ(time, 0.0); 
}

TEST(StopwatchTest, Basic_1_sec)
{
  double duration = 1.0;
  Stopwatch timer;

  timer.start();
  sleep(duration);
  timer.stop();

  double time = timer.get_time_in_seconds();

  ASSERT_NEAR(time, duration, TIME_TOLERANCE);
}

TEST(StopwatchTest, Basic_2_sec)
{
  double duration = 2.0;
  Stopwatch timer;

  timer.start();
  sleep(duration);
  timer.stop();

  double time = timer.get_time_in_seconds();

  ASSERT_NEAR(time, duration, TIME_TOLERANCE);
}


TEST(StopwatchTest, Lap_1_sec)
{
  double duration = 0.5;  // seconds
  double delay = 0.2; // arbitrary non-zero, non-duration value
  int laps = 2;
  double total_time = duration * laps;
  double current_lap, current_total;
  Stopwatch timer;

  for (int i = 0; i < laps; i++)
  {
    timer.start();
    usleep(duration * 1000000); // usleep is in microseconds
    timer.stop();

    current_total = timer.get_time_in_seconds();
    current_lap = timer.get_lap_time_in_seconds();

    ASSERT_NEAR(current_total, duration * (i + 1), TIME_TOLERANCE);
    ASSERT_NEAR(current_lap, duration, TIME_TOLERANCE);

    usleep(delay * 1000000);
  }
}

TEST(StopwatchTest, Lap_2_sec)
{
  double duration = 0.5;  // seconds
  double delay = 0.2; // arbitrary non-zero, non-duration value
  int laps = 4;
  double total_time = duration * laps;
  double current_lap, current_total;
  Stopwatch timer;

  for (int i = 0; i < laps; i++)
  {
    timer.start();
    usleep(duration * 1000000); // usleep is in microseconds
    timer.stop();

    current_total = timer.get_time_in_seconds();
    current_lap = timer.get_lap_time_in_seconds();

    ASSERT_NEAR(current_total, duration * (i + 1), TIME_TOLERANCE);
    ASSERT_NEAR(current_lap, duration, TIME_TOLERANCE);

    usleep(delay * 1000000);
  }
}

TEST(StopwatchTest, Lap_10_sec)
{
  double duration = 0.25;  // seconds
  double delay = 0.2; // arbitrary non-zero, non-duration value
  int laps = 40;
  double total_time = duration * laps;
  double current_lap, current_total;
  Stopwatch timer;

  for (int i = 0; i < laps; i++)
  {
    timer.start();
    usleep(duration * 1000000); // usleep is in microseconds
    timer.stop();

    current_total = timer.get_time_in_seconds();
    current_lap = timer.get_lap_time_in_seconds();

    ASSERT_NEAR(current_total, duration * (i + 1), TIME_TOLERANCE);
    ASSERT_NEAR(current_lap, duration, TIME_TOLERANCE);

    usleep(delay * 1000000);
  }
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
