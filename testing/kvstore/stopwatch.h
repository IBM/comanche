#ifndef __STOPWATCH_H__
#define __STOPWATCH_H__

#include <chrono> 

class Stopwatch
{
public:
    bool is_running()
    {
        return running;
    }

    void start()
    {
        if (!running)
        {
            start_time = std::chrono::high_resolution_clock::now();
            running = true;
        }
    }

    void stop()
    {
        if (running)
        {
            std::chrono::system_clock::time_point stop_time = std::chrono::high_resolution_clock::now();
            running = false;

            total += (stop_time - start_time);
        }
    }

    double get_time_in_seconds()
    {
        std::chrono::duration<double, std::milli> running_time = std::chrono::duration<double>(std::chrono::duration_values<double>::zero());

        if (running)
        {
            // take into account the time that's running but hasn't stopped, like glancing at the clock
            std::chrono::system_clock::time_point stop_time = std::chrono::high_resolution_clock::now();

            running_time = (stop_time - start_time);
        }

        double seconds = std::chrono::duration_cast<std::chrono::milliseconds>(total + running_time).count() / 1000.0;

        return seconds;
    }

private:
    std::chrono::duration<double, std::milli> total = std::chrono::duration<double>(std::chrono::duration_values<double>::zero());
    bool running = false;
    std::chrono::system_clock::time_point start_time;
};


#endif //  __STOPWATCH_H__
