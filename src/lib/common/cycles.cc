#include <common/cycles.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

namespace Core
{

float get_rdtsc_frequency_mhz()
{
  FILE* fp;
  char buffer[1024];
  size_t bytes_read;
  char* match;
  float clock_speed = 0.0;
  float clock_speed_in_MHz = 0;
  /* Read the entire contents of /proc/cpuinfo into the buffer. */
  fp = fopen ("/proc/cpuinfo", "r");
  bytes_read = fread (buffer, 1, sizeof (buffer), fp);
  fclose(fp);

  assert(bytes_read > 0);
  buffer[bytes_read] = '\0';
 
  /* First try to get cpu freq from model name. */
  match = strstr (buffer, "model name");
  assert(match);
  while(*match != '@') match++;
  /* Parse the line to extract the clock speed. */
  sscanf (match, "@ %f", &clock_speed);

  if(clock_speed > 0)
    clock_speed_in_MHz = clock_speed * 1000.0;
  else{
    /* some qemu guest and the aep1 host don't have freq in model name */
    match = strstr (buffer, "cpu MHz");
    assert(match);
    while(*match != ':') match++;

    sscanf (match, ": %f", &clock_speed_in_MHz);
  }
  assert(clock_speed_in_MHz > 0);

  return clock_speed_in_MHz;
}

} // namespace Core
