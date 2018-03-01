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
  /* Read the entire contents of /proc/cpuinfo into the buffer. */
  fp = fopen ("/proc/cpuinfo", "r");
  bytes_read = fread (buffer, 1, sizeof (buffer), fp);
  fclose(fp);

  assert(bytes_read > 0);
  buffer[bytes_read] = '\0';
 
  /* Locate the line that starts with "cpu MHz". */
  match = strstr (buffer, "model name");
  assert(match);
  while(*match != '@') match++;
  /* Parse the line to extract the clock speed. */
  sscanf (match, "@ %f", &clock_speed);
  assert(clock_speed > 0);
  
  return clock_speed * 1000.0;
}

} // namespace Core
