#include <stdio.h>
#include <common/logging.h>

int CheckForIntelShaExtensions() {
  int a, b, c, d;

  // Look for CPUID.7.0.EBX[29]
  // EAX = 7, ECX = 0
  a = 7;
  c = 0;

  asm volatile ("cpuid"
                :"=a"(a), "=b"(b), "=c"(c), "=d"(d)
                :"a"(a), "c"(c)
                );

  // Intel® SHA Extensions feature bit is EBX[29]
  return ((b >> 29) & 1);
}

int main()
{
  PLOG("Process SHA extension:%d", CheckForIntelShaExtensions());
  return 0;
}
