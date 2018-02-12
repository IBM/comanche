#include <iostream>
#include <fstream>
#include <stdio.h>
#include <assert.h>

int main()
{
  FILE * fp = fopen("./fs/data","w+");
  if(fp==NULL) {
    perror("error:");
  }

  char buf[256];
  for(unsigned i=0;i<100;i++) {
    size_t rc = fread(buf, 256, 1, fp);
    assert(rc == 256);
  }
  
  fclose(fp);
  return 0;
}
