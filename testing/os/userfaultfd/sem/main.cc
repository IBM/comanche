/* test raw semaphore performance */

#include <stdio.h>
#include <unistd.h>
#include <semaphore.h>
#include <sys/wait.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <assert.h>
#include <fcntl.h>
#include <stdlib.h>

#include <common/cpu.h>

#define ITERATIONS 100000000

unsigned prod_core = 0;
unsigned cons_core = 0;

int main(int argc, char * argv[])
{

  if(argc != 3) {
    printf("raw-sem-perf <producer core> <consumer core>\n");
    return -1;
  }

  prod_core = atoi(argv[1]);
  cons_core = atoi(argv[2]);

  //  ftruncate(fd, sizeof(sem_t));

  sem_t *sem = sem_open("sem", O_CREAT, 0644, 3); /* Initial value is 3. */
  assert(sem);
    
  sem_init(sem, 1, 0);
    
  if(fork()==0) {

    set_cpu_affinity(1UL << prod_core);

    printf("forked producer\n");

    for(unsigned long i=0;i<ITERATIONS;i++) {
      while(sem_post(sem) != 0);
    }
    printf("producer complete\n");
    return 0;
  }
  else {

    set_cpu_affinity(1UL << cons_core);

    printf("forked consumer\n");
    sem_t * sem = NULL;
    do {
      sem = sem_open("sem", 0); /* Open a preexisting semaphore. */
    }
    while(sem==NULL);
    assert(sem);

    struct timeval tv_start, tv_end;
    gettimeofday(&tv_start, NULL);

    for(unsigned long i=0;i<ITERATIONS;i++) {
      while(sem_wait(sem)!=0);
      //      printf("consume!\n");
    }

    gettimeofday(&tv_end, NULL);
    double seconds = (tv_end.tv_sec - tv_start.tv_sec) + ((double)(tv_end.tv_usec - tv_start.tv_usec))/1000000.0;
    printf("consumer complete %g seconds\n", seconds);
    double rate = (((double)ITERATIONS)/seconds)/1000000.0;
    printf("rate: %g million items/sec\n",rate);
    printf("rate: %g usec per iteration\n", (seconds * 1000000.0)/ITERATIONS);
  }
  int status;
  wait(&status);


  return 0;
}
