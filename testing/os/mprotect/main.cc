#include <sys/mman.h>
#include <stdio.h>
#include <signal.h>
#include <errno.h>
#include <assert.h>
#include <set>
#include <common/utils.h>
#include <common/logging.h>
#include <common/cycles.h>

std::set<addr_t> gFaults;

static void
SIGSEGV_handler(int sig, siginfo_t *si, void *context)
{
  
  // unprotect memory region ; we'd have to know the region size in pages
  //
  void* fault_area = (void*) (round_down_page((addr_t)si->si_addr));
  PLOG("fault = %p round_down=%p",(void*)si->si_addr,fault_area);
  int rc = mprotect(fault_area, 4096, PROT_READ | PROT_WRITE);
  if(rc) {
    PERR("SIGSEGV_handler mprotect rc=%d errno=%d",rc,errno);
    exit(0);
  }

  gFaults.insert((addr_t) fault_area);
  // get calling RIP
  //
  //  long long bp = ((ucontext_t*)context)->uc_mcontext.gregs[REG_RIP];
  
  //  printf("Got SIGSEGV (from errno:%d ) accessing address: %p\n",si->si_errno, si->si_addr);

}


int main()
{
  struct sigaction sa;
  sa.sa_flags = SA_SIGINFO;
  sigemptyset(&sa.sa_mask);
  sa.sa_sigaction = SIGSEGV_handler;
  if(sigaction(SIGSEGV, &sa, NULL) == -1)
    assert(0);
  
  size_t data_size = 4096;
  void * data = aligned_alloc(4096*5,data_size);

  // populate
  uint64_t * p = (uint64_t *) data;
  for(unsigned i=0;i<data_size/sizeof(uint64_t);i++) {
    p[i] = rand();
  }

  int rc;

  rc = ::mprotect(data, data_size, PROT_NONE);
  if(rc) {
    PERR("mprotect rc=%d errno=%d",rc,errno);
    exit(0);
  }
  assert(rc == 0);

  cpu_time_t start,end;
  cpu_time_t start2,end2;
  cpu_time_t start3,end3;

  start = rdtsc();
  p[0] = 1;
  end = rdtsc();
  p[1] = 1;

  start2 = rdtsc();
  p[4096] = 1;
  end2 = rdtsc();

  start3 = rdtsc();
  p[4096*2] = 1;
  end3 = rdtsc();

  PLOG("first fault:%ld cycles", end - start);
  PLOG("second fault:%ld cycles", end2 - start2);
  PLOG("third fault:%ld cycles", end3 - start3);
  
  free(data);

  for(auto i : gFaults) {
    PLOG("Fault: %lx",i);
  }
  
  return 0;
}
