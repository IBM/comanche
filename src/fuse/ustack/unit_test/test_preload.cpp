#include <common/logging.h>
#include <fcntl.h>  // O_flags
#include <string.h>
#include <sys/types.h>  //open
#include <unistd.h>     // close
#include <cstdlib>
#include <string>
#include <chrono> /* milliseconds */
#include <common/utils.h>
/** Preload the ustack libray and those operations will be intecepted
 * 1. malloc free
 * 2. open/read/write
 * Without preload: ./src/fuse/ustack/unit_test/test-preload /tmp/mymount/
 * With preload:  LD_PRELOAD=./src/fuse/ustack/libustack_client.so ./src/fuse/ustack/unit_test/test-preload /tmp/mymount */

static size_t file_size=KB(4);
static constexpr unsigned nr_buffer_copies=3; // make sure each time using different buffer
static constexpr unsigned ITERATIONS = 10;

/* TODO: direct start 4k write will be slow*/
status_t do_warm_up(std::string dir_name, int open_flags){
  void* buffer;

  for(uint warmup_size  = KB(4);warmup_size < MB(2); warmup_size *= 4){
    PLOG("Warmup with size %u)", warmup_size);
    std::string filepath = dir_name + "/warmupfile-size" + std::to_string(warmup_size)+  ".dat";
    int fd = open(filepath.c_str(), open_flags, S_IRWXU);
    assert(fd>0);
    if(posix_memalign(&buffer, 4096, warmup_size)){PERR("aligned alloc files"); return -1;}
      ssize_t res = write(fd, buffer, warmup_size);
    if(res > warmup_size || res < 0){
      PWRN("write return %lu  ptr = %p", res,buffer);
    }
    close(fd);
    free(buffer);
  }
  PMAJOR("Warmup finished");
  return S_OK;
}

int                main(int argc, char * argv[])
{

  std::chrono::high_resolution_clock::time_point _start_time, _end_time;

  if(argc != 3){
    PERR("command format: (LD_PRELOAD=pathto-libfuseclient.so)./test_preload mount_path iosize(KB)")
    return -1;
  }
  file_size= KB(atoi(argv[2]));
  int use_O_DIRECT = 1;

  int open_flags = O_RDWR | O_CREAT |O_SYNC;
  if(use_O_DIRECT)
    open_flags |= O_DIRECT;

  int use_preload = (getenv("LD_PRELOAD"))?1:0;
  std::string dir_name(argv[1]);
  int is_read = 0;

  assert(S_OK == do_warm_up(dir_name, open_flags));

  std::string method_str = "preload"+ std::to_string(use_preload) + "-sz" + std::to_string(file_size) + "-f" + std::to_string(open_flags) + "-iter"+ std::to_string(ITERATIONS);

  std::string filepath = dir_name + "/foobar-" + method_str +".dat";

  PMAJOR("Using file (%s), size %lu)", filepath.c_str(), file_size);

  int fd = open(filepath.c_str(), open_flags, S_IRWXU);
  if (fd == -1) {
    PERR("Open failed, delete file (%s)first", filepath.c_str());
    return -1;
  }
  void * buffer;
  // buffer = malloc(file_size*nr_buffer_copies);
  if(posix_memalign(&buffer, 4096, file_size*nr_buffer_copies)){PERR("aligned alloc files"); return -1;}

  /* Operate on different buffer each time*/
  for(unsigned i = 0, offset=0; i < nr_buffer_copies; i+= 1){
    memset((char *)buffer+offset, 'a'+i, file_size);
    offset += file_size;
  }

  if(is_read){
    char* ptr = (char*)buffer;
    ssize_t res = write(fd, ptr, file_size);
    fsync(fd);
    if(res > file_size || res < 0){
      PWRN("write return %lu , ptr = %p", res, ptr);
    }
    lseek(fd, 0, SEEK_SET);
  }

  _start_time = std::chrono::high_resolution_clock::now();
  for (unsigned i = 0; i < ITERATIONS; i++) {
    char* ptr = (char*)buffer  + file_size*(i % nr_buffer_copies);
    ssize_t res;
    if(is_read)
      res = read(fd, ptr, file_size);
    else{
      res = write(fd, ptr, file_size);
      fsync(fd);
    }
    if(res > file_size || res < 0){
      PWRN("write/read return %lu in iteration %u, ptr = %p", res, i, ptr);
    }
    lseek(fd, 0, SEEK_SET);
  }
  _end_time = std::chrono::high_resolution_clock::now();

  double secs = std::chrono::duration_cast<std::chrono::milliseconds>(
                      _end_time - _start_time)
                      .count() /
                  1000.0;
  double iops = ((double) ITERATIONS) / secs;
  double throughput_in_mb = iops*file_size/(MB(1));

  PMAJOR("[WHOLEFILE-RESULT]: %s, iops(%.1f), throughput(%.3f MB/s), iterations(%u)", method_str.c_str(), iops, throughput_in_mb, ITERATIONS);

  close(fd);
  free(buffer);
  PLOG("done!");
  return 0;
}
