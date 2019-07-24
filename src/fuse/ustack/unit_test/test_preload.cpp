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
static constexpr unsigned ITERATIONS = 1000;

int                main(int argc, char * argv[])
{

  std::chrono::high_resolution_clock::time_point _start_time, _end_time;

  if(argc != 3){
    PERR("command format: (LD_PRELOAD=pathto-libfuseclient.so)./test_preload mount_path iosize(KB)")
    return -1;
  }
  file_size= KB(atoi(argv[2]));
  int open_flags = O_WRONLY | O_CREAT | O_DIRECT |O_SYNC;
  // int open_flags = O_WRONLY | O_CREAT  |O_SYNC;

  int use_preload = (getenv("LD_PRELOAD"))?1:0;
  std::string dir_name(argv[1]);
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

  for(unsigned i = 0, offset=0; i < nr_buffer_copies; i+= 1){
    memset((char *)buffer+offset, 'a'+i, file_size);
    offset += file_size;
  }

#if 0
  int fd = open("foobar.dat", O_SYNC | O_CREAT | O_TRUNC, O_WRONLY);
  assert(fd != -1);

#endif
  _start_time = std::chrono::high_resolution_clock::now();
  for (unsigned i = 0; i < ITERATIONS; i++) {
    char* ptr = (char*)buffer  + file_size*(i % nr_buffer_copies);
    ssize_t res = write(fd, ptr, file_size);
    if(res > file_size || res < 0){
      PWRN("write return %lu in iteration %u, ptr = %p", res, i, ptr);
    }
    fsync(fd);
    lseek(fd, 0, SEEK_SET);
  }
  _end_time = std::chrono::high_resolution_clock::now();

  double secs = std::chrono::duration_cast<std::chrono::milliseconds>(
                      _end_time - _start_time)
                      .count() /
                  1000.0;
  double iops = ((double) ITERATIONS) / secs;
  double throughput_in_mb = iops*file_size/(MB(1));

  PMAJOR("iops(%.1f), throughput(%.3f MB/s), iterations(%u)", iops, throughput_in_mb, ITERATIONS);

  close(fd);
  free(buffer);
  PLOG("done!");
  return 0;
}
