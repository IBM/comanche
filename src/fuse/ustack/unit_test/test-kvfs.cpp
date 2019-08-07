#include <gtest/gtest.h>
#include <common/utils.h>
#include <api/components.h>
#include <api/kvstore_itf.h>
#include <chrono> /* milliseconds */
#include <common/utils.h>
struct {
  std::string dir_name;

  size_t file_size = MB(4);
  size_t nr_buffer_copies = 3;
  int open_flags = O_RDWR | O_CREAT |O_SYNC | O_DIRECT;
  bool use_preload=false;
} opt;

namespace{
// The fixture for testing class KVFS.
class KVFS_test : public ::testing::Test {

 protected:

  // If the constructor and destructor are not enough for setting up
  // and cleaning up each test, you can define the following methods:
  
  virtual void SetUp() {
    // Code here will be called immediately after the constructor (right
    // before each test).
  }

  virtual void TearDown() {
    // Code here will be called immediately after each test (right
    // before the destructor).
  }
  
  // Objects declared here can be used by all tests in the test case
};


TEST_F(KVFS_test, DISABLED_warmup){

  void* buffer;

  for(uint warmup_size  = KB(4);warmup_size < MB(32); warmup_size *= 4){
    PLOG("Warmup with size %u)", warmup_size);
    std::string filepath = opt.dir_name + "/warmupfile-size" + std::to_string(warmup_size)+  ".dat";

    int fd = open(filepath.c_str(), opt.open_flags, S_IRWXU);
    EXPECT_GT(fd, 0);

    EXPECT_EQ(0, posix_memalign(&buffer, 4096, warmup_size));

    ssize_t res = pwrite(fd, buffer, warmup_size,0);
    EXPECT_LE(res, warmup_size);
    EXPECT_GT(res, 0);

    close(fd);
    free(buffer);
  }
}

/** Write each mb in the file and verify*/
TEST_F(KVFS_test, PartialFileWrite){
  void * buffer;
  size_t file_size = MB(32);
  bool is_read = false;

  size_t slab_size = MB(1);
  size_t nr_slabs = file_size/slab_size;

  // buffer = malloc(file_size*nr_buffer_copies);
  EXPECT_EQ(0, posix_memalign(&buffer, 4096, slab_size));

  // prepare file
  std::string method_str = "-preload"+ std::to_string(opt.use_preload) + "-sz" + std::to_string(file_size) + "-f" + std::to_string(opt.open_flags);

  std::string filepath = opt.dir_name + "/kvfs-paritialfile-" + method_str +".dat";

  PMAJOR("Using file (%s), size %lu)", filepath.c_str(), file_size);

  int fd = open(filepath.c_str(), opt.open_flags, S_IRWXU);
  EXPECT_GT(fd, 0);

  /* write to each slab*/
  off_t file_off = 0;
  for(unsigned i = 0; i < nr_slabs; i+= 1){
    memset((char *)buffer, 'a'+i, slab_size);
    EXPECT_EQ(slab_size, pwrite(fd, buffer, slab_size, file_off));
    file_off += slab_size;
  }
  close(fd);
  // TODO: somehow it's not totally flushed here

  sleep(2);

  // reopen for read
  fd = open(filepath.c_str(), opt.open_flags, S_IRWXU);
  EXPECT_GT(fd, 0);

  //very file each slab
  file_off = 0;
  for(unsigned i = 0; i < nr_slabs; i+= 1){
    EXPECT_EQ(slab_size, pread(fd, buffer, slab_size, file_off));
    file_off += slab_size;
    EXPECT_EQ(((char *)buffer)[0], 'a' + i);
  }
  close(fd);


  free(buffer);
  PLOG("done!");
}

#if 0
TEST_F(KVFS_test, WholeFileWrite){
  void * buffer;
  size_t file_size = opt.file_size;
  size_t nr_buffer_copies = opt.nr_buffer_copies;
  size_t ITERATIONS = 10000;
  bool is_read = false;

  std::chrono::high_resolution_clock::time_point _start_time, _end_time;

  // buffer = malloc(file_size*nr_buffer_copies);
  EXPECT_EQ(0, posix_memalign(&buffer, 4096, file_size*nr_buffer_copies));
  /* Operate on different buffer each time*/
  for(unsigned i = 0, offset=0; i < nr_buffer_copies; i+= 1){
    memset((char *)buffer+offset, 'a'+i, file_size);
    offset += file_size;
  }

  std::string method_str = "-preload"+ std::to_string(opt.use_preload) + "-sz" + std::to_string(file_size) + "-f" + std::to_string(opt.open_flags) + "-iter"+ std::to_string(ITERATIONS);

  std::string filepath = opt.dir_name + "/kvfs-wholefile-" + method_str +".dat";

  PMAJOR("Using file (%s), size %lu)", filepath.c_str(), file_size);

  int fd = open(filepath.c_str(), opt.open_flags, S_IRWXU);
  EXPECT_GT(fd, 0);

  _start_time = std::chrono::high_resolution_clock::now();
  for (unsigned i = 0; i < ITERATIONS; i++) {
    char* ptr = (char*)buffer  + file_size*(i % nr_buffer_copies);
    ssize_t res;
      res = pwrite(fd, ptr, file_size, 0);
      fsync(fd);
    if(res > file_size || res < 0){
      PWRN("write/read return %lu in iteration %u, ptr = %p", res, i, ptr);
    }
  }
  _end_time = std::chrono::high_resolution_clock::now();

  double secs = std::chrono::duration_cast<std::chrono::milliseconds>(
                      _end_time - _start_time)
                      .count() /
                  1000.0;
  double iops = ((double) ITERATIONS) / secs;
  double throughput_in_mb = iops*file_size/(MB(1));

  PMAJOR("[WHOLEFILE-RESULT]: %s, iops(%.1f), throughput(%.3f MB/s), iterations(%lu)", method_str.c_str(), iops, throughput_in_mb, ITERATIONS);

  close(fd);
  free(buffer);
  PLOG("done!");
}

TEST_F(KVFS_test, WholeFileRead){
  void * buffer;
  size_t file_size = opt.file_size;
  size_t nr_buffer_copies = opt.nr_buffer_copies;
  size_t ITERATIONS = 10000;

  std::chrono::high_resolution_clock::time_point _start_time, _end_time;

  // buffer = malloc(file_size*nr_buffer_copies);
  EXPECT_EQ(0, posix_memalign(&buffer, 4096, file_size*nr_buffer_copies));
  /* Operate on different buffer each time*/
  for(unsigned i = 0, offset=0; i < nr_buffer_copies; i+= 1){
    memset((char *)buffer+offset, 'a'+i, file_size);
    offset += file_size;
  }

  std::string method_str = "-preload"+ std::to_string(opt.use_preload) + "-sz" + std::to_string(file_size) + "-f" + std::to_string(opt.open_flags) + "-iter"+ std::to_string(ITERATIONS);

  std::string filepath = opt.dir_name + "/kvfs-wholefile-" + method_str +".dat";

  PMAJOR("Using file (%s), size %lu)", filepath.c_str(), file_size);

  int fd = open(filepath.c_str(), opt.open_flags, S_IRWXU);
  EXPECT_GT(fd, 0);

  {// Prepare file
    char* ptr = (char*)buffer;
    ssize_t res = pwrite(fd, ptr, file_size, 0);
    fsync(fd);
    if(res > file_size || res < 0){
      PWRN("write return %lu , ptr = %p", res, ptr);
    }
  }

  _start_time = std::chrono::high_resolution_clock::now();
  for (unsigned i = 0; i < ITERATIONS; i++) {
    char* ptr = (char*)buffer  + file_size*(i % nr_buffer_copies);
    ssize_t res;
      res = pread(fd, ptr, file_size, 0);
    if(res > file_size || res < 0){
      PWRN("write/read return %lu in iteration %u, ptr = %p", res, i, ptr);
    }
  }
  _end_time = std::chrono::high_resolution_clock::now();

  double secs = std::chrono::duration_cast<std::chrono::milliseconds>(
                      _end_time - _start_time)
                      .count() /
                  1000.0;
  double iops = ((double) ITERATIONS) / secs;
  double throughput_in_mb = iops*file_size/(MB(1));

  PMAJOR("[WHOLEFILE-RESULT]: %s, iops(%.1f), throughput(%.3f MB/s), iterations(%lu)", method_str.c_str(), iops, throughput_in_mb, ITERATIONS);

  close(fd);
  free(buffer);
  PLOG("done!");
}
#endif




}

int main(int argc, char **argv) {
  if (argc != 2) {
    PINF("test-kvfs mountdir");
    return 0;
  }

  opt.dir_name = argv[1];

  opt.use_preload = (getenv("LD_PRELOAD"))?1:0;
  
  ::testing::InitGoogleTest(&argc, argv);
  auto r = RUN_ALL_TESTS();

  return r;
}


