%module Block
%{
#define SWIG_FILE_WITH_INIT
#include <memory>
#include "block_i.h"
%}
%include "std_string.i"

typedef int      status_t;
typedef uint64_t workid_t;
typedef uint64_t addr_t;
typedef void (*io_callback_t)(uint64_t, void*, void*);

class Comanche_exception;

class IO_buffer
{
public:
  IO_buffer() {}
  IO_buffer(Component::IBlock_device * owner, uint64_t iob);
  ~IO_buffer();
  void show() const;
};

class Block
{
public:
  Block(const char * config, unsigned long cpu_mask, const char * lib_name = nullptr) throw(Comanche_exception);
  IO_buffer* allocate_io_buffer(size_t size, unsigned alignment, int numa_node);
  std::string get_volume_info();
};



