#include "metadata.h"

Metadata::Metadata(Component::IBlock_device * block_device, bool force_init) : _block(block_device)
{
  assert(block_device);

  _block->add_ref();
  _block->get_volume_info(_vi);

  _n_lba = _vi.max_lba + 1;
  size_t buffer_size = (_n_lba) * _vi.block_size;

  assert(buffer_size < GB(2)); // sanity check
  
  _iob = _block->allocate_io_buffer(buffer_size, _vi.block_size, Component::NUMA_NODE_ANY);
  _n_records = buffer_size / sizeof(__md_record);

  if(option_DEBUG)
    PLOG("Metadata: #records = %lu", _n_records);

  _records = static_cast<__md_record*>(_block->virt_addr(_iob));
  assert(check_aligned(_records, _vi.block_size));

  /* read data from store */
  read_data(0,_n_lba);

  if(force_init || !check_validity()) {
    initialize_space();
  }
  else {
    if(option_DEBUG) PLOG("using existing metadata space");
  }
  
  // if(!check_aligned(buffer, page_size))
  //   throw Constructor_exception("Metadata: buffer must be page aligned");

  // if(buffer_size % page_size)
  //   throw Constructor_exception("Metadata: buffer_size must be page modulo");
}

void Metadata::read_data(size_t start_lba, size_t count)
{
  /* synchronous read */
  _block->read(_iob,
               0,
               start_lba,
               count,
               0); /* queue id */  
}

void Metadata::write_data(size_t start_lba, size_t count)
{
  /* synchronous write */
  _block->write(_iob,
                0,
                start_lba,
                count,
                0);
}

Metadata::~Metadata()
{
  _block->free_io_buffer(_iob);
  _block->release_ref();
}

bool Metadata::check_validity()
{
  for(unsigned i=0;i<_n_records;i++) {
    if(_records[i].magic != MD_MAGIC) return false;
    /* to do, verify checksum */
  }
  return true;
}   

void Metadata::initialize_space()
{
  if(option_DEBUG)
    PLOG("Metadata: initializing space n_records:%lu", _n_records);
  
  parallel_for<int>(0,_n_records,
                    [=](int i)
                    {
                      memset(&_records[i], 0, sizeof(__md_record));
                      _records[i].magic = MD_MAGIC;    
                    });

  write_data(0,_n_lba);
}
