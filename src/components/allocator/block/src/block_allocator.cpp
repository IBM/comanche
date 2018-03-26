#include <assert.h>
#include <core/physical_memory.h>
#include "block_allocator.h"
#include "block_bitmap.h"
#include "segment.h"

#define RESET_STATE // testing only

using namespace Component;


/*

Allocation strategy is worst-fit-first; this is because we want to enable "copy-less"
expansion as much as possible.

This is NOT a bitmap. It is a list of ranges.

Segments are put in log 2 bins according to the maximum free contiguous space they have

[1] - [Lockfree-FIFO: {SEG5}-{SEG1}-{SEG2} ]
[2] - [Lockfree-FIFO: {SEG3}-{SEG4} ]
[4] - [Lockfree-FIFO: {SEG6}-{SEG8}-{SEG7} ]
...
[28] - [Lockfree-FIFO: {SEG203} ]

 */
static constexpr size_t PER_SEGMENT_FOOTPRINT    = 4096;
static constexpr size_t LBA_RANGE_PER_SEGMENT    = (GB(2)/4096);
//static constexpr size_t NUM_SEGMENTS = 1000;

static unsigned size_to_bin(uint64_t size)
{
  return 64 - __builtin_clzll(size);
}

static size_t bin_to_size(unsigned bin)
{
  return 1 << bin;
}

Block_allocator::
Block_allocator(IPersistent_memory * pmem,
                size_t max_lba,
                persist_id_t id,
                int numa_node,
                bool force_init)  : _pmem(pmem)
{
  assert(_pmem);
  assert(max_lba > 0);
  _pmem->add_ref();

  PLOG("Block_allocator: max_lba=%lu", max_lba);
  
  /* create bins (lfq for each bin) */
  unsigned num_bins = size_to_bin(max_lba) + 1;
  PLOG("Block_allocator: %u bins", num_bins);
  _last_bin_index = num_bins - 1;

  _num_segments = (max_lba / LBA_RANGE_PER_SEGMENT) + 1;
  PLOG("Block_allocator: %lu segments (rounded up log2 %lu)",
       _num_segments, round_up_log2(_num_segments));

  PLOG("Block_allocator: max allocations = %lu",
       Segment::slot_count(PER_SEGMENT_FOOTPRINT) * _num_segments);

  /* open persistent memory */
  bool reused;
  byte* p = nullptr;

  size_t memory_needed = _num_segments * PER_SEGMENT_FOOTPRINT;  
  PLOG("Block_allocator: metadata footprint = %ld KiB", REDUCE_KB(memory_needed));

  _pmem_pages = pmem->open(id, memory_needed, numa_node, reused, (void*&)p);
  PLOG("Block_allocator: persistent memory area @ %p reused = %s", p, reused ? "y" : "n");

  if(option_DEBUG) {
    if(!reused || 1) { // TESTING ONLY
      memset(p, 0, memory_needed);
      PLOG("Block_allocator: memset OK %ld bytes", memory_needed);
    }
  }

  /* create an FIFO per bin */
  for(unsigned b=0;b<=num_bins;b++) {
    _vofsl.push_back(new Common::Mpmc_bounded_lfq<Segment*>(round_up_log2(_num_segments),
                                                            &_stdalloc));
  }

  //  PLOG("Block_allocator: LBA per segment (max alloc) = %lu (%lu MB)",
  //       LBA_RANGE_PER_SEGMENT, REDUCE_MB(LBA_RANGE_PER_SEGMENT*4096));
  
  /* create segments */
  lba_t lba = 0;
  Segment * last = nullptr;
  _root_segment = nullptr;
  
  for(unsigned s=0;s<_num_segments;s++) {
#ifdef RESET_STATE
    reused = false; // testing
#endif
    
    Segment * new_page = new Segment(p,
                                     PER_SEGMENT_FOOTPRINT,
                                     lba,
                                     LBA_RANGE_PER_SEGMENT,
                                     !reused);
    //    new_page->dump_info();

    if(!_root_segment)
      _root_segment = new_page;
    
    lba += LBA_RANGE_PER_SEGMENT;
    p += PER_SEGMENT_FOOTPRINT;

    if(last) {
      last->set_adjacent(new_page);
      bool status = _vofsl[num_bins - 1]->enqueue(last);
      if(!status) throw Logic_exception("vofsl enqueue failed");
    }
    last = new_page;    
  }
  assert(last);
  last->set_adjacent(nullptr);
  _vofsl[num_bins - 1]->enqueue(last); // push on FIFO

  assert(lba > max_lba);

  if(!reused)
    _pmem->persist(_pmem_pages);

}


Block_allocator::
~Block_allocator()
{
  for(auto& p: _vofsl) {
    while(!p->empty()) {
      Segment *s;
      p->pop(s);
      delete s;
    }
    delete p;
  }

  _pmem->close(_pmem_pages);
  _pmem->release_ref();
}

/* IBlock_allocator */

/** 
 * Allocate N contiguous blocks
 * 
 * @param size Number of blocks to allocate
 * 
 * @return Logical block address of start of allocation. Throw exception on insufficient blocks.
 */
lba_t Block_allocator::alloc(size_t count, void ** handle)
{
  lba_t result;
  
  /* worst-fit-first */
  for(unsigned b = _last_bin_index; b > 0; b--) {

    if(count > bin_to_size(b)) break;

    if(option_DEBUG)
      PLOG("checking bin %u for count %ld", b, count);
    
    if(!_vofsl[b]->empty()) {
      
      Segment * s;
      while(!_vofsl[b]->dequeue(s) && !_vofsl[b]->empty());
      
      Segment * first_marker = s;
      
      do {
        if(s->alloc(count, result) == S_OK) {
          size_t new_max = s->max_allocation();

          /* put back on queue */
          auto bin = size_to_bin(new_max);
          assert(_vofsl[bin]);          
          bool status = _vofsl[bin]->enqueue(s);
          if(!status) throw Logic_exception("vofsl enqueue failed in alloc");
          
          if(handle)
            *handle = static_cast<void*>(s);
        
          return result;
        }
        else {
          /* put back on queue */
          bool status = _vofsl[b]->enqueue(s);
          if(!status) throw Logic_exception("vofsl enqueue failed in alloc");

          while(!_vofsl[b]->dequeue(s) && !_vofsl[b]->empty());
        }
      }
      while(s != first_marker);
    }
    else {
      if(option_DEBUG) {
        PLOG("bin %u was empty", b);
      }
    }   
  }
  throw General_exception("out of blocks in block-allocator");
  return 0;
}

/** 
 * Free a previous allocation
 * 
 * @param addr Logical block address of allocation
 */
void Block_allocator::free(lba_t lba, void* handle)
{
  /* NOTE: future alloc will shift the new size to the appropriate bin later */
  
  /* fast release - we find the lba in the segment and mark it as free */
  if(handle) { 
    Segment * s = static_cast<Segment *>(handle);
    if(!s->free(lba)) {
      throw General_exception("block_allocator free failed unexpectedly");
    }
    return;
  }

  /* slow path - go through all segments linearly */
  Segment * s = _root_segment;
  while(s) {
    if(s->free(lba))
      return;
    s = s->adjacent();
  }
  throw API_exception("unable to locate lba: bad free");
}

/** 
 * Attempt to resize an allocation without relocation
 * 
 * @param addr Logical block address of allocation
 * @param size New size in blocks
 * 
 * @return S_OK on successful resize; E_FAIL otherwise
 */
status_t Block_allocator::resize(lba_t addr, size_t size)
{
  return E_FAIL;
}

  
/** 
 * Get number of free units
 * 
 * 
 * @return Free capacity in units
 */
size_t Block_allocator::get_free_capacity()
{
  return 0;
}

/** 
 * Get total capacity
 * 
 * 
 * @return Capacity in units
 */
size_t Block_allocator::get_capacity()
{
  return 0;
}


void Block_allocator::dump_info()
{
}

/** 
 * Factory entry point
 * 
 */
extern "C" void * factory_createInstance(Component::uuid_t& component_id)
{
  if(component_id == Block_allocator_factory::component_id()) {
    return static_cast<void*>(new Block_allocator_factory());
  }
  else return NULL;
}

#undef RESET_STATE
