#ifndef __NUPM_MAPPERS_H__
#define __NUPM_MAPPERS_H__

namespace nupm
{
  

 
class Log2_bucket_mapper
{
  static constexpr size_t REGION_SIZE = MB(64); //GB(1);
public:  
  unsigned bucket(size_t object_size) {
    return get_log2_bin(object_size);
  }

  void * base(void * addr) {
    return round_down(addr,GB(1));
  }

  size_t region_size(size_t object_size) {
    return REGION_SIZE;
  }

  size_t rounded_up_object_size(size_t size) {
    size_t rup = 1UL << get_log2_bin(size);
    assert(rup >= size);
    return rup;
  }

private:
  inline static unsigned get_log2_bin(size_t a) {
    unsigned fsmsb = unsigned(((sizeof(size_t) * 8) - __builtin_clzl(a)));
    if ((addr_t(1) << (fsmsb - 1)) == a) fsmsb--;
    return fsmsb;
  }

  inline static void *round_down(void *p, unsigned long alignment) {
    if (mword_t(p) % alignment == 0) return p;
    return reinterpret_cast<void *>(mword_t(p) & ~(mword_t(alignment) - 1UL));
  }
 
};

} // nupm

#endif
