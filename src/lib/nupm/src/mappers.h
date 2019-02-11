#ifndef __NUPM_MAPPERS_H__
#define __NUPM_MAPPERS_H__

inline static unsigned get_log2_bin(size_t a) {
  unsigned fsmsb = unsigned(((sizeof(size_t) * 8) - __builtin_clzl(a)));
  if ((addr_t(1) << (fsmsb - 1)) == a) fsmsb--;
  return fsmsb;
}
  


namespace nupm
{

class Large_and_small_bucket_mapper
{
private:

  static constexpr size_t L0_MAX_SMALL_OBJECT_SIZE = KiB(4);
  static constexpr size_t L0_REGION_SIZE = MiB(1);
  
public:
  Large_and_small_bucket_mapper() :
    _other_bucket_index(get_log2_bin(L0_MAX_SMALL_OBJECT_SIZE) + 1) {
  }
  
  unsigned bucket(size_t object_size) {    
    if(object_size <= L0_MAX_SMALL_OBJECT_SIZE)
      return get_log2_bin(object_size);
    else {
      return _other_bucket_index;
    }
  }

  void * base(void * addr, size_t object_size) {
    if(object_size <= L0_MAX_SMALL_OBJECT_SIZE )
      return round_down(addr, L0_REGION_SIZE);
    else
      return addr;
  }

  size_t region_size(size_t object_size) {
    if(object_size <= L0_MAX_SMALL_OBJECT_SIZE)
      return L0_REGION_SIZE;
    else
      return object_size;
  }

  size_t rounded_up_object_size(size_t object_size) {
    if(object_size <= L0_MAX_SMALL_OBJECT_SIZE)
      return (1UL << get_log2_bin(object_size));
    else
      return object_size;
  }
  
private:

  const size_t _other_bucket_index;
};
  
class Log2_bucket_mapper
{
  static constexpr size_t REGION_SIZE = GB(1);
  
public:  
  unsigned bucket(size_t object_size) {
    return get_log2_bin(object_size);
  }

  void * base(void * addr, size_t size) {
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
 
};


  //using Bucket_mapper = Log2_bucket_mapper;
using Bucket_mapper = Large_and_small_bucket_mapper;
  

  
} // nupm

#endif
