#ifndef COMANCHE_HSTORE_NUPM_REGION_H
#define COMANCHE_HSTORE_NUPM_REGION_H

/* requires persist_data_t definition */

class region
{
  static constexpr std::uint64_t magic_value = 0xc74892d72eed493a;
  std::uint64_t magic;
public:
  persist_data_t persist_data;
#if USE_CC_HEAP == 3
  heap_rc heap;
#else
  heap_cc heap;
#endif

  void initialize() { magic = magic_value; }
  bool is_initialized() const noexcept { return magic == magic_value; }
  /* region used by heap_cc follows */
};

#endif
