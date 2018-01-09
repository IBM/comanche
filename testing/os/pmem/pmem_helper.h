#ifndef __PMEM_HELPER_H__
#define __PMEM_HELPER_H__


class Pmem_base
{
public:
  Pmem_base(std::string filename, size_t nb_size) : _nb_size(nb_size) {

    assert(nb_size >= PMEMOBJ_MIN_POOL);

    bool newobj = true;
    _pop = pmemobj_create(filename.c_str(), "mb_0", _nb_size * 2, 0666);
    if(_pop == NULL) {
      newobj = false;
      _pop = pmemobj_open(filename.c_str(), "mb_0");
      if(_pop == nullptr)
        throw Constructor_exception("pmemobj create/open failed size=%lu", nb_size);
    }
      
    assert(_pop);

    _base = pmemobj_root(_pop, nb_size + KB(4));

    _pbase = (void*) round_up_page((addr_t)pmemobj_direct(_base));

    PLOG("pbase: %p", _pbase);
    assert(_pbase);
    pmem_memset_persist(_pbase, 0, MB(4));
  }

  virtual ~Pmem_base() {
    pmemobj_close(_pop);
  }

  void * p_base() const { return (void*) _pbase; }

  size_t size() const { return _nb_size; }
  
protected:
  
  size_t           _nb_size;
  PMEMoid          _base;
  PMEMobjpool *    _pop;
  void * _pbase;
};

#endif
