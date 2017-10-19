#ifndef __ITF_REF_H__
#define __ITF_REF_H__

namespace Component
{

template <class I>
class Itf_ref
{
public:
  Itf_ref(I * obj) : _obj(obj) {
    assert(obj);
  }
  
  ~Itf_ref() {
    _obj->release_ref();
  }

  I * get() {
    return _obj;
  }

  I* operator->() {
    return _obj;
  }
  
private:
  I * _obj;
};

} // Component
#endif
