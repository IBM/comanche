#ifndef __DAWN_SERVER_TASK_KEY_FIND_H__
#define __DAWN_SERVER_TASK_KEY_FIND_H__

#include <string>
#include <unistd.h>
#include "task.h"

namespace Dawn
{

/** 
 * Key search task.  We limit the number of hops we search so as to bound
 * the worst case execution time.
 * 
 */
class Key_find_task : public Shard_task
{
  static constexpr unsigned MAX_COMPARES_PER_WORK = 5;
  static const unsigned _debug_level = 0;
  
public:
  Key_find_task(const std::string& expression,
                offset_t offset,
                Connection_handler* handler,
                Component::IKVIndex* index) :
    Shard_task(handler), _offset(offset), _index(index)
  {
    using namespace Component;
    assert(_index);
    _index->add_ref();

    if(_debug_level > 0) {
      PLOG("offset=%lu", offset);
      PLOG("expr: (%s)", expression.c_str());
    }
    
    if(expression == "next:") {
      _type = IKVIndex::FIND_TYPE_NEXT;
      _expr = expression.substr(5);
    }
    else if(expression.substr(0,6) == "regex:") {
      _type = IKVIndex::FIND_TYPE_REGEX;
      _expr = expression.substr(6);
    }
    else if(expression.substr(0,6) == "exact:") {
      _type = IKVIndex::FIND_TYPE_EXACT;
      _expr = expression.substr(6);
    }
    else if(expression.substr(0,7) == "prefix:") {
      _type = IKVIndex::FIND_TYPE_PREFIX;
      _expr = expression.substr(7);
    }
    else throw Logic_exception("unhandled expression");
    
    assert(_index);
  }

  ~Key_find_task() {
    _index->release_ref();
  }

    //   case IKVIndex::FIND_TYPE_REGEX:
    // case IKVIndex::FIND_TYPE_EXACT:
    // case IKVIndex::FIND_TYPE_PREFIX:
    // }


  
  status_t do_work() {

    using namespace Component;

    status_t hr;
    try {
      hr = _index->find(_expr, _offset, _type, _offset, _out_key, MAX_COMPARES_PER_WORK);
      PLOG("OFFSET=%lu", _offset);
      if(hr == IKVIndex::E_MAX_REACHED) {
        _offset++;
        return Component::IKVStore::S_MORE;
      }
      else if(hr == S_OK) {
        PLOG("matched: (%s)", _out_key.c_str());
        return S_OK;
      }
    }
    catch(std::out_of_range e) {
      return E_FAIL;
    }
    
    throw Logic_exception("unexpected code path (hr=%d)", hr);
  }
  
  const void * get_result() const override {
    return _out_key.data();
  }
  
  size_t get_result_length() const override {
    return _out_key.length();
  }

  offset_t matched_position() const override {
    return _offset;
  }

private:
  std::string                 _expr;
  std::string                 _out_key;
  Component::IKVIndex::find_t _type;
  offset_t                    _offset;
  Component::IKVIndex*        _index;

};

}
#endif // __DAWN_SERVER_TASK_KEY_FIND_H__
