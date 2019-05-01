#ifndef __DAWN_SERVER_TASK_H__
#define __DAWN_SERVER_TASK_H__

namespace Dawn
{

class Shard_task
{
public:
  Shard_task(Connection_handler* handler) : _handler(handler) {}
  virtual status_t do_work() = 0;
  virtual const void * get_result() const = 0;
  virtual size_t get_result_length() const = 0;
  virtual offset_t matched_position() const = 0;
  Connection_handler * handler() const { return _handler; }
  
protected:
  Connection_handler* _handler;
};

}

#endif // __DAWN_SERVER_TASK_H__
