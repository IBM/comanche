#ifndef __COMANCHE_EXCEPTION_H__
#define __COMANCHE_EXCEPTION_H__

class Comanche_exception
{
public:
  Comanche_exception()
  {
  }

  Comanche_exception(const char* cause)
  {
    PWRN("Comanche_exception: %s", cause);
    __builtin_strncpy(_cause, cause, 256);
  }

  const char* cause() const
  {
    return _cause;
  }

  void set_cause(const char* cause)
  {
    __builtin_strncpy(_cause, cause, 256);
  }

private:
  char _cause[256];
};

#endif // __COMANCHE_EXCEPTION_H__
