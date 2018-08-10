#ifndef __DATA_H__
#define __DATA_H__
#include <vector>

class KV_pair 
{
public:
    std::string key;
    std::string value;

    KV_pair()
    {

    }

    ~KV_pair()
    {

    }

};

class Data
{
public:

    size_t _num_elements = 1000000;
    size_t _key_len = 8;  
    size_t _val_len = 64; 

    KV_pair * _data;
  
    Data() 
    {
        initialize_data(); 
    }
 
    Data(size_t num_elements)
    {
        _num_elements = num_elements;

        initialize_data();
    }

    Data(size_t num_elements, size_t key_len, size_t val_len)
    {
        _num_elements = num_elements;
        _key_len = key_len;
        _val_len = val_len;

        initialize_data();
    }

    ~Data() 
    {
      delete [] _data;
    }

    void initialize_data()
    {
        PLOG("Initializing data: %d key length, %d value length, %d elements....", (int)_key_len, (int)_val_len, (int)_num_elements);
        _data = new KV_pair[_num_elements];
    
        for(size_t i=0;i<_num_elements;i++) 
        {
            auto key = Common::random_string(_key_len);
            auto val = Common::random_string(_val_len);

            _data[i].key = key;
            _data[i].value = val;
        }

        PLOG("..OK.");
    }
  
    const char * key(size_t i) const 
    {
      if(i >= _num_elements) throw General_exception("out of bounds");
      return _data[i].key.c_str();
    }
  
    const char * value(size_t i) const 
    {
      if(i >= _num_elements) throw General_exception("out of bounds");
      return _data[i].value.c_str();
    }
  
    size_t value_len() const { return _val_len; }
  
    size_t num_elements() const { return _num_elements; }
};

#endif
