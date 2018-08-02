#ifndef __DATA_H__
#define __DATA_H__

class Data
{
public:

    size_t NUM_ELEMENTS = 1000000; // 1M
    static constexpr size_t KEY_LEN = 8;  // TODO: allow variable key lengths?
    static constexpr size_t VAL_LEN = 64;  // TODO: allow variable value lengths?
  
    struct KV_pair {
        char key[KEY_LEN + 1];
        char value[VAL_LEN + 1];
    };
  
    KV_pair * _data;
  
    Data() 
    {
        initialize_data(); 
    }
 
    Data(size_t num_elements)
    {
        NUM_ELEMENTS = num_elements;

        initialize_data();
    }

    ~Data() 
    {
      delete [] _data;
    }

    void initialize_data()
    {
        PLOG("Initializing data with %d elements....", (int)NUM_ELEMENTS);
        _data = new KV_pair[NUM_ELEMENTS];
    
        for(size_t i=0;i<NUM_ELEMENTS;i++) 
        {
            auto key = Common::random_string(KEY_LEN);
            auto val = Common::random_string(VAL_LEN);
            strncpy(_data[i].key, key.c_str(), key.length());
            _data[i].key[KEY_LEN] = '\0';
            strncpy(_data[i].value, val.c_str(), val.length());
           _data[i].value[VAL_LEN] = '\0';
        }

        PLOG("..OK.");
    }
  
    const char * key(size_t i) const 
    {
      if(i >= NUM_ELEMENTS) throw General_exception("out of bounds");
      return _data[i].key;
    }
  
    const char * value(size_t i) const 
    {
      if(i >= NUM_ELEMENTS) throw General_exception("out of bounds");
      return _data[i].value;
    }
  
    size_t value_len() const { return VAL_LEN; }
  
    size_t num_elements() const { return NUM_ELEMENTS; }
};

#endif
