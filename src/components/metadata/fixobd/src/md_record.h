#ifndef __MD_RECORD_H__
#define __MD_RECORD_H__

#include <common/spinlocks.h>

static constexpr unsigned MD_MAGIC = 1972;

enum {
  MD_STATUS_NOT_ASSIGN = 0,
  MD_STATUS_FREE = 1,
  MD_STATUS_USED = 2,
};

enum {
  MD_BLOCK_SIZE_512 = 0,
  MD_BLOCK_SIZE_4096 = 1,
};

/** 
 * Manage locks in a seperate array from the records
 * 
 */
class Lock_array
{
public:
  Lock_array(unsigned nlocks) : _n_locks(nlocks) {
    _locks = new Common::Ticket_lock[nlocks]() ;
  }
  ~Lock_array() {
    delete [] _locks;
  }

  inline void lock(unsigned long index) {
    assert(index < _n_locks);
    _locks[index].lock();
  }

  inline void unlock(unsigned long index) {
    assert(index < _n_locks);
    _locks[index].unlock();
  }
  
private:
  size_t                _n_locks;
  Common::Ticket_lock * _locks;
};


struct __md_record
{
  struct {
    unsigned magic:      29;
    unsigned status:     2;
    unsigned block_size: 1;
  };
  uint32_t index;
  uint32_t crc;
  uint32_t resvd;
  uint64_t start_lba;       
  uint64_t lba_count;
    
  char id[64];     
  char owner[64];  
  char datatype[32];
  char utc_modified[32]; // e.g. 2017-11-16T00:08:24+00:00
  char utc_created[32];  // e.g. 2017-11-16T00:08:24+00:00

  // helpers
  bool check_magic() const { return magic == MD_MAGIC; }
  void set_magic() { magic = MD_MAGIC; }
  void clear() {
    memset(this, 0, sizeof(struct __md_record));
    set_magic();
    status = MD_STATUS_FREE;
  }
  inline void set_id(const char * id_param) {
    strncpy(id, id_param, 63); id[63]='\0';
  }
  inline void set_owner(const char * owner_param) {
    strncpy(owner, owner_param, 63); id[63]='\0';
  }
  inline void set_datatype(const char * param) {
    strncpy(datatype, param, 31); id[31]='\0';
  }
  inline void set_used() { status = MD_STATUS_USED; }
  inline void set_free() { status = MD_STATUS_FREE; }
  inline bool is_used() const { return status == MD_STATUS_USED; }
  inline bool is_free() const { return status == MD_STATUS_FREE; }
  void dump_record() {
    PINF("%p: %08ld-%08ld %s %s %s", this,
         start_lba,
         start_lba + lba_count,
         id, owner, datatype);
  }
  
} __attribute__((packed));


static_assert(sizeof(Common::Ticket_lock) == 4, "unexpected ticket lock size");
static_assert(sizeof(__md_record)==256,"md record size invalid");



#endif
