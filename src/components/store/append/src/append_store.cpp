/*
   Copyright [2017] [IBM Corporation]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
#include <sstream>
#include <core/physical_memory.h>
#include "append_store.h"

using namespace Component;


class Semaphore {
public:
  Semaphore() { sem_init(&_sem,0,0); }
  inline void post() { sem_post(&_sem); }
  inline void wait() { sem_wait(&_sem); }
private:
  sem_t _sem;
};

struct __record_desc
{
  int64_t lba;
  int64_t len;
};

struct __prefetch_desc
{
  uint64_t gwid;
  Component::io_buffer_t iob;
};

static constexpr uint32_t APPEND_STORE_ITERATOR_MAGIC = 0x11110000;

struct __iterator_t
{
  uint32_t                            magic;
  uint64_t                            current_idx;
  uint64_t                            exceeded_idx;
  __prefetch_desc                     prefetch_desc;
  std::vector<__record_desc>          record_vector;
  std::vector<Component::io_buffer_t> iob_vector;
};
  


static int callback(void *NotUsed, int argc, char **argv, char **azColName) {
   int i;
   for(i = 0; i<argc; i++) {
      printf("%s = %s\n", azColName[i], argv[i] ? argv[i] : "NULL");
   }
   printf("\n");
   return 0;
}

static int print_callback(void *NotUsed, int argc, char **argv, char **azColName) {
   int i;
   for(i = 0; i < argc; i++) {
     //     printf("%s = %s\n", azColName[i], argv[i] ? argv[i] : "NULL");
     if(i==0)
       printf("%s[%d] %s, ", NORMAL_BLUE, i, argv[i] ? argv[i] : "NULL");
     else
       printf("%s, ", argv[i] ? argv[i] : "NULL");
   }
   printf("\n%s", ESC_END);
   return 0;
}

// forward decls
//
static Component::IBlock_allocator *
create_block_allocator(Component::IPersistent_memory * pmem,
                       size_t n_blocks,
                       std::string name,
                       bool force_init);



Append_store::Append_store(std::string owner,
                           std::string name,
                           Component::IBlock_device* block,
                           int flags)
  : _block(block),
    _hdr(block, owner, name, flags & FLAGS_FORMAT) /* initialize header object */
{
  int rc;
  
  if(owner.empty() || name.empty() || block == nullptr)
    throw API_exception("bad Append_store constructor parameters");

  _lower_layer = _block;
  _block->add_ref();
  
  assert(_block);
  _block->get_volume_info(_vi);
  PLOG("Append-store: block device capacity=%lu max_dma_block=%ld",
       _vi.max_lba, _vi.max_dma_len / _vi.block_size);

  _max_io_blocks = _vi.max_dma_len / _vi.block_size;
  _max_io_bytes  = _vi.max_dma_len;
  assert(_vi.max_dma_len % _vi.block_size == 0);
  
  /* database inititalization */
  _table_name = name;
  _db_filename = owner + "." + _table_name + ".db";

  if(flags & FLAGS_FORMAT) {
    std::remove(_db_filename.c_str());
  }
  
  if(option_DEBUG) {
    PLOG("Append-store: opening db %s", _db_filename.c_str());
  }

  if(sqlite3_open_v2(_db_filename.c_str(),
                     &_db,
                     SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX,
                     NULL) != SQLITE_OK) {
    throw General_exception("failed to open sqlite3 db (%s)", _db_filename.c_str());
  }

  /* create table if needed */
  std::stringstream sqlss;
  sqlss << "CREATE TABLE IF NOT EXISTS " << name;
  sqlss << "(ID TEXT PRIMARY KEY NOT NULL, LBA INT8, NBLOCKS INT8, METADATA TEXT);";
   
  execute_sql(sqlss.str());

}

Append_store::~Append_store()
{
  //  show_db();
  
  if(sqlite3_close(_db) != SQLITE_OK)
    throw General_exception("failed to close sqlite3 db (%s)", _db_filename.c_str());

  _block->release_ref();
}

uint64_t Append_store::insert_row(std::string& key, std::string& metadata, uint64_t lba, uint64_t length)
{
  std::stringstream sqlss;
  sqlss << "INSERT INTO " << _table_name << " VALUES ('" << key << "', " << lba << "," << length << ",'" << metadata << "')";
  execute_sql(sqlss.str());
  return 0;
}

bool Append_store::find_row(std::string& key, uint64_t& out_lba)
{
  std::stringstream sqlss;
  sqlss << "SELECT LBA FROM " << _table_name << " WHERE ID = '" << key << "';";
  std::string sql = sqlss.str();

  sqlite3_stmt * stmt;
  sqlite3_prepare_v2(_db, sql.c_str(), sql.size(), &stmt, nullptr);
  int s = sqlite3_step(stmt);
  if(s == SQLITE_ROW) {
    out_lba = sqlite3_column_int64(stmt, 0);
    sqlite3_finalize(stmt);
    return true;
  }
  else if(s == SQLITE_DONE) {
    sqlite3_finalize(stmt);
    return false;
  }
}

void Append_store::execute_sql(const std::string& sql, bool print_callback_flag)
{
  if(option_DEBUG) {
    PLOG("SQL:%s", sql.c_str());
  }

  char *zErrMsg = 0;
  if(print_callback_flag) {
    if(sqlite3_exec(_db, sql.c_str(), print_callback, 0, &zErrMsg)!=SQLITE_OK)
      throw General_exception("bad SQL statement (%s)", zErrMsg);    
  }
  else {
    if(sqlite3_exec(_db, sql.c_str(), callback, 0, &zErrMsg)!=SQLITE_OK)
      throw General_exception("bad SQL statement (%s)", zErrMsg);
  }
}

// IStore::item_t Append_store::open(std::string key)
// {
//   uint64_t lba;
//   if(find_row(key, lba)) {
//     return lba;
//   }
//   /* create new key */
//   // void * handle;
//   // lba_t new_lba = _blkallocator->alloc(1, &handle);
//   // insert_row(key, new_lba);
//   // show_db();
//   return 0;
// }

static void memory_free_cb(uint64_t gwid, void  *arg0, void* arg1)
{
  Append_store * pThis = reinterpret_cast<Append_store*>(arg0);
  io_buffer_t iob = reinterpret_cast<io_buffer_t>(arg1);
  pThis->phys_mem_allocator()->free_io_buffer(iob);
}

status_t Append_store::put(std::string key,
                           std::string metadata,
                           void * data,
                           size_t data_len,
                           int queue_id)
{
  assert(_block);
  assert(data_len > 0);
  assert(data);

  char * p = static_cast<char *>(data);
  size_t n_blocks;
  lba_t start_lba;

  try {
    start_lba = _hdr.allocate(data_len, n_blocks); /* allocate contiguous segment of blocks */
  }
  catch(...) { panic("unexpected condition"); }


  if(option_DEBUG)
    PLOG("[+] Append-store: append %ld bytes. Used blocks=%ld/%ld", data_len,
         start_lba+n_blocks, _vi.max_lba); 

  auto iob = _phys_mem_allocator.allocate_io_buffer(round_up(data_len,_vi.block_size),
                                                    DMA_ALIGNMENT_BYTES,
                                                    NUMA_NODE_ANY);
  assert(iob);
  
  memcpy(_phys_mem_allocator.virt_addr(iob), data, data_len);

  static __thread Semaphore sem;
  
  /* issue aync */
  _block->async_write(iob,
                      0,
                      start_lba, /* append store header */
                      n_blocks,
                      queue_id,
                      [](uint64_t gwid, void* arg0, void* arg1)
                      {
                        ((Semaphore *)arg0)->post();
                      },
                      (void*) &sem);

  /* write metadata */
  insert_row(key, metadata, start_lba, n_blocks);

  /* wait for io to complete */
  sem.wait();

  _phys_mem_allocator.free_io_buffer(iob);
  
  return S_OK;
}

status_t Append_store::put(std::string key,
                           std::string metadata,
                           Component::io_buffer_t iob,
                           size_t offset,
                           size_t data_len,
                           int queue_id)
{
  assert(_block);

  size_t n_blocks;
  lba_t start_lba;
  try {
    start_lba = _hdr.allocate(data_len, n_blocks); /* allocate contiguous segment of blocks */
  }
  catch(...) { panic("unexpected condition"); }

  if(option_DEBUG)
    PLOG("[+] Append-store: append %ld bytes. Used blocks=%ld/%ld", data_len,
         start_lba+n_blocks, _vi.max_lba); 

  static __thread Semaphore sem;

  /* issue aync */
  _block->async_write(iob,
                      offset,
                      start_lba, /* append store header */
                      n_blocks,
                      queue_id,
                      [](uint64_t gwid, void* arg0, void* arg1)
                      {
                        ((Semaphore *)arg0)->post();
                      },
                      (void*) &sem);

  /* write metadata */
  insert_row(key, metadata, start_lba, n_blocks);

  /* wait for io to complete */
  sem.wait();

  return S_OK;
}


IStore::iterator_t Append_store::open_iterator(uint64_t rowid_start,
                                               uint64_t rowid_end,
                                               unsigned prefetch_buffers)
{
  __iterator_t *iter = new __iterator_t;
  assert(iter);

  if(rowid_end < rowid_start)
    throw API_exception("open_iterator bad params");
  
  iter->current_idx = 0;
  iter->magic = APPEND_STORE_ITERATOR_MAGIC;
  iter->prefetch_desc = {0};
  
  std::stringstream sqlss;
  sqlss << "SELECT LBA,NBLOCKS FROM " << _table_name << " WHERE ROWID >= " << rowid_start <<
    " AND ROWID <= " << rowid_end << ";";
  std::string sql = sqlss.str();

  sqlite3_stmt * stmt;
  sqlite3_prepare_v2(_db, sql.c_str(), sql.size(), &stmt, nullptr);
  int s;
  while((s = sqlite3_step(stmt)) != SQLITE_DONE) {

    if(s == SQLITE_ERROR || s == SQLITE_MISUSE)
      throw API_exception("failed to open iterator: SQL statement failed (%s)", sql.c_str());
    
    iter->record_vector.push_back({sqlite3_column_int64(stmt, 0),sqlite3_column_int64(stmt, 1)});
  }
  sqlite3_finalize(stmt);
  iter->exceeded_idx = iter->record_vector.size();

  if(option_DEBUG) 
    PLOG("opened iterator (%p): records=%ld prefetch buffers=%u",
         iter, iter->exceeded_idx, prefetch_buffers);
    
  for(unsigned i=0;i<prefetch_buffers;i++) {
    auto iob = _lower_layer->allocate_io_buffer(MB(8),KB(4),NUMA_NODE_ANY);
    iter->iob_vector.push_back(iob);
  }
    
  return iter;
}

/** 
 * Close iterator
 * 
 * @param iter Iterator
 */
void Append_store::close_iterator(IStore::iterator_t iter)
{
  auto iteritf = static_cast<__iterator_t*>(iter);
  assert(iteritf != nullptr);
  
  if(option_DEBUG) 
    PLOG("closing iterator (%p) iob-count=%ld", iteritf, iteritf->iob_vector.size());

  // broken
  //  for(auto& e: iteritf->iob_vector)
  //    _lower_layer->free_io_buffer(e);
  
  delete iteritf;
}


size_t Append_store::iterator_get(iterator_t iter,
                                  Component::io_buffer_t* iob,
                                  size_t offset,
                                  int queue_id)
{
  assert(iob);
  auto i = static_cast<__iterator_t*>(iter);
  assert(i);

  if(unlikely(i->magic != APPEND_STORE_ITERATOR_MAGIC))
    throw API_exception("Append_store::iterator_get - bad iterator");
  
  if(i->current_idx == i->exceeded_idx) {
    PWRN("last record exceeded (%lu)", i->current_idx);
    return 0;
  }

  auto& record = i->record_vector[i->current_idx];

  if(option_DEBUG) {
    PLOG("Append_store::iterator_get lba=%lu len=%lu", record.lba, record.len);
  }

  if(unlikely(get_size(*iob) < record.len))
    throw API_exception("insufficient space in iob for record len");

  _lower_layer->read(*iob,
                     offset,
                     record.lba, /* add one for store header */
                     record.len,
                     queue_id);

  i->current_idx++;
  return record.len * _vi.block_size;
}


size_t Append_store::iterator_get(iterator_t iter,
                                  Component::io_buffer_t& iob,
                                  int queue_id)
{
  auto i = static_cast<__iterator_t*>(iter);
  assert(i);

  if(i->magic != APPEND_STORE_ITERATOR_MAGIC)
    throw API_exception("Append_store::iterator_get - bad iterator");

  if(i->iob_vector.empty())
    throw API_exception("over-called Append_store::iterator_get with freeing IOB");

  if(i->current_idx == i->exceeded_idx) {
    PWRN("last record exceeded");
    return 0;
  }

  auto& record = i->record_vector[i->current_idx];
  
  if(option_DEBUG)
    PLOG("Append_store::iterator_get lba=%lu len=%lu", record.lba, record.len);

  if(likely(i->prefetch_desc.gwid > 0)) { /* prefetch underway */
    while(!_lower_layer->check_completion(i->prefetch_desc.gwid))  cpu_relax();
    iob = i->prefetch_desc.iob;
  }
  else { /* no prefetch underway */    
    iob = i->iob_vector.back();
    i->iob_vector.pop_back();

    if(get_size(iob) < record.len)
      throw API_exception("insufficient space in IOB for record len");
    
    _lower_layer->read(iob,
                       0, // offset
                       record.lba, /* add one for store header */
                       record.len,
                       queue_id);    
  }
  
  i->current_idx++;      

  /* start asynchronous prefetch if possible */
  {
    if(i->current_idx != i->exceeded_idx) {
      auto& next_record = i->record_vector[i->current_idx];
      i->prefetch_desc.iob = i->iob_vector.back();
      i->iob_vector.pop_back();
      i->prefetch_desc.gwid = _lower_layer->async_read(i->prefetch_desc.iob,
                                                       0, // offset
                                                       next_record.lba + 1, /* add one for store header */
                                                       next_record.len,
                                                       queue_id);
    }    
  }
  
  return record.len * _vi.block_size;
}

void Append_store::free_iterator_buffer(iterator_t iter,
                                        Component::io_buffer_t iob)
{
  auto i = static_cast<__iterator_t*>(iter);
  assert(i);
  i->iob_vector.push_back(iob);
}



status_t Append_store::flush()
{
  _block->check_completion(0,0); /* wait for all pending */
  return S_OK;
}


void Append_store::dump_info()
{
  _hdr.dump_info();

  std::stringstream sqlss;
  sqlss << "SELECT * FROM " << _table_name << " LIMIT(100);";
  std::string sql = sqlss.str();

  /* dump keys */
  sqlite3_stmt * stmt;
  sqlite3_prepare_v2(_db, sql.c_str(), sql.size(), &stmt, nullptr);
  int s;
  while((s = sqlite3_step(stmt)) != SQLITE_DONE) {
    auto key = sqlite3_column_text(stmt, 0);
    auto start_lba = sqlite3_column_int64(stmt, 1);
    auto len = sqlite3_column_int64(stmt, 2);
    PLOG("start_lba=%lld len=%lld key: %s ", start_lba, len, key);
  }
  PLOG("...");
  sqlite3_finalize(stmt);

  /* read from file */
    
}

void Append_store::show_db()
{
  assert(_db);
  std::stringstream sqlss;
  sqlss << "SELECT * FROM " << _table_name << ";";
  execute_sql(sqlss.str(), true);
}

size_t Append_store::get_record_count()
{
  assert(_db);
  std::stringstream sqlss;
  sqlss << "SELECT MAX(ROWID) FROM " << _table_name << ";";
  std::string sql = sqlss.str();
  
  sqlite3_stmt * stmt;
  sqlite3_prepare_v2(_db, sql.c_str(), sql.size(), &stmt, nullptr);
  int s = sqlite3_step(stmt);
  auto max_row_id = sqlite3_column_int64(stmt, 0);
  sqlite3_finalize(stmt);
  return max_row_id;
}

status_t Append_store::get(uint64_t rowid,
                           Component::io_buffer_t iob,
                           size_t offset,
                           int queue_id)
{
  std::stringstream sqlss;
  sqlss << "SELECT * FROM " << _table_name << " WHERE ROWID=" << rowid << ";";
  std::string sql = sqlss.str();

  if(offset % _vi.block_size)
    throw API_exception("offset must be aligned with block size");
  
  sqlite3_stmt * stmt;
  sqlite3_prepare_v2(_db, sql.c_str(), sql.size(), &stmt, nullptr);
  int s = sqlite3_step(stmt);
  int64_t data_lba = sqlite3_column_int64(stmt, 1);
  int64_t data_len = sqlite3_column_int64(stmt, 2);
  sqlite3_finalize(stmt);
  if(option_DEBUG) {
    PLOG("get(rowid=%lu) --> lba=%ld len=%ld", rowid, data_lba, data_len);
  }

  if((_lower_layer->get_size(iob) - offset) < (data_len * _vi.block_size)) {
    if(option_DEBUG)
      PWRN("Append_store:get call with too smaller IO buffer");
    
    return E_INSUFFICIENT_SPACE;
  }

  _lower_layer->read(iob,
                     offset,
                     data_lba,
                     data_len,
                     queue_id);

  return S_OK;
}


std::string Append_store::get_metadata(uint64_t rowid)
{
  std::stringstream sqlss;
  sqlss << "SELECT ID FROM " << _table_name << " WHERE ROWID=" << rowid << ";";
  std::string sql = sqlss.str();

  sqlite3_stmt * stmt;
  sqlite3_prepare_v2(_db, sql.c_str(), sql.size(), &stmt, nullptr);
  int s = sqlite3_step(stmt);

  if(s == SQLITE_ERROR)
    throw API_exception("unable to get metadata for row %lu", rowid);
  std::string result;
  result = (char*) sqlite3_column_text(stmt, 0);
  sqlite3_finalize(stmt);

  return result;
}


/** 
 * Static functions
 * 
 */
static Component::IBlock_allocator *
create_block_allocator(Component::IPersistent_memory * pmem,
                       size_t n_blocks,
                       std::string name,
                       bool force_init)
{
  assert(pmem);
  
  IBase * comp = load_component("libcomanche-allocblock.so",
                                Component::block_allocator_factory);
  assert(comp);
  IBlock_allocator_factory * fact = static_cast<IBlock_allocator_factory *>
    (comp->query_interface(IBlock_allocator_factory::iid()));

  auto alloc = fact->open_allocator(pmem, n_blocks, name+"-blka", Component::NUMA_NODE_ANY, force_init);
  fact->release_ref();  
  assert(alloc);
  return alloc;
}

/** 
 * Factory entry point
 * 
 */
extern "C" void * factory_createInstance(Component::uuid_t& component_id)
{
  if(component_id == Append_store_factory::component_id()) {
    return static_cast<void*>(new Append_store_factory());
  }
  else return NULL;
}
