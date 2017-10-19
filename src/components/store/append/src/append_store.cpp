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

static Component::IBlock_allocator *   create_block_allocator(Component::IPersistent_memory * pmem,
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
  sqlss << "(ID TEXT PRIMARY KEY NOT NULL, LBA INT8, LEN INT8, METADATA TEXT);";
   
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
  lba_t start_lba = _hdr.allocate(data_len, n_blocks); /* allocate contiguous segment of blocks */
  lba_t curr_lba = start_lba;
  size_t remaining_blocks = n_blocks;

  if(option_DEBUG)
    PLOG("[+] Append-store: append %ld bytes. Used blocks=%ld/%ld", data_len,
         start_lba+n_blocks, _vi.max_lba); 
  

  /* write max IO size segments */
  while(remaining_blocks > _max_io_blocks) {
    
    auto iob = _phys_mem_allocator.allocate_io_buffer(_max_io_bytes,
                                                      DMA_ALIGNMENT_BYTES,
                                                      NUMA_NODE_ANY);
  
    memcpy(_phys_mem_allocator.virt_addr(iob), p, _max_io_bytes);

    _block->async_write(iob, 0, curr_lba, _max_io_blocks, queue_id /* qid */,
                        &memory_free_cb, this, reinterpret_cast<void*>(iob));
    p+=_max_io_bytes;
    remaining_blocks -= _max_io_blocks;
    curr_lba += _max_io_blocks;
  }

  /* write remainder if anything left */
  if(remaining_blocks > 0) {
    auto bytes_left = remaining_blocks * _vi.block_size;
    auto iob = _phys_mem_allocator.allocate_io_buffer(bytes_left,
                                                      DMA_ALIGNMENT_BYTES,
                                                      NUMA_NODE_ANY);
    size_t n_blocks = 0;
    char * v = reinterpret_cast<char*>(_phys_mem_allocator.virt_addr(iob));
    v += (remaining_blocks - 1) * _vi.block_size;
    memset(v, 0, _vi.block_size - (data_len % _vi.block_size)); // zero trailer
    memcpy(_phys_mem_allocator.virt_addr(iob), p, bytes_left);
    
    /* issue async write */
    _block->async_write(iob, 0, curr_lba, remaining_blocks, queue_id /* qid */,
                        &memory_free_cb, this, reinterpret_cast<void*>(iob));
  }

  /* write metadata */
  insert_row(key, metadata, start_lba, n_blocks);
  
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
  lba_t start_lba = _hdr.allocate(data_len, n_blocks); /* allocate contiguous segment of blocks */
  lba_t curr_lba = start_lba;
  size_t remaining_blocks = n_blocks;

  if(option_DEBUG)
    PLOG("[+] Append-store: append %ld bytes. Used blocks=%ld/%ld", data_len,
         start_lba+n_blocks, _vi.max_lba); 
  

  /* write max IO size segments */
  uint64_t off = offset;
  while(remaining_blocks > _max_io_blocks) {
    
    _block->async_write(iob,
                        off,
                        curr_lba + 1, /* append store header */
                        _max_io_blocks,
                        queue_id);
    off += _max_io_bytes;
    remaining_blocks -= _max_io_blocks;
    curr_lba += _max_io_blocks;
  }

  assert(remaining_blocks <= _max_io_blocks);
  /* write remainder if anything left */
  if(remaining_blocks > 0) {   
    _block->async_write(iob,
                        off,
                        curr_lba + 1, /* append store header */
                        remaining_blocks,
                        queue_id);
  }

  /* write metadata */
  insert_row(key, metadata, start_lba, n_blocks);

  _block->check_completion(0);
  return S_OK;
}

struct __record_desc
{
  int64_t lba;
  int64_t len;
};
  
struct __iterator_t
{
  uint64_t current_idx;
  uint64_t exceeded_idx;
  std::vector<__record_desc> records;
};
  

IStore::iterator_t Append_store::open_iterator(uint64_t rowid_start,
                                               uint64_t rowid_end)
{
  __iterator_t *iter = new __iterator_t;
  assert(iter);
  iter->current_idx = 0;
  
  std::stringstream sqlss;
  sqlss << "SELECT LBA,LEN FROM " << _table_name << " WHERE ROWID >= " << rowid_start <<
    " AND ROWID <= " << rowid_end << ";";
  std::string sql = sqlss.str();

  sqlite3_stmt * stmt;
  sqlite3_prepare_v2(_db, sql.c_str(), sql.size(), &stmt, nullptr);
  int s;
  while((s = sqlite3_step(stmt)) != SQLITE_DONE) {
    iter->records.push_back({sqlite3_column_int64(stmt, 0),sqlite3_column_int64(stmt, 1)});
  }
  sqlite3_finalize(stmt);
  iter->exceeded_idx = iter->records.size();

  if(option_DEBUG)
    PLOG("opened iterator: records = %ld", iter->exceeded_idx);
    
  return iter;
}

/** 
 * Close iterator
 * 
 * @param iter Iterator
 */
void Append_store::close_iterator(IStore::iterator_t iter)
{
  assert(iter != nullptr);
  delete static_cast<__iterator_t*>(iter);
}


status_t Append_store::iterator_get(IStore::iterator_t iter,
                                    Component::io_buffer_t iob,
                                    size_t offset,
                                    int queue_id)
{
  auto i = static_cast<__iterator_t*>(iter);
  assert(i);
  if(i->current_idx == i->exceeded_idx) return E_NOT_FOUND;

  auto& record = i->records[i->current_idx];

  if(option_DEBUG) {
    PLOG("Append_store::iterator_get lba=%lu len=%lu", record.lba, record.len);
  }

  _lower_layer->read(iob,
                     offset,
                     record.lba + 1, /* add one for store header */
                     record.len,
                     queue_id);

  i->current_idx++;  
  return S_OK;
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
