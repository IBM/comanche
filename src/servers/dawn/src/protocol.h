/*
   Copyright [2017-2019] [IBM Corporation]
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
#ifndef __DAWN_PROTOCOL_H__
#define __DAWN_PROTOCOL_H__

#include <assert.h>
#include <common/exceptions.h>
#include <common/logging.h>
#include <common/utils.h>
#include <api/dawn_itf.h>
#include <cstring>

#define PROTOCOL_VERSION (0xFB)
#define PROTOCOL_DEBUG

namespace Dawn
{

namespace Protocol
{
enum {
  MSG_TYPE_HANDSHAKE       = 0x1,
  MSG_TYPE_HANDSHAKE_REPLY = 0x2,
  MSG_TYPE_CLOSE_SESSION   = 0x3,
  MSG_TYPE_STATS           = 0x4,
  MSG_TYPE_POOL_REQUEST    = 0x10,
  MSG_TYPE_POOL_RESPONSE   = 0x11,
  MSG_TYPE_IO_REQUEST      = 0x20,
  MSG_TYPE_IO_RESPONSE     = 0x21,
  MSG_TYPE_INFO_REQUEST    = 0x30,
  MSG_TYPE_INFO_RESPONSE   = 0x31,
  MSG_TYPE_MAX             = 0xFF,
};

enum {
//   INFO_TYPE_VALUE_LEN = 0x1,
//   INFO_TYPE_COUNT     = 0x1,
  /* must be above IKVStore::Attributes */
  INFO_TYPE_FIND_KEY  = 0xF0,
  INFO_TYPE_GET_STATS = 0xF1,
};

enum {
  PROTOCOL_V1 = 0x1, /*< Key-Value Store */
  PROTOCOL_V2 = 0x2, /*< Memory-Centric Active Storage */
};

enum {
  MSG_RESVD_SCBE   = 0x2, /* indicates short-circuit function (testing only) */
  MSG_RESVD_DIRECT = 0x4, /* indicate get_direct from client side */
};

enum {
  OP_NONE        = 0,
  OP_CREATE      = 1,
  OP_OPEN        = 2,
  OP_CLOSE       = 3,
  OP_PUT         = 4,
  OP_SET         = 4,
  OP_GET         = 5,
  OP_PUT_ADVANCE = 6,  // allocate space for subsequence put or partial put
  OP_PUT_SEGMENT = 7,
  OP_DELETE      = 8,
  OP_ERASE       = 8,
  OP_PREPARE     = 9,  // prepare for immediately following operation
  OP_COUNT       = 10,
  OP_CONFIGURE   = 11,
  OP_STATS       = 12,
  OP_INVALID     = 0xFE,
  OP_MAX         = 0xFF
};

enum { S_OK = 0, E_KEY_EXISTS = 1, STATUS_MAX = 0xFF };

enum {
  IO_READ      = 0x1,
  IO_WRITE     = 0x2,
  IO_ERASE = 0x4,
  IO_MAX       = 0xFF,
};

/* Base for all messages */

struct Message {
  Message(uint64_t auth_id, uint8_t type_id, uint8_t op_param)
      : auth_id(auth_id), type_id(type_id), version(PROTOCOL_VERSION)
  {
    status = S_OK;
    assert(op_param);
    op = op_param;
    assert(this->op);
  }

  /* responses generally have no good value in the "op" field */
  Message(uint64_t auth_id, uint8_t type_id)
      : Message(auth_id, type_id, OP_INVALID)
  {
  }

  void print()
  {
    PLOG("Message:%p auth_id:%lu msg_len=%u type=%x", this, auth_id, msg_len,
         type_id);
  }

  /* Convert the response to the expected type after verifying that the
     type_id field matches what is expected.
   */
  template <typename Type>
  const Type *ptr_cast() const
  {
    if (this->type_id != Type::id)
      throw Protocol_exception("expected %s (0x%x) message - got 0x%x, len %lu",
                               Type::description, Type::id,
                               this->type_id, this->msg_len);
    return static_cast<const Type *>(this);
  }

  uint64_t auth_id;  // authorization token
  uint32_t msg_len;
  uint8_t  version;
  uint8_t  type_id;  // message type id
  union {
    uint8_t op;
    int8_t  status;
  };
  uint8_t resvd;
} __attribute__((packed));

namespace
{
  const Message *message_cast(const void *b)
  {
    auto pm = static_cast<const Message *>(b);
    assert(pm->version == PROTOCOL_VERSION);
    if (pm->version != PROTOCOL_VERSION)
    {
      Protocol_exception e("expected protocol version 0x%x, got got 0x%x", PROTOCOL_VERSION,
                           pm->version);
#if 0
      throw e;
#else
      PWRN("%s", e.cause()); 
#endif
    }
    return pm;
  }
}

static_assert(sizeof(Message) == 16, "Unexpected Message data structure size");

/* Constructor definitions */

////////////////////////////////////////////////////////////////////////
// POOL OPERATIONS - create, delete

struct Message_pool_request : public Message {

  static constexpr uint8_t id = MSG_TYPE_POOL_REQUEST;
  static constexpr const char *description = "Message_pool_request";

  Message_pool_request(size_t             buffer_size,
                       uint64_t           auth_id,
                       uint64_t           request_id,
                       size_t             pool_size,
                       uint8_t            op,
                       const std::string& pool_name)
      : Message(auth_id, id, op), pool_size(pool_size),
        expected_object_count(0)
  {
    assert(op);
    assert(this->op);
    if(buffer_size < (sizeof *this)) throw std::length_error(description);
    auto max_data_len = buffer_size - (sizeof *this);

    size_t len = pool_name.length();
    if (len >= max_data_len)
      throw std::length_error(description);

    strncpy(data, pool_name.c_str(), len);
    data[len] = '\0';

    msg_len = (sizeof *this) + len + 1;
  }

  Message_pool_request(size_t   buffer_size,
                       uint64_t auth_id,
                       uint64_t request_id,
                       uint8_t  op)
      : Message_pool_request(buffer_size, auth_id, request_id, 0, op, "")
  {
  }

  const char* pool_name() const { return data; }

  size_t pool_size; /*< size of pool in bytes */
  size_t expected_object_count;
  union {
    uint64_t pool_id;
    uint32_t flags;
  };
  char data[]; /*< unique name of pool (for this client) */

} __attribute__((packed));

struct Message_pool_response : public Message {
  static constexpr uint8_t id = MSG_TYPE_POOL_RESPONSE;
  static constexpr const char *description = "Message_pool_response";

  Message_pool_response(uint64_t auth_id)
      : Message(auth_id, id)
  {
    msg_len = (sizeof *this);
  }

  uint64_t pool_id;
  char     data[];
} __attribute__((packed));

////////////////////////////////////////////////////////////////////////
// IO OPERATIONS

struct Message_IO_request : public Message {
  static constexpr uint8_t id = MSG_TYPE_IO_REQUEST;
  static constexpr const char *description = "Message_IO_request";

  Message_IO_request(size_t      buffer_size,
                     uint64_t    auth_id,
                     uint64_t    request_id,
                     uint64_t    pool_id,
                     uint8_t     op,
                     const void* key,
                     size_t      key_len,
                     const void* value,
                     size_t      value_len,
                     uint32_t    flags)
      : Message(auth_id, MSG_TYPE_IO_REQUEST, op), request_id(request_id),
        pool_id(pool_id), flags(flags)
  {
    set_key_and_value(buffer_size, key, key_len, value, value_len);
    msg_len = sizeof(Message_IO_request) + key_len + value_len + 1;
  }

  Message_IO_request(size_t             buffer_size,
                     uint64_t           auth_id,
                     uint64_t           request_id,
                     uint64_t           pool_id,
                     uint8_t            op,
                     const std::string& key,
                     const std::string& value,
                     uint32_t           flags)
      : Message_IO_request(buffer_size, auth_id, request_id, pool_id, op, key.data(), key.size(), value.data(), value.size(), flags)
  {
  }

  /* key, and data_length */
  Message_IO_request(size_t      buffer_size,
                     uint64_t    auth_id,
                     uint64_t    request_id,
                     uint64_t    pool_id,
                     uint8_t     op,
                     const void* key,
                     size_t      key_len,
                     size_t      value_len,
                     uint32_t    flags)
      : Message(auth_id, id, op), request_id(request_id),
        pool_id(pool_id), flags(flags)
  {
    set_key_value_len(buffer_size, key, key_len, value_len);
    msg_len = (sizeof *this) + key_len + 1; /* we don't add value len, this will be in next buffer */
  }

  Message_IO_request(size_t       buffer_size,
                     uint64_t     auth_id,
                     uint64_t     request_id,
                     uint64_t     pool_id,
                     uint8_t      op,
                     const std::string& key,
                     size_t       value_len,
                     uint32_t     flags)
      : Message_IO_request(buffer_size, auth_id, request_id, pool_id, op, key.data(), key.size(), value_len, flags)
  {
  }

  /*< version used for configure_pool command */
  Message_IO_request(size_t       buffer_size,
                     uint64_t     auth_id,
                     uint64_t     request_id,
                     uint64_t     pool_id,
                     uint8_t      op,
                     const std::string& data)
      : Message_IO_request(buffer_size, auth_id, request_id, pool_id, op, data.data(), data.size(), 0, 0)
  {
  }

  const char* key() const { return &data[0]; }
  const char* cmd() const { return &data[0]; }
  const char* value() const { return &data[key_len + 1]; }

  const size_t get_key_len() const { return key_len; }
  const size_t get_value_len() const { return val_len; }

  void set_key_value_len(size_t       buffer_size,
                         const void*  key,
                         const size_t key_len,
                         const size_t value_len)
  {
    if (unlikely((key_len + 1 + (sizeof *this)) > buffer_size))
      throw API_exception(
          "%s::%s - insufficient buffer for "
          "key-value_len pair (key_len=%lu) (val_len=%lu)",
          description, __func__, key_len, value_len);

    memcpy(data, key, key_len); /* only copy key and set value length */
    data[key_len] = '\0';
    this->val_len = value_len;
    this->key_len = key_len;
  }

  void set_key_and_value(const size_t buffer_size,
                         const void*  p_key,
                         const size_t p_key_len,
                         const void*  p_value,
                         const size_t p_value_len)
  {
    assert(buffer_size > 0);
    if (unlikely((p_key_len + p_value_len + 1 + (sizeof *this)) >
                 buffer_size))
      throw API_exception(
          "%s::%s - insufficient buffer for "
          "key-value pair (key_len=%lu) (val_len=%lu) (buffer_size=%lu)",
          description, __func__, p_key_len, p_value_len, buffer_size);

    memcpy(data, p_key, p_key_len);
    data[p_key_len] = '\0';
    memcpy(&data[p_key_len + 1], p_value, p_value_len);
    this->val_len = p_value_len;
    this->key_len = p_key_len;
  }

  // fields
  uint64_t pool_id;
  uint64_t request_id; /*< id or sender timestamp counter */
  uint64_t key_len;
  uint64_t val_len;
  uint32_t flags;
  uint32_t padding;
  char     data[];

} __attribute__((packed));

struct Message_IO_response : public Message {
  static constexpr uint64_t BIT_TWOSTAGE = 1ULL << 63;
  static constexpr uint8_t id = MSG_TYPE_IO_RESPONSE;
  static constexpr const char *description = "Message_IO_response";

  Message_IO_response(size_t buffer_size, uint64_t auth_id)
      : Message(auth_id, id)
  {
    data_len = 0;
    msg_len  = (sizeof *this);
  }

  void copy_in_data(const void* in_data, size_t len)
  {
    assert((len + (sizeof *this)) < data_len);
    memcpy(data, in_data, len);
    data_len = len;
    msg_len  = (sizeof *this) + data_len;
  }

  size_t base_message_size() const { return (sizeof *this); }

  void set_twostage_bit() { data_len |= BIT_TWOSTAGE; }

  bool is_set_twostage_bit() const { return data_len & BIT_TWOSTAGE; }

  size_t data_length() const { return data_len & (~BIT_TWOSTAGE); }

  // fields
  uint64_t request_id; /*< id or sender time stamp counter */
  uint64_t data_len;   /* bit 63 is twostage flag */
  char     data[];
} __attribute__((packed));

////////////////////////////////////////////////////////////////////////
// INFO REQUEST/RESPONSE
struct Message_INFO_request : public Message {
  static constexpr uint8_t id = MSG_TYPE_INFO_REQUEST;
  static constexpr const char *description = "Message_INFO_request";

  Message_INFO_request(uint64_t authid) : Message(auth_id, id),
                                          key_len(0) {
  }

  const char* key() const { return data; }
  const char* c_str() const { return data; }
  size_t base_message_size() const { return (sizeof *this); }
  size_t message_size() const { return (sizeof *this) + key_len + 1; }

  void set_key(const size_t buffer_size,
               const std::string& key) {
    key_len = key.length();
    if((key_len + base_message_size() + 1) > buffer_size)
      throw API_exception("%s::%s - insufficient buffer for key (len=%lu)",
                          description, __func__, key_len);

    memcpy(data, key.c_str(), key_len);
    data[key_len] = '\0';
  }
  
  // fields
  uint64_t pool_id;
  uint32_t type;
  uint32_t pad;
  uint64_t offset;
  uint64_t key_len;
  char     data[0];  
} __attribute__((packed));

struct Message_INFO_response : public Message {
  static constexpr uint8_t id = MSG_TYPE_INFO_RESPONSE;
  static constexpr const char *description = "Message_INFO_response";

  Message_INFO_response(uint64_t authid) : Message(auth_id, id) { }

  size_t base_message_size() const { return (sizeof *this); }
  size_t message_size() const { return (sizeof *this) + value_len + 1; }
  const char * c_str() const { return static_cast<const char*>(data); }

  void set_value(size_t buffer_size, const void * value, size_t len) {
    if (unlikely((len + 1 + (sizeof *this)) > buffer_size))
      throw API_exception("%s::%s - insufficient buffer (value len=%lu)", description, __func__, len);

    memcpy(data, value, len); /* only copy key and set value length */
    data[len] = '\0';
    value_len = len; 
  }
  
  // fields
  union {
    size_t value;
    size_t value_len;
  };
  offset_t offset;
  char   data[0];
} __attribute__((packed));

////////////////////////////////////////////////////////////////////////
// HANDSHAKE

struct Message_handshake : public Message {
  Message_handshake(uint64_t auth_id, uint64_t sequence)
      : Message(auth_id, id), seq(sequence),
        protocol(PROTOCOL_V1)
  {
    msg_len = (sizeof *this);
  }

  static constexpr uint8_t id = MSG_TYPE_HANDSHAKE;
  static constexpr const char *description = "Message_handshake";
  // fields
  uint64_t seq;
  uint8_t  protocol;

  void set_as_protocol() { protocol = PROTOCOL_V2; }

} __attribute__((packed));

////////////////////////////////////////////////////////////////////////
// HANDSHAKE REPLY

struct Message_handshake_reply : public Message {
  static constexpr uint8_t id = MSG_TYPE_HANDSHAKE_REPLY;
  static constexpr const char *description = "Message_handshake_reply";

  Message_handshake_reply(uint64_t auth_id,
                          uint64_t sequence,
                          uint64_t session_id,
                          size_t   mms)
      : Message(auth_id, id), seq(sequence),
        session_id(session_id), max_message_size(mms)
  {
    msg_len = (sizeof *this);
  }

    // fields
  uint64_t seq;
  uint64_t session_id;
  size_t   max_message_size;

} __attribute__((packed));

////////////////////////////////////////////////////////////////////////
// CLOSE SESSION

struct Message_close_session : public Message {
  static constexpr uint8_t id = MSG_TYPE_CLOSE_SESSION;
  static constexpr const char *description = "Message_close_session";

  Message_close_session(uint64_t auth_id)
      : Message(auth_id, id)
  {
    msg_len = (sizeof *this);
  }

  // fields
  uint64_t seq;

} __attribute__((packed));


struct Message_stats : public Message {

  static constexpr uint8_t id = MSG_TYPE_STATS;
  static constexpr const char *description = "Message_stats";

  Message_stats(uint64_t auth,
                const Component::IDawn::Shard_stats& shard_stats) : Message(auth_id, id)
  {
    stats = shard_stats;
  }

  size_t message_size() const { return sizeof(Message_stats); }
  // fields
  Component::IDawn::Shard_stats stats; 
} __attribute__((packed));



static_assert(sizeof(Message_IO_request) % 8 == 0,
              "Message_IO_request should be 64bit aligned");
static_assert(sizeof(Message_IO_response) % 8 == 0,
              "Message_IO_request should be 64bit aligned");

}  // namespace Protocol
}  // namespace Dawn

#endif
