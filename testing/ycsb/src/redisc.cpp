#include "redisc.h"
#include <assert.h>
#include <mpi.h>
#include <string.h>
#include <iostream>

using namespace std;
using namespace ycsb;

RedisC::RedisC(Properties &props, unsigned core) { init(props, core); }
RedisC::~RedisC() { clean(); }

void RedisC::init(Properties &props, unsigned core)
{
  string address = props.getProperty("address");
  size_t mid     = address.find(":");
  string host    = address.substr(0, mid);
  int    port    = stoi(address.substr(mid + 1));
  int    rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  port += rank / 6;
  c              = redisConnect(host.c_str(), port);
  assert(c);
  if (c->err) {
    cout << c->errstr << endl;
    clean();
    exit(-1);
  }
}

int RedisC::get(const string &table,
                const string &key,
                char *        value,
                bool          direct)
{
  reply = (redisReply *) redisCommand(c, "GET %s", key.c_str());
  if (reply == NULL) return -1;
  // assert(strlen(reply->str) == strlen(value));
  memcpy(value, reply->str, strlen(value));
  freeReplyObject(reply);
  return 0;
}

int RedisC::put(const string &table,
                const string &key,
                const string &value,
                bool          direct)
{
  reply = (redisReply *) redisCommand(c, "SET %b %b", key.c_str(), key.length(),
                                      value.c_str(), value.length());
  if (reply == NULL) return -1;
  freeReplyObject(reply);
  return 0;
}

int RedisC::update(const string &table,
                   const string &key,
                   const string &value,
                   bool          direct)
{
  return put(table, key, value, direct);
}

int RedisC::erase(const string &table, const string &key)
{
  reply = (redisReply *) redisCommand(c, "DEL %s", key.c_str());
  if (reply == NULL) return -1;
  freeReplyObject(reply);
  return 0;
}

int RedisC::scan(const string &                table,
                 const string &                key,
                 int                           count,
                 vector<pair<string, string>> &results)
{
  return -1;
}

void RedisC::clean()
{
  redisCommand(c, "FLUSHALL");
  freeReplyObject(reply);
  redisFree(c);
}
