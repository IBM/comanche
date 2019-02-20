#include "dawn_client_jni.h"
#include <api/components.h>
#include <api/kvstore_itf.h>
#include <common/cpu.h>
#include <common/str_utils.h>
#include <core/dpdk.h>
#include <core/task.h>
#include <gtest/gtest.h>
#include <sys/mman.h>
#include <boost/program_options.hpp>
#include <iostream>

using namespace Component;
using namespace Common;
using namespace std;

Component::IKVStore *client;

string get_string(JNIEnv *env, jstring jstr)
{
  const char *chars = env->GetStringUTFChars(jstr, NULL);
  string str(chars);
  env->ReleaseStringUTFChars(jstr, chars);
  return str;
}

JNIEXPORT void JNICALL Java_DawnClient_init(JNIEnv *env,
                                            jobject obj,
                                            jint    debug,
                                            jstring user,
                                            jstring addr,
                                            jstring device)
{
  Component::IBase *comp = Component::load_component(
      "libcomanche-dawn-client.so", dawn_client_factory);

  IKVStore_factory *fact =
      (IKVStore_factory *) comp->query_interface(IKVStore_factory::iid());
  string username = get_string(env, user);
  string address  = get_string(env, addr);
  string dev      = get_string(env, device);

  client = fact->create(debug, username, address, dev);
  fact->release_ref();
}

JNIEXPORT jint JNICALL Java_DawnClient_put(JNIEnv *   env,
                                           jobject    obj,
                                           jstring    table,
                                           jstring    key,
                                           jbyteArray value,
                                           jboolean   direct)
{
  string p = get_string(env, table);
  /* open or create pool */
  Component::IKVStore::pool_t pool =
      client->open_pool("/mnt/pmem0/dawn", p.c_str(), 0);

  if (pool == Component::IKVStore::POOL_ERROR) {
    /* ok, try to create pool instead */
    pool = client->create_pool("/mnt/pmem0/dawn", p.c_str(), GB(1));
  }
  string k      = get_string(env, key);
  jbyte *buffer = env->GetByteArrayElements(value, NULL);
  jsize  length = env->GetArrayLength(value);
  jint   ret    = 0;
  if (direct) {
    auto handle = client->register_direct_memory(buffer, length);
    ret         = client->put_direct(pool, k, buffer, length, handle);
    client->unregister_direct_memory(handle);
  }
  else {
    ret = client->put(pool, k, buffer, length);
  }
  env->ReleaseByteArrayElements(value, buffer, JNI_ABORT);
  client->close_pool(pool);
  return ret;
}

JNIEXPORT jint JNICALL Java_DawnClient_get(JNIEnv *   env,
                                           jobject    obj,
                                           jstring    table,
                                           jstring    key,
                                           jbyteArray value,
                                           jboolean   direct)
{
  string p = get_string(env, table);
  /* open or create pool */
  Component::IKVStore::pool_t pool =
      client->open_pool("/mnt/pmem0/dawn", p.c_str(), 0);

  if (pool == Component::IKVStore::POOL_ERROR) {
    /* ok, try to create pool instead */
    pool = client->create_pool("/mnt/pmem0/dawn", p.c_str(), GB(1));
  }
  string k      = get_string(env, key);
  jint   ret    = 0;
  jbyte *buffer = env->GetByteArrayElements(value, NULL);
  if (direct) {
    size_t length = env->GetArrayLength(value);
    auto handle = client->register_direct_memory(buffer, length);
    ret         = client->get_direct(pool, k, buffer, length, handle);
    client->unregister_direct_memory(handle);
  }
  else {
    void * b   = nullptr;
    size_t len = 0;
    ret        = client->get(pool, k, b, len);
    memcpy(buffer, b, len);
    client->free_memory(b);
  }

  env->ReleaseByteArrayElements(value, buffer, 0);
  client->close_pool(pool);
  return ret;
}

JNIEXPORT jint JNICALL Java_DawnClient_erase(JNIEnv *env,
                                             jobject obj,
                                             jstring table,
                                             jstring key)
{
  string p = get_string(env, table);
  /* open or create pool */
  Component::IKVStore::pool_t pool =
      client->open_pool("/mnt/pmem0/dawn", p.c_str(), 0);

  if (pool == Component::IKVStore::POOL_ERROR) {
    /* ok, try to create pool instead */
    pool = client->create_pool("/mnt/pmem0/dawn", p.c_str(), GB(1));
  }
  string k = get_string(env, key);
  jint   ret = client->erase(pool, k);
  client->close_pool(pool);
  return ret;
}

JNIEXPORT jint JNICALL Java_DawnClient_clean(JNIEnv *env, jobject obj)
{
  client->release_ref();
}
