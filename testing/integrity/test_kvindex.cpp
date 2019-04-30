/* note: we do not include component source, only the API definition */
#include <api/components.h>
#include <api/kvindex_itf.h>
#include <common/str_utils.h>
#include <common/utils.h>
#include <gtest/gtest.h>
#include <boost/filesystem.hpp>
#include <iostream>
#include "component_info.h"

using namespace Component;

ComponentInfo component_info;

class IndexTest : public ::testing::Test {
protected:
  Component::IKVIndex*        _g_index;
  IKVIndex_factory*           _fact;

  void SetUp() override
  {
    component_info.load_component();
    _g_index = component_info.index;
    _fact    = component_info.index_factory;
  }

  void TearDown() override
  {
    _fact->release_ref();
  }
};

TEST_F(IndexTest, PutGet_RandomK)
{
  // randomly generate key and value
  const int key_length = 8;
  std::string key        = Common::random_string(key_length);

  _g_index->insert(key);

  std::string rc = _g_index->get(0);

  ASSERT_STREQ(rc.c_str(), key.c_str());
}

TEST_F(IndexTest, Get_NoValidKey)
{
  // randomly generate key to look up (we shouldn't find it)
  const int key_length = 8;
  const std::string key = Common::random_string(key_length);
  IKVIndex::offset_t end        = _g_index->count() - 1;

  std::string rc;
  status_t hr = _g_index->find(key, 0, IKVIndex::FIND_TYPE_EXACT, end, rc);
  ASSERT_TRUE(hr == S_OK);
  ASSERT_STREQ(rc.c_str(), "");
}

TEST_F(IndexTest, Put_DuplicateKey)
{
  // randomly generate key and value
  const int key_length = 8;
  const std::string key        = Common::random_string(key_length);
  int               count      = _g_index->count();

  _g_index->insert(key);
  _g_index->insert(key);

  ASSERT_EQ(_g_index->count(), count);
}

TEST_F(IndexTest, Put_Erase)
{
  const int         key_length = 8;
  const std::string key        = Common::random_string(key_length);
  IKVIndex::offset_t end        = _g_index->count() - 1;

  _g_index->insert(key);
  _g_index->erase(key);
  std::string rc;
  status_t hr = _g_index->find(key, 0, IKVIndex::FIND_TYPE_EXACT, end, rc);
  ASSERT_TRUE(hr == S_OK);
  ASSERT_STREQ(rc.c_str(), "");
}

TEST_F(IndexTest, Put_EraseInvalid)
{
  // randomly generate key and value
  const int key_length = 8;

  const std::string key = Common::random_string(key_length);
  int                count = _g_index->count();
  IKVIndex::offset_t end   = count - 1;
  std::string rc;
  status_t hr = _g_index->find(key, 0, IKVIndex::FIND_TYPE_EXACT, end, rc);
  ASSERT_TRUE(hr == S_OK);
  ASSERT_STREQ(rc.c_str(), "");

  _g_index->erase(key);

  ASSERT_EQ(_g_index->count(), count) << "erase return code failed";
}


struct {
  std::string test;
  std::string component;
  unsigned cores;
  unsigned time_secs;
} Options;

int main(int argc, char **argv) 
{
  component_info.initialize_component(argc, argv);

  testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}

