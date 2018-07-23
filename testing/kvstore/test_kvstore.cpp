#include <gtest/gtest.h>

/* note: we do not include component source, only the API definition */
#include <api/components.h>
#include <api/kvstore_itf.h>
#include <common/utils.h>
#include <common/str_utils.h>
#include <core/task.h>
//#include <chrono>
#include <iostream>

#define FILESTORE_PATH "libcomanche-storefile.so"

using namespace Component;

class PoolTest: public::testing::Test
{
protected:
    Component::IKVStore * g_store;
    Component::IBase * comp;
    IKVStore_factory * fact;

    Component::IKVStore::pool_t pool;
    std::string pool_path = "./data";
    std::string pool_name = "test.pool.0";
    size_t pool_size = MB(100);

    void SetUp() override 
    {
        comp = Component::load_component(FILESTORE_PATH, Component::filestore_factory);
        assert(comp); // TODO: assert message possible here?

        fact = (IKVStore_factory *)comp->query_interface(IKVStore_factory::iid());

        g_store = fact->create("owner", "name");  // TODO: what does this do?

        cpu_mask_t cpus;
        unsigned core = 1;
        static unsigned core_count = 1;

        for(unsigned core = 0; core < core_count; core++)
        {
            cpus.add_core(core);
        }

       //    Core::Per_core_tasking<Experiment_Put, Component::IKVStore*> exp(cpus, g_store);
        Component::IKVStore::pool_t pool = g_store->create_pool(pool_path, pool_name, pool_size);
    }

    void TearDown() override
    {
        fact->release_ref();  // TODO: what does this do?
     
        const Component::IKVStore::pool_t pool = g_store->open_pool(pool_path, pool_name);
        g_store->delete_pool(pool);
    }
};

TEST_F(PoolTest, PutGet)
{
    const Component::IKVStore::pool_t pool = g_store->open_pool(pool_path, pool_name);

    // randomly generate key and value
    int key_length = 8;
    int value_length = 64;
    const std::string key = Common::random_string(key_length);
    std::string value = Common::random_string(value_length);
    printf("random value: %s, size %lu\n", value.c_str(), value_length);

    printf("generated random key and value\n");
    status_t rc = g_store->put(pool, key, value.c_str(), value_length);

    ASSERT_EQ(rc, S_OK); 

    void * pval;
    size_t pval_len;
    
    rc  = g_store->get(pool, key, pval, pval_len);

    printf("pval = %s, size %lu\n", static_cast<std::string*>(pval), pval_len);

    ASSERT_EQ(rc, S_OK);
    ASSERT_STREQ((const char*)pval, value.c_str());

    free(pval);
}

TEST_F(PoolTest, dummy_test)
{
    ASSERT_EQ(1, 1);
}

int main(int argc, char **argv) 
{
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}

