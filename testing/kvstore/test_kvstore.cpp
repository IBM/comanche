#include <gtest/gtest.h>

/* note: we do not include component source, only the API definition */
#include <api/components.h>
#include <api/kvstore_itf.h>
#include <boost/program_options.hpp>
#include <common/utils.h>
#include <common/str_utils.h>
#include <core/task.h>
//#include <chrono>
#include <iostream>

#define DEFAULT_COMPONENT "filestore"
#define FILESTORE_PATH "libcomanche-storefile.so"

using namespace Component;

std::string component_path = FILESTORE_PATH;
Component::uuid_t component_uuid = Component::filestore_factory;
std::string pool_path = "./data";

class PoolTest: public::testing::Test
{
protected:
    Component::IKVStore * g_store;
    Component::IBase * comp;
    IKVStore_factory * fact;

    Component::IKVStore::pool_t pool;

    std::string pool_name = "test.pool.0";
    size_t pool_size = MB(100);

    void SetUp() override 
    {
        comp = Component::load_component(component_path, component_uuid);

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

        Component::IKVStore::pool_t pool = g_store->create_pool(pool_path, pool_name, pool_size);
    }

    void TearDown() override
    {
        fact->release_ref();  // TODO: what does this do?
     
        const Component::IKVStore::pool_t pool = g_store->open_pool(pool_path, pool_name);
        g_store->delete_pool(pool);
    }
};

TEST_F(PoolTest, PutGet_RandomKVP)
{
    const Component::IKVStore::pool_t pool = g_store->open_pool(pool_path, pool_name);

    // randomly generate key and value
    int key_length = 8;
    int value_length = 64;
    const std::string key = Common::random_string(key_length);
    std::string value = Common::random_string(value_length);
    printf("random value: %s, size %d\n", value.c_str(), (int)value_length);

    printf("generated random key and value\n");
    status_t rc = g_store->put(pool, key, value.c_str(), value_length);

    ASSERT_EQ(rc, S_OK); 

    void * pval;
    size_t pval_len;
    
    rc  = g_store->get(pool, key, pval, pval_len);

    printf("pval = %s, size %d\n", static_cast<char*>(pval), (int)pval_len);

    ASSERT_EQ(rc, S_OK);
    ASSERT_STREQ((const char*)pval, value.c_str());

    free(pval);
}

TEST_F(PoolTest, dummy_test)
{
    ASSERT_EQ(1, 1);
}


struct {
  std::string test;
  std::string component;
  unsigned cores;
  unsigned time_secs;
} Options;

int main(int argc, char **argv) 
{
    namespace po = boost::program_options; 

    po::options_description desc("Options"); 
    desc.add_options()
    ("help", "Show help")
    ("test", po::value<std::string>(), "Test name <all|Put|Get>")
    ("component", po::value<std::string>(), "Implementation selection <pmstore|nvmestore|filestore>")
    ("cores", po::value<int>(), "Number of threads/cores")
    ("time", po::value<int>(), "Duration to run in seconds")
    ;

    try 
    {
        po::variables_map vm; 
        po::store(po::parse_command_line(argc, argv, desc),  vm);

        if(vm.count("help")) {
          std::cout << desc;

          return 0;
        }

        Options.test = vm.count("test") > 0 ? vm["test"].as<std::string>() : "all";

        if(vm.count("component")) 
        {
            Options.component = vm["component"].as<std::string>();
           
            printf("checking component arg\n");
            if(Options.component.compare("filestore") == 0) 
            {
                printf("USE FILESTORE\n");                   
                
                component_path = FILESTORE_PATH;
                component_uuid = Component::filestore_factory;
            }
            else if (Options.component.compare("pmstore") == 0) {
                printf("USE PMSTORE\n");

                component_path = "libcomanche-pmstore.so";
                component_uuid = Component::pmstore_factory;
                
                pool_path = "/mnt/pmem0";
            }
            else if (Options.component.compare("nvmestore") == 0) {
                printf("USE NVMESTORE\n");

                component_path = "libcomanche-nvmestore.so";
                component_uuid = Component::nvmestore_factory;
                
                pool_path = "/mnt/pmem0";
            }
            else
            {
                printf("UNHANDLED COMPONENT\n");
                return 1;
            }
        }
        else
        {
            Options.component = DEFAULT_COMPONENT;
        }
    }
    catch (const po::error &ex)
    {
        std::cerr << ex.what() << '\n';
    }

    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}

