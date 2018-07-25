/* note: we do not include component source, only the API definition */
#include <api/components.h>
#include <api/kvstore_itf.h>
#include <boost/program_options.hpp>
#include <common/utils.h>
#include <common/str_utils.h>
#include <core/task.h>
#include <iostream>
#include <gtest/gtest.h>

#define DEFAULT_COMPONENT "filestore"
#define FILESTORE_PATH "libcomanche-storefile.so"

using namespace Component;

std::string component_path = FILESTORE_PATH;
Component::uuid_t component_uuid = Component::filestore_factory;
std::string pci_address = "";  // optional parameter
std::string pool_path = "./data";

class PoolTest: public::testing::Test
{
protected:
    Component::IKVStore * _g_store;
    IKVStore_factory * _fact;
    Component::IKVStore::pool_t _pool;
    std::string _pool_name = "test.pool.0";
    size_t _pool_size = MB(100);

    void SetUp() override 
    {
        // TODO: make sure pool_path directory exists
        Component::IBase * comp;
        comp = Component::load_component(component_path, component_uuid);

        ASSERT_TRUE(comp > 0);

        _fact = (IKVStore_factory *)comp->query_interface(IKVStore_factory::iid());

        if (pci_address.compare("") == 0)
        {            
            _g_store = _fact->create("owner", "name"); 
        }
        else
        {
            _g_store = _fact->create("owner", "name", pci_address);
        }

        _pool = _g_store->create_pool(pool_path, _pool_name, _pool_size);

        ASSERT_TRUE(_pool > 0);  // just fail here if pool creation didn't work
    }

    void TearDown() override
    {
        _fact->release_ref();  // TODO: figure out why we need this
     
        if (_pool <= 0)
        {
           _pool = _g_store->open_pool(pool_path, _pool_name);
        }
        
        ASSERT_TRUE(_pool > 0);
        _g_store->delete_pool(_pool);
    }
};

TEST_F(PoolTest, OpenPool)
{
    const Component::IKVStore::pool_t pool = _g_store->open_pool(pool_path, _pool_name);
    
    ASSERT_TRUE(pool > 0);
}

TEST_F(PoolTest, PutGet_RandomKVP)
{
    // randomly generate key and value
    const int key_length = 8;
    const int value_length = 64;
    const std::string key = Common::random_string(key_length);

    std::string value = Common::random_string(value_length);
    status_t rc = _g_store->put(_pool, key, value.c_str(), value_length);

    ASSERT_EQ(rc, S_OK); 

    void * pval;
    size_t pval_len;
    
    rc  = _g_store->get(_pool, key, pval, pval_len);
    
    ASSERT_EQ(rc, S_OK);
    ASSERT_STREQ((const char*)pval, value.c_str());

    free(pval);
}

TEST_F(PoolTest, Get_NoValidKey)
{
    // randomly generate key to look up (we shouldn't find it)
    const int key_length = 8;
    const std::string key = Common::random_string(key_length);

    void * pval = NULL;  // initialize so we can check if value has been changed
    size_t pval_len;
    
    status_t rc  = _g_store->get(_pool, key, pval, pval_len);
    ASSERT_EQ((int)rc, (int)IKVStore::E_KEY_NOT_FOUND);  // expect failure code here

    if(pval != NULL)
    {
        free(pval);
    }
}

TEST_F(PoolTest, Put_DuplicateKey)
{
    // randomly generate key and value
    const int key_length = 8;
    const int value_length = 64;

    const std::string key = Common::random_string(key_length);
    std::string value = Common::random_string(value_length);

    status_t rc = _g_store->put(_pool, key, value.c_str(), value_length);

    ASSERT_EQ(rc, S_OK); 

    // now try to add another value to that key
    rc = _g_store->put(_pool, key, value.c_str(), value_length);

    ASSERT_EQ(rc, IKVStore::E_KEY_EXISTS);
}


TEST_F(PoolTest, Put_Erase)
{
    // randomly generate key and value
    const int key_length = 8;
    const int value_length = 64;
    const std::string key = Common::random_string(key_length);
    const std::string value = Common::random_string(value_length);

    status_t rc = _g_store->put(_pool, key, value.c_str(), value_length);

    ASSERT_EQ(rc, S_OK); 

    // now delete key and try to get it again post-deletion
    void * pval = NULL;  // initialize so we can check if value has changed
    size_t pval_len;

    rc = _g_store->erase(_pool, key);
 
    rc  = _g_store->get(_pool, key, pval, pval_len);

    ASSERT_EQ(rc, IKVStore::E_KEY_NOT_FOUND);

    if (pval != NULL)
    {
        free(pval);
    }
}

TEST_F(PoolTest, DISABLED_Count_Changes)
{
    const int key_length = 8;
    const int value_length = 64;

    size_t size;
    
    // check size on creation (0)
    size = _g_store->count(_pool);
    ASSERT_EQ(size, 0);
    
    // put random key and value, check size again (1)
    const std::string key_1 = Common::random_string(key_length);
    std::string value = Common::random_string(value_length);

    status_t rc = _g_store->put(_pool, key_1, value.c_str(), value_length);  
    ASSERT_EQ(rc, S_OK);
    ASSERT_EQ(_g_store->count(_pool), 1);

    // put another random key, check size again (2) 
    const std::string key_2 = Common::random_string(key_length);
    value = Common::random_string(value_length);

    rc = _g_store->put(_pool, key_2, value.c_str(), value_length);  

    ASSERT_EQ(_g_store->count(_pool), 2);

    // check size on delete (1)
    rc = _g_store->erase(_pool, key_1);
    ASSERT_EQ(rc, S_OK);
    ASSERT_EQ(_g_store->count(_pool), 1);

    // check size on delete again (0)
    rc = _g_store->erase(_pool, key_2);
    ASSERT_EQ(rc, S_OK);
    ASSERT_EQ(_g_store->count(_pool), 0);
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
                pci_address = "09:00.0";  // IMPORTANT: this is what will show up as the first part of command "$ lspci | grep Non"
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

