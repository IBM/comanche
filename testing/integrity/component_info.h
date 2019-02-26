#ifndef __TEST_COMPONENT_H__
#define __TEST_COMPONENT_H__

#include <api/components.h>
#include <api/kvindex_itf.h>
#include <api/kvstore_itf.h>
#include <sys/mman.h>
#include <boost/program_options.hpp>

class ComponentInfo
{
public:
    // tunable parameters
    std::string component_name = "filestore";
    std::string component_object = "libcomanche-storefile.so";
    Component::uuid_t component_uuid = Component::filestore_factory;
    std::string owner = "owner";
    std::string owner_param = "name";
    std::string pool_path = "./data";
    std::string pci_address = "";
    std::string server_address = "";
    std::string device_name = "";
    int debug_level = 0;
    bool verbose = true;
    bool component_initialized = false;
    bool              isIndex               = false;
    Component::IKVStore::memory_handle_t memory_handle = Component::IKVStore::HANDLE_NONE;

    // handles for use
    Component::IKVStore * store;
    Component::IKVStore_factory* factory;
    Component::IKVIndex*         index;
    Component::IKVIndex_factory* index_factory;
    bool uses_direct_memory = false;

    ComponentInfo()
    {

    }
    ~ComponentInfo()
    {

    }

    int initialize_component(int argc, char **argv)
    {
        namespace po = boost::program_options; 

        po::options_description desc("Options"); 
        desc.add_options()
        ("help", "Show help")
        ("component", po::value<std::string>(), "Component name to test. Defaults to Filestore.")
        ("component_object", po::value<std::string>(), "Component object filename (*.so)")
        ("path", po::value<std::string>(), "Path for pool location")
        ("pci_address", po::value<std::string>(), "PCI address for component")
        ("server_address", po::value<std::string>(), "Server address with port")
        ("device_name", po::value<std::string>(), "Device name to use with component")
        ("debug_level", po::value<int>(), "Debug level")
        ("gtest_filter", po::value<std::string>(), "Run specific Google Tests")
        ;

        try 
        {
            po::variables_map vm; 
            po::store(po::parse_command_line(argc, argv, desc),  vm);

            if(vm.count("help")) {
              std::cout << desc;

              return 0;
            }

            // handle defaults if only setting component option
            if(vm.count("component")) 
            {
                component_name = vm["component"].as<std::string>();
               
                if(component_name.compare("filestore") == 0) 
                {
                    component_object = "libcomanche-storefile.so";
                    component_uuid = Component::filestore_factory;
                }
                else if (component_name.compare("pmstore") == 0) {
                    component_object = "libcomanche-pmstore.so";
                    component_uuid = Component::pmstore_factory;
                    pool_path = "/mnt/pmem0/";
                }
                else if (component_name.compare("nvmestore") == 0) {
                    component_object = "libcomanche-nvmestore.so";
                    component_uuid = Component::nvmestore_factory;
                    pool_path = "/mnt/pmem0/";
                    pci_address = "09:00.0";  // IMPORTANT: this is what will show up as the first part of command "$ lspci | grep Non"
                }
                else if (component_name.compare("dawn") == 0)
                {
                    DECLARE_STATIC_COMPONENT_UUID(dawn_factory, 0xfac66078,0xcb8a,0x4724,0xa454,0xd1,0xd8,0x8d,0xe2,0xdb,0x87);  // TODO: find a better way to register arbitrary components to promote modular use
                    component_uuid = dawn_factory;
                    component_object = "libcomanche-dawn-client.so";
                    pool_path = "/mnt/pmem0/";
                    uses_direct_memory = true;
                }
                else if (component_name.compare("hstore") == 0)
                {
                    component_uuid = Component::hstore_factory;
                    component_object = "libcomanche-hstore.so";
                    pool_path = "/mnt/pmem0/pool/";
                }
                else if (component_name.compare("mapstore") == 0)
                {
                    component_uuid = Component::mapstore_factory;
                    component_object = "libcomanche-storemap.so";
                    pool_path = "/mnt/pmem0/";
                }
                else if (component_name.compare("rbtreeindex") == 0) {
                  component_object = "libcomanche-indexrbtree.so";
                  component_uuid   = Component::rbtreeindex_factory;
                  isIndex          = true;
                }
                else
                {
                    printf("UNHANDLED COMPONENT\n");
                    return 1;
                }
            }

            // handle straightforward program options - defaults set above
            component_object = vm.count("component_object") > 0 ? vm["componnet_object"].as<std::string>() : component_object;
            pool_path = vm.count("path") > 0 ? vm["path"].as<std::string>() : pool_path;
            pci_address = vm.count("pci_address") > 0 ? vm["pci_address"].as<std::string>() : pci_address;
            server_address = vm.count("server_address") > 0 ? vm["server_address"].as<std::string>() : server_address;
            device_name = vm.count("device_name") > 0 ? vm["device_name"].as<std::string>() : device_name;
            debug_level = vm.count("debug_level") > 0 ? vm["debug_level"].as<int>() : debug_level;
        }
        catch (const po::error &ex)
        {
            std::cerr << ex.what() << '\n';
        }
        
        if (verbose)
        {
            print_component_info();            
        }

        return 0;
    }
    
    static std::string quote(const std::string &s)
    {
      return "\"" + s + "\"";
    }

    static std::string json_map(const std::string &key, const std::string &value)
    {
      return quote(key) + ": " + value;
    }

    void load_component()
    {        
        Component::IBase* comp = Component::load_component(component_object, component_uuid);

        if (!isIndex)
          factory = (Component::IKVStore_factory*) comp->query_interface(
              Component::IKVStore_factory::iid());
        else {
          index_factory = (Component::IKVIndex_factory*) comp->query_interface(
              Component::IKVIndex_factory::iid());
        }

        if (component_name.compare("nvmestore") == 0)
        {
            store = factory->create(owner, owner_param, pci_address);
        } 
        else if (component_name.compare("dawn") == 0)
        {
            store = factory->create(debug_level, owner, server_address, device_name);
        }
        else if (component_name.compare("hstore") == 0)
        {
          unsigned core = 0;  // always use core 0 for tests
          unsigned long dax_size = 0x8000000000;
          unsigned region_id = 0;
          std::ostringstream addr;
          addr << std::showbase << std::hex << 0x7000000000 + dax_size * core;
          std::ostringstream device_map;
          device_map <<
            "[ " 
              " { "
                + json_map("region_id", std::to_string(region_id))
                /* actual device name is <idevice_name>.<core>, e.g. /dev/dax0.2 */
                + ", " + json_map("path", quote(device_name + "." + std::to_string(core)))
                + ", " + json_map("addr", quote(addr.str()))
                + " }"
            " ]";
          store = factory->create(debug_level, "name", owner, device_map.str());

        }
        else if (!isIndex) {
          store = factory->create(owner, owner_param);
        }
        else {
          store = NULL;
          index = index_factory->create(owner, owner_param);
        }
    } 

    int* setup_direct_memory_for_size(size_t data_size)
    {
        int* aligned_memory = nullptr;

        if (uses_direct_memory)
        {
           data_size += data_size % 64;  // align
           aligned_memory = (int*)aligned_alloc(MiB(2), data_size);
           madvise(aligned_memory, data_size, MADV_HUGEPAGE);

           memory_handle = store->register_direct_memory(aligned_memory, data_size);
        }

        return aligned_memory;  // pointer to aligned memory region
    }

    void print_component_info()
    {
        // print component setup info
        print_value("component_name", component_name);
        print_value("component_object", component_object);
        print_value("pool_path", pool_path);
        print_value("pci_address", pci_address);
        print_value("server_address", server_address);
        print_value("device_name", device_name);
        print_value("debug_level", debug_level);
    }

    template <typename T>
    void print_value(std::string label, T value)
    {
        std::cout << label << " = " << value << std::endl;
    }
};

#endif
