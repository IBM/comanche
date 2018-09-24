#ifndef __TEST_COMPONENT_H__
#define __TEST_COMPONENT_H__

#include <api/components.h>
#include <api/kvstore_itf.h>
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
    Component::IKVStore::memory_handle_t memory_handle = Component::IKVStore::HANDLE_NONE;

    // handles for use
    Component::IKVStore * store;
    Component::IKVStore_factory* factory;
    Component::IBase* comp;
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
                    pool_path = "/mnt/pmem0";
                }
                else if (component_name.compare("nvmestore") == 0) {
                    component_object = "libcomanche-nvmestore.so";
                    component_uuid = Component::nvmestore_factory;
                    pool_path = "/mnt/pmem0";
                    pci_address = "09:00.0";  // IMPORTANT: this is what will show up as the first part of command "$ lspci | grep Non"
                }
                else if (component_name.compare("dawn_client") == 0)
                {
                    DECLARE_STATIC_COMPONENT_UUID(dawn_client_factory, 0xfac66078,0xcb8a,0x4724,0xa454,0xd1,0xd8,0x8d,0xe2,0xdb,0x87);  // TODO: find a better way to register arbitrary components to promote modular use
                    component_uuid = dawn_client_factory;
                    component_object = "libcomanche-dawn-client.so";
                    pool_path = "/mnt/pmem0";
                    uses_direct_memory = true;
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
    
    void load_component()
    {        
        comp = Component::load_component(component_object, component_uuid);

        factory = (Component::IKVStore_factory*)comp->query_interface(Component::IKVStore_factory::iid());

        if (component_name.compare("nvmestore") == 0)
        {
            store = factory->create(owner, owner_param, pci_address);
        } 
        else if (component_name.compare("dawn_client") == 0)
        {
            store = factory->create(debug_level, owner, server_address, device_name);
        }
        else
        {
            store = factory->create(owner, owner_param);
        }
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
