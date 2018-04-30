/* note: we do not include component source, only the API definition */
#include <gtest/gtest.h>
#include <api/components.h>
#include <api/fabric_itf.h>

#include <memory>

using namespace Component;

namespace {

// The fixture for testing class Foo.
class Fabric_test : public ::testing::Test {

 protected:

  // If the constructor and destructor are not enough for setting up
  // and cleaning up each test, you can define the following methods:
  
  virtual void SetUp() {
    // Code here will be called immediately after the constructor (right
    // before each test).
  }

  virtual void TearDown() {
    // Code here will be called immediately after each test (right
    // before the destructor).
  }
  
  // Objects declared here can be used by all tests in the test case
  static std::shared_ptr<Component::IFabric_factory> _fabric;
};

std::ostream &describe_ep(std::ostream &o_, const IFabric_endpoint &e_)
{
  return o_ << "provider '" << e_.get_provider_name() << "' max_message_size " << e_.max_message_size();
}

TEST_F(Fabric_test, InstantiateServer)
{
  /* create object instance through factory */
  Component::IBase * comp = Component::load_component("libcomanche-fabric.so",
                                                      Component::net_fabric_factory);

  ASSERT_TRUE(comp);
  auto fabric = std::shared_ptr<Component::IFabric_factory>(static_cast<IFabric_factory *>(comp->query_interface(IFabric_factory::iid())));

  {
    auto pep1 = std::shared_ptr<Component::IFabric_endpoint>(fabric->open_endpoint("{}"));
    describe_ep(std::cout << "Endpoint 1 ", *pep1) << "\n";
  }
  {
    auto pep2 = std::shared_ptr<Component::IFabric_endpoint>(fabric->open_endpoint("{}"));
    describe_ep(std::cout << "Endpoint 2 ", *pep2) << "\n";
  }
  fabric->release_ref();
}

TEST_F(Fabric_test, InstantiateServerDual)
{
  /* create object instance through factory */
  Component::IBase * comp = Component::load_component("libcomanche-fabric.so",
                                                      Component::net_fabric_factory);

  ASSERT_TRUE(comp);
  auto fabric = std::shared_ptr<Component::IFabric_factory>(static_cast<IFabric_factory *>(comp->query_interface(IFabric_factory::iid())));

  {
    auto pep1 = std::shared_ptr<Component::IFabric_endpoint>(fabric->open_endpoint("{}"));
    auto pep2 = std::shared_ptr<Component::IFabric_endpoint>(fabric->open_endpoint("{}"));
    describe_ep(std::cout << "Endpoint 1 ", *pep1) << "\n";
    describe_ep(std::cout << "Endpoint 2 ", *pep2) << "\n";
  }
  fabric->release_ref();
}

TEST_F(Fabric_test, InstantiateServerAndClient)
{
  /* create object instance through factory */
  Component::IBase * comp = Component::load_component("libcomanche-fabric.so",
                                                      Component::net_fabric_factory);

  ASSERT_TRUE(comp);
  auto fabric = std::shared_ptr<Component::IFabric_factory>(static_cast<IFabric_factory *>(comp->query_interface(IFabric_factory::iid())));

  {
    auto pep1 = std::shared_ptr<Component::IFabric_endpoint>(fabric->open_endpoint("{}"));
    auto pep2 = std::shared_ptr<Component::IFabric_endpoint>(fabric->open_endpoint("{}"));
    describe_ep(std::cout << "Endpoint 1 ", *pep1) << "\n";
    describe_ep(std::cout << "Endpoint 2 ", *pep2) << "\n";
  }
  fabric->release_ref();
}

TEST_F(Fabric_test, JsonSucceed)
{
  /* create object instance through factory */
  Component::IBase * comp = Component::load_component("libcomanche-fabric.so",
                                                      Component::net_fabric_factory);

  ASSERT_TRUE(comp);
  /* Feed the ednpoint a good JSON spec */
  auto fabric = std::shared_ptr<Component::IFabric_factory>(static_cast<IFabric_factory *>(comp->query_interface(IFabric_factory::iid())));

  try
  {
    auto pep1 = std::shared_ptr<Component::IFabric_endpoint>(fabric->open_endpoint("{ \"tx_attr\" : { \"comp_order\" : [ \"FI_ORDER_STRICT\", \"FI_ORDER_DATA\" ], \"inject_size\" : 16 } }"));
    describe_ep(std::cout << "Endpoint 1 ", *pep1) << "\n";
  }
  catch ( const std::domain_error &e )
  {
    std::cout << "Unexpected domain error: " << e.what() << "\n";
    ASSERT_TRUE(false);
  }
  fabric->release_ref();
}

TEST_F(Fabric_test, JsonParseFail)
{
  /* create object instance through factory */
  Component::IBase * comp = Component::load_component("libcomanche-fabric.so",
                                                      Component::net_fabric_factory);

  ASSERT_TRUE(comp);
  auto fabric = std::shared_ptr<Component::IFabric_factory>(static_cast<IFabric_factory *>(comp->query_interface(IFabric_factory::iid())));

  {
    /* Feed the ednpoint a good JSON spec */
    try
    {
      auto pep1 = std::shared_ptr<Component::IFabric_endpoint>(fabric->open_endpoint("{ \"tx_attr\" : { \"max\" : \"xyz\"} } }"));
      ASSERT_TRUE(false);
    }
    catch ( const std::domain_error &e )
    {
      std::cout << "Expected domain error: " << e.what() << "\n";
      /* Error mesage should mention "parse" */
      ASSERT_TRUE(::strpbrk(e.what(), "parse"));
    }
  }
  fabric->release_ref();
}

TEST_F(Fabric_test, JsonKeyFail)
{
  /* create object instance through factory */
  Component::IBase * comp = Component::load_component("libcomanche-fabric.so",
                                                      Component::net_fabric_factory);

  ASSERT_TRUE(comp);
  auto fabric = std::shared_ptr<Component::IFabric_factory>(static_cast<IFabric_factory *>(comp->query_interface(IFabric_factory::iid())));

  {
    /* Feed the ednpoint a good JSON spec */
    try
    {
      auto pep1 = std::shared_ptr<Component::IFabric_endpoint>(fabric->open_endpoint("{ \"tx_attr\" : { \"maX\" : \"xyz\"} }"));
      ASSERT_TRUE(false);
    }
    catch ( const std::domain_error &e )
    {
      std::cout << "Expected domain error: " << e.what() << "\n";
      /* Error mesage should mention "key", "tx_attr", and "maX" */
      ASSERT_TRUE(::strpbrk(e.what(), "key"));
      ASSERT_TRUE(::strpbrk(e.what(), "tx_attr"));
      ASSERT_TRUE(::strpbrk(e.what(), "maX"));
    }
  }
  fabric->release_ref();
}

} // namespace

int main(int argc, char **argv) {

  ::testing::InitGoogleTest(&argc, argv);
  auto r = RUN_ALL_TESTS();

  return r;
}
