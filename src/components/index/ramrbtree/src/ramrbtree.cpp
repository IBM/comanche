#include "ramrbtree.h"
#include <stdlib.h>
#include <set>

#define SINGLE_THREADED

using namespace Component;
using namespace std;

static set<string> m_index;

RamRBTree::RamRBTree(const std::string& owner, const std::string& name) {}

RamRBTree::RamRBTree() {}

RamRBTree::~RamRBTree() {}

void RamRBTree::insert(const string& key)
{
  if (!m_index.insert(key).second) {
    throw(API_exception("insert index failed"));
  }
}

void RamRBTree::erase(const std::string& key) { m_index.erase(key); }

void RamRBTree::clear() { m_index.clear(); }

/**
 * Get next item.  Throw std::out_of_range for out of bounds
 *
 * @param position Position counting from zero
 *
 * @return Key
 */
string RamRBTree::get(offset_t position) const
{
  if (position >= m_index.size()) {
    throw out_of_range("Position out of range");
  }

  auto it = m_index.begin();
  advance(it, position);
  return *it;
}

size_t RamRBTree::count() const { return m_index.size(); }

/**
 * Factory entry point
 *
 */
extern "C" void* factory_createInstance(Component::uuid_t& component_id)
{
  if (component_id == RamRBTree_factory::component_id()) {
    return static_cast<void*>(new RamRBTree_factory());
  }
  else
    return NULL;
}
