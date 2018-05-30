/*
   Copyright [2017] [IBM Corporation]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

/* 
 * Authors: 
 * 
 * Daniel G. Waddington (daniel.waddington@ibm.com)
 *
 */

/* 
   Based on OSS code from https://github.com/wichtounet/btrees 
   by Baptiste Wicht (baptiste.wicht@gmail.com) 
*/

#ifndef CONC_AVL_TREE
#define CONC_AVL_TREE

#include <functional>
#include <mutex>

#include "hazard.h"  // ABA protection

namespace Concurrent
{
namespace AVL
{
typedef std::lock_guard<std::mutex> scoped_lock;

static int SpinCount = 100;

static long begin_change(long ovl)
{
  return ovl | 1;
}
static long end_change(long ovl)
{
  return (ovl | 3) + 1;
}

static bool is_shrinking(long ovl)
{
  return (ovl & 1) != 0;
}
static bool is_unlinked(long ovl)
{
  return (ovl & 2) != 0;
}
static bool is_shrinking_or_unlinked(long ovl)
{
  return (ovl & 3) != 0L;
}

static long UnlinkedOVL = 2;

/* Conditions on nodes */
static const int UnlinkRequired    = -1;
static const int RebalanceRequired = -2;
static const int NothingRequired   = -3;

enum Function { UpdateIfPresent, UpdateIfAbsent };

template <typename V>
struct Base_node {
  int        height;
  key_t      key;
  V          value;
  long       version;
  bool       is_value;
  Base_node* parent;
  Base_node* left;
  Base_node* right;
  std::mutex lock;

  Base_node* child(int direction)
  {
    return (direction > 0) ? right : left;
  }

  //Should only be called with lock on Node
  void set_child(int direction, Base_node* child)
  {
    if (direction > 0) {
      right = child;
    }
    else if (direction < 0) {
      left = child;
    }
  }
};

enum Result { FOUND, NOT_FOUND, RETRY };

//typedef Base_node<void*> Node;

template <typename Value_T, typename Key_T = uint64_t, int Threads = 256>
class Tree
{
  typedef Base_node<Value_T> Node;

 public:
  Tree();
  ~Tree();

  /** 
   * Determine if key exists
   * 
   * @param key 
   * 
   * @return 
   */
  bool contains(Key_T key);

  /** 
   * Add new key/value pair
   * 
   * @param key 
   * @param value 
   * 
   * @return 
   */
  bool add(Key_T key, Value_T value);

  /** 
   * Remove a key
   * 
   * @param key 
   * 
   * @return 
   */
  bool remove(Key_T key);

  /** 
   * Traverse and apply function - NOT THREAD CONCURRENT
   * 
   * @param func 
   */
  void apply_topdown(std::function<void(Key_T key, Value_T value)> func);

 private:
  /* Allocate new nodes */
  Node* create_node(Key_T key);
  Node* create_node(int height, Key_T key, Value_T value, long version, bool is_value, Node* parent, Node* left,
                    Node* right);

  //Search
  Result attempt_get(Key_T key, Node* node, int dir, long nodeV);

  /* Update stuff  */
  Result update_under_root(Key_T key, Value_T value, Function func, bool expected, bool new_value, Node* holder);

  bool attempt_insert_into_empty(Key_T key, Value_T value, bool is_value, Node* holder);

  Result attempt_update(Key_T key, Value_T value, Function func, bool expected, bool new_value, Node* parent,
                        Node* node, long nodeOVL);

  Result attempt_node_update(Function func, bool expected, bool new_value, Node* parent, Node* node);

  bool attempt_unlink_nl(Node* parent, Node* node);

  /* To wait during shrinking  */
  void wait_until_not_changing(Node* node);

  /* Rebalancing stuff  */
  void fix_height_and_rebalance(Node* node);
  Node* rebalance_nl(Node* nParent, Node* n);
  Node* rebalance_to_right_nl(Node* nParent, Node* n, Node* nL, int hR0);
  Node* rebalance_to_left_nl(Node* nParent, Node* n, Node* nR, int hL0);

  /* Rotation stuff */
  Node* rotate_left_over_right_nl(Node* nParent, Node* n, int hL, Node* nR, Node* nRL, int hRR, int hRLR);

  Node* rotate_right_over_left_nl(Node* nParent, Node* n, Node* nL, int hR, int hLL, Node* nLR, int hLRL);
  Node* rotate_left_nl(Node* nParent, Node* n, int hL, Node* nR, Node* nRL, int hRL, int hRR);
  Node* rotate_right_nl(Node* nParent, Node* n, Node* nL, int hR, int hLL, Node* nLR, int hLR);

  void publish(Node* ref);
  void release_all();

  Node* _root_holder;

  Hazard_manager<Node, Threads, 6> hazard;

  unsigned int Current[Threads];

  static Node* fix_height_nl(Node* n);
  static int height(Node* node);
  static int node_condition(Node* node);
};


template <typename Value_T, typename Key_T, int Threads>
Tree<Value_T, Key_T, Threads>::Tree()
{
  _root_holder = create_node(std::numeric_limits<int>::min());

  for (unsigned int i = 0; i < Threads; ++i) {
    Current[i] = 0;
  }
}

template <typename Value_T, typename Key_T, int Threads>
Tree<Value_T, Key_T, Threads>::~Tree()
{
  hazard.release_node(_root_holder);
}

template <typename Value_T, typename Key_T, int Threads>
void Tree<Value_T, Key_T, Threads>::apply_topdown(std::function<void(Key_T key, Value_T value)> func)
{
  std::vector<Node*> stack;
  stack.push_back(_root_holder);

  while (!stack.empty()) {
    Node* node = stack.back();
    stack.pop_back();

    if (node->is_value) func(node->key, node->value);

    Node* l = node->left;
    if (l) stack.push_back(l);

    Node* r = node->right;
    if (r) stack.push_back(r);
  }
}


template <typename Value_T, typename Key_T, int Threads>
void Tree<Value_T, Key_T, Threads>::publish(Node* ref)
{
  hazard.publish(ref, Current[__hazard_thread_num]);
  ++Current[__hazard_thread_num];
}

template <typename Value_T, typename Key_T, int Threads>
void Tree<Value_T, Key_T, Threads>::release_all()
{
  for (unsigned int i = 0; i < Current[__hazard_thread_num]; ++i) {
    hazard.release(i);
  }

  Current[__hazard_thread_num] = 0;
}

template <typename Value_T, typename Key_T, int Threads>
typename Tree<Value_T, Key_T, Threads>::Node* Tree<Value_T, Key_T, Threads>::create_node(Key_T key)
{
  Value_T v;
  return create_node(1, key, v, 0L, false, nullptr, nullptr, nullptr);
}

template <typename Value_T, typename Key_T, int Threads>
typename Tree<Value_T, Key_T, Threads>::Node* Tree<Value_T, Key_T, Threads>::create_node(int height, Key_T key,
                                                                                         Value_T value, long version,
                                                                                         bool is_value, Node* parent,
                                                                                         Node* left, Node* right)
{
  Node* node = hazard.get_free_node();

  node->value    = value;
  node->height   = height;
  node->key      = key;
  node->version  = version;
  node->is_value = is_value;
  node->parent   = parent;
  node->left     = left;
  node->right    = right;

  return node;
}

template <typename Value_T, typename Key_T, int Threads>
bool Tree<Value_T, Key_T, Threads>::contains(Key_T key)
{
  while (true) {
    Node* right = _root_holder->right;

    if (!right) {
      return false;
    }
    else {
      int rightCmp = key - right->key;
      if (rightCmp == 0) {
        return right->is_value;
      }

      int ovl = right->version;
      if (is_shrinking_or_unlinked(ovl)) {
        wait_until_not_changinging(right);
      }
      else if (right == _root_holder->right) {
        Result vo = attempt_get(key, right, rightCmp, ovl);
        if (vo != RETRY) {
          return vo == FOUND;
        }
      }
    }
  }
}

template <typename Value_T, typename Key_T, int Threads>
Result Tree<Value_T, Key_T, Threads>::attempt_get(Key_T key, Node* node, int dir, long nodeV)
{
  while (true) {
    Node* child = node->child(dir);

    if (!child) {
      if (node->version != nodeV) {
        return RETRY;
      }

      return NOT_FOUND;
    }
    else {
      int childCmp = key - child->key;
      if (childCmp == 0) {
        return child->is_value ? FOUND : NOT_FOUND;  //Verify that it's a value node
      }

      long childOVL = child->version;
      if (is_shrinking_or_unlinked(childOVL)) {
        wait_until_not_changinganging(child);

        if (node->version != nodeV) {
          return RETRY;
        }
      }
      else if (child != node->child(dir)) {
        if (node->version != nodeV) {
          return RETRY;
        }
      }
      else {
        if (node->version != nodeV) {
          return RETRY;
        }

        Result result = attempt_get(key, child, childCmp, childOVL);
        if (result != RETRY) {
          return result;
        }
      }
    }
  }
}

inline bool should_update(Function func, bool prev, bool /* expected*/)
{
  return func == UpdateIfAbsent ? !prev : prev;
}

inline Result update_result(Function func, bool /* prev*/)
{
  return func == UpdateIfAbsent ? NOT_FOUND : FOUND;
}

inline Result no_update_result(Function func, bool /* prev*/)
{
  return func == UpdateIfAbsent ? FOUND : NOT_FOUND;
}

template <typename Value_T, typename Key_T, int Threads>
bool Tree<Value_T, Key_T, Threads>::add(Key_T key, Value_T value)
{
  return update_under_root(key, value, UpdateIfAbsent, false, true, _root_holder) == NOT_FOUND;
}

template <typename Value_T, typename Key_T, int Threads>
bool Tree<Value_T, Key_T, Threads>::remove(Key_T key)
{
  Value_T v;
  return update_under_root(key, v, UpdateIfPresent, true, false, _root_holder) == FOUND;
}

template <typename Value_T, typename Key_T, int Threads>
Result Tree<Value_T, Key_T, Threads>::update_under_root(Key_T key, Value_T value, Function func, bool expected,
                                                        bool new_value, Node* holder)
{
  while (true) {
    Node* right = holder->right;

    if (!right) {
      if (!should_update(func, false, expected)) {
        return no_update_result(func, false);
      }

      if (!new_value || attempt_insert_into_empty(key, value, new_value, holder)) {
        return update_result(func, false);
      }
    }
    else {
      long ovl = right->version;

      if (is_shrinking_or_unlinked(ovl)) {
        wait_until_not_changing(right);
      }
      else if (right == holder->right) {
        Result vo = attempt_update(key, value, func, expected, new_value, holder, right, ovl);
        if (vo != RETRY) {
          return vo;
        }
      }
    }
  }
}

template <typename Value_T, typename Key_T, int Threads>
bool Tree<Value_T, Key_T, Threads>::attempt_insert_into_empty(Key_T key, Value_T value, bool is_value, Node* holder)
{
  publish(holder);
  scoped_lock lock(holder->lock);

  if (!holder->right) {
    holder->right  = create_node(1, key, value, 0, is_value, holder, nullptr, nullptr);
    holder->height = 2;
    release_all();
    return true;
  }
  else {
    release_all();
    return false;
  }
}

template <typename Value_T, typename Key_T, int Threads>
Result Tree<Value_T, Key_T, Threads>::attempt_update(Key_T key, Value_T value, Function func, bool expected,
                                                     bool new_value, Node* parent, Node* node, long nodeOVL)
{
  int cmp = key - node->key;
  if (cmp == 0) {
    return attempt_node_update(func, expected, new_value, parent, node);
  }

  while (true) {
    Node* child = node->child(cmp);

    if (node->version != nodeOVL) {
      return RETRY;
    }

    if (!child) {
      if (!new_value) {
        return NOT_FOUND;
      }
      else {
        bool  success;
        Node* damaged;

        {
          publish(node);
          scoped_lock lock(node->lock);

          if (node->version != nodeOVL) {
            release_all();
            return RETRY;
          }

          if (node->child(cmp)) {
            success = false;
            damaged = nullptr;
          }
          else {
            if (!should_update(func, false, expected)) {
              release_all();
              return no_update_result(func, false);
            }

            Node* newChild = create_node(1, key, value, 0, true, node, nullptr, nullptr);
            node->set_child(cmp, newChild);

            success = true;
            damaged = fix_height_nl(node);
          }

          release_all();
        }

        if (success) {
          fix_height_and_rebalance(damaged);
          return update_result(func, false);
        }
      }
    }
    else {
      long childOVL = child->version;

      if (is_shrinking_or_unlinked(childOVL)) {
        wait_until_not_changing(child);
      }
      else if (child != node->child(cmp)) {
        //RETRY
      }
      else {
        if (node->version != nodeOVL) {
          return RETRY;
        }

        Result vo = attempt_update(key, value, func, expected, new_value, node, child, childOVL);
        if (vo != RETRY) {
          return vo;
        }
      }
    }
  }
}

template <typename Value_T, typename Key_T, int Threads>
Result Tree<Value_T, Key_T, Threads>::attempt_node_update(Function func, bool expected, bool new_value, Node* parent,
                                                          Node* node)
{
  if (!new_value) {
    if (!node->is_value) {
      return NOT_FOUND;
    }
  }

  if (!new_value && (!node->left || !node->right)) {
    bool  prev;
    Node* damaged;

    {
      publish(parent);
      scoped_lock parentLock(parent->lock);

      if (is_unlinked(parent->version) || node->parent != parent) {
        release_all();
        return RETRY;
      }

      {
        publish(node);
        scoped_lock lock(node->lock);

        prev = node->is_value;

        if (!should_update(func, prev, expected)) {
          release_all();
          return no_update_result(func, prev);
        }

        if (!prev) {
          release_all();
          return update_result(func, prev);
        }

        if (!attempt_unlink_nl(parent, node)) {
          release_all();
          return RETRY;
        }
      }

      release_all();

      damaged = fix_height_nl(parent);
    }

    fix_height_and_rebalance(damaged);

    return update_result(func, prev);
  }
  else {
    publish(node);
    scoped_lock lock(node->lock);

    if (is_unlinked(node->version)) {
      release_all();
      return RETRY;
    }

    bool prev = node->is_value;
    if (!should_update(func, prev, expected)) {
      release_all();
      return no_update_result(func, prev);
    }

    if (!new_value && (!node->left || !node->right)) {
      release_all();
      return RETRY;
    }

    node->is_value = new_value;

    release_all();
    return update_result(func, prev);
  }
}

template <typename Value_T, typename Key_T, int Threads>
void Tree<Value_T, Key_T, Threads>::wait_until_not_changing(Node* node)
{
  long version = node->version;

  if (is_shrinking(version)) {
    for (int i = 0; i < SpinCount; ++i) {
      if (version != node->version) {
        return;
      }
    }

    node->lock.lock();
    node->lock.unlock();
  }
}

template <typename Value_T, typename Key_T, int Threads>
bool Tree<Value_T, Key_T, Threads>::attempt_unlink_nl(Node* parent, Node* node)
{
  Node* parentL = parent->left;
  Node* parentR = parent->right;

  if (parentL != node && parentR != node) {
    return false;
  }

  Node* left  = node->left;
  Node* right = node->right;

  if (left && right) {
    return false;
  }

  Node* splice = left ? left : right;

  if (parentL == node) {
    parent->left = splice;
  }
  else {
    parent->right = splice;
  }

  if (splice) {
    splice->parent = parent;
  }

  node->version  = UnlinkedOVL;
  node->is_value = false;

  hazard.release_node(node);

  return true;
}

template <typename Value_T, typename Key_T, int Threads>
int Tree<Value_T, Key_T, Threads>::height(Node* node)
{
  return !node ? 0 : node->height;
}

template <typename Value_T, typename Key_T, int Threads>
int Tree<Value_T, Key_T, Threads>::node_condition(Node* node)
{
  Node* nL = node->left;
  Node* nR = node->right;

  // unlink is required
  if ((!nL || !nR) && !node->is_value) {
    return UnlinkRequired;
  }

  int hN     = node->height;
  int hL0    = height(nL);
  int hR0    = height(nR);
  int hNRepl = 1 + std::max(hL0, hR0);
  int bal    = hL0 - hR0;

  // rebalance is required ?
  if (bal < -1 || bal > 1) {
    return RebalanceRequired;
  }

  return hN != hNRepl ? hNRepl : NothingRequired;
}

template <typename Value_T, typename Key_T, int Threads>
void Tree<Value_T, Key_T, Threads>::fix_height_and_rebalance(Node* node)
{
  while (node && node->parent) {
    int condition = node_condition(node);
    if (condition == NothingRequired || is_unlinked(node->version)) {
      return;
    }

    if (condition != UnlinkRequired && condition != RebalanceRequired) {
      publish(node);
      scoped_lock lock(node->lock);

      node = fix_height_nl(node);

      release_all();
    }
    else {
      Node* nParent = node->parent;
      publish(nParent);
      scoped_lock lock(nParent->lock);

      if (!is_unlinked(nParent->version) && node->parent == nParent) {
        publish(node);
        scoped_lock nodeLock(node->lock);

        node = rebalance_nl(nParent, node);
      }

      release_all();
    }
  }
}

template <typename Value_T, typename Key_T, int Threads>
typename Tree<Value_T, Key_T, Threads>::Node* Tree<Value_T, Key_T, Threads>::fix_height_nl(Node* node)
{
  int c = node_condition(node);

  switch (c) {
    case RebalanceRequired:
    case UnlinkRequired:
      return node;
    case NothingRequired:
      return nullptr;
    default:
      node->height = c;
      return node->parent;
  }
}

template <typename Value_T, typename Key_T, int Threads>
typename Tree<Value_T, Key_T, Threads>::Node* Tree<Value_T, Key_T, Threads>::rebalance_nl(Node* nParent, Node* n)
{
  Node* nL = n->left;
  Node* nR = n->right;

  if ((!nL || !nR) && !n->is_value) {
    if (attempt_unlink_nl(nParent, n)) {
      return fix_height_nl(nParent);
    }
    else {
      return n;
    }
  }

  int hN     = n->height;
  int hL0    = height(nL);
  int hR0    = height(nR);
  int hNRepl = 1 + std::max(hL0, hR0);
  int bal    = hL0 - hR0;

  if (bal > 1) {
    return rebalance_to_right_nl(nParent, n, nL, hR0);
  }
  else if (bal < -1) {
    return rebalance_to_left_nl(nParent, n, nR, hL0);
  }
  else if (hNRepl != hN) {
    n->height = hNRepl;

    return fix_height_nl(nParent);
  }
  else {
    return nullptr;
  }
}

template <typename Value_T, typename Key_T, int Threads>
typename Tree<Value_T, Key_T, Threads>::Node* Tree<Value_T, Key_T, Threads>::rebalance_to_right_nl(Node* nParent,
                                                                                                   Node* n, Node* nL,
                                                                                                   int hR0)
{
  publish(nL);
  scoped_lock lock(nL->lock);

  int hL = nL->height;
  if (hL - hR0 <= 1) {
    return n;
  }
  else {
    publish(nL->right);
    Node* nLR = nL->right;

    int hLL0 = height(nL->left);
    int hLR0 = height(nLR);

    if (hLL0 > hLR0) {
      return rotate_right_nl(nParent, n, nL, hR0, hLL0, nLR, hLR0);
    }
    else {
      {
        if (reinterpret_cast<long>(&nLR->lock) == 0x30) {
          return n;
        }
        scoped_lock subLock(nLR->lock);

        int hLR = nLR->height;
        if (hLL0 >= hLR) {
          return rotate_right_nl(nParent, n, nL, hR0, hLL0, nLR, hLR);
        }
        else {
          int hLRL = height(nLR->left);
          int b    = hLL0 - hLRL;
          if (b >= -1 && b <= 1) {
            return rotate_right_over_left_nl(nParent, n, nL, hR0, hLL0, nLR, hLRL);
          }
        }
      }

      return rebalance_to_left_nl(n, nL, nLR, hLL0);
    }
  }
}

template <typename Value_T, typename Key_T, int Threads>
typename Tree<Value_T, Key_T, Threads>::Node* Tree<Value_T, Key_T, Threads>::rebalance_to_left_nl(Node* nParent,
                                                                                                  Node* n, Node* nR,
                                                                                                  int hL0)
{
  publish(nR);
  scoped_lock lock(nR->lock);

  int hR = nR->height;
  if (hL0 - hR >= -1) {
    return n;
  }
  else {
    Node* nRL  = nR->left;
    int   hRL0 = height(nRL);
    int   hRR0 = height(nR->right);

    if (hRR0 >= hRL0) {
      return rotate_left_nl(nParent, n, hL0, nR, nRL, hRL0, hRR0);
    }
    else {
      {
        publish(nRL);
        scoped_lock subLock(nRL->lock);

        int hRL = nRL->height;
        if (hRR0 >= hRL) {
          return rotate_left_nl(nParent, n, hL0, nR, nRL, hRL, hRR0);
        }
        else {
          int hRLR = height(nRL->right);
          int b    = hRR0 - hRLR;
          if (b >= -1 && b <= 1) {
            return rotate_left_over_right_nl(nParent, n, hL0, nR, nRL, hRR0, hRLR);
          }
        }
      }

      return rebalance_to_right_nl(n, nR, nRL, hRR0);
    }
  }
}

template <typename Value_T, typename Key_T, int Threads>
typename Tree<Value_T, Key_T, Threads>::Node* Tree<Value_T, Key_T, Threads>::rotate_right_nl(Node* nParent, Node* n,
                                                                                             Node* nL, int hR, int hLL,
                                                                                             Node* nLR, int hLR)
{
  long  nodeOVL = n->version;
  Node* nPL     = nParent->left;
  n->version    = begin_change(nodeOVL);

  n->left = nLR;
  if (nLR) {
    nLR->parent = n;
  }

  nL->right = n;
  n->parent = nL;

  if (nPL == n) {
    nParent->left = nL;
  }
  else {
    nParent->right = nL;
  }
  nL->parent = nParent;

  int hNRepl = 1 + std::max(hLR, hR);
  n->height  = hNRepl;
  nL->height = 1 + std::max(hLL, hNRepl);

  n->version = end_change(nodeOVL);

  int balN = hLR - hR;
  if (balN < -1 || balN > 1) {
    return n;
  }

  int balL = hLL - hNRepl;
  if (balL < -1 || balL > 1) {
    return nL;
  }

  return fix_height_nl(nParent);
}

template <typename Value_T, typename Key_T, int Threads>
typename Tree<Value_T, Key_T, Threads>::Node* Tree<Value_T, Key_T, Threads>::rotate_left_nl(Node* nParent, Node* n,
                                                                                            int hL, Node* nR, Node* nRL,
                                                                                            int hRL, int hRR)
{
  long  nodeOVL = n->version;
  Node* nPL     = nParent->left;
  n->version    = begin_change(nodeOVL);

  n->right = nRL;
  if (nRL) {
    nRL->parent = n;
  }

  nR->left  = n;
  n->parent = nR;

  if (nPL == n) {
    nParent->left = nR;
  }
  else {
    nParent->right = nR;
  }
  nR->parent = nParent;

  int hNRepl = 1 + std::max(hL, hRL);
  n->height  = hNRepl;
  nR->height = 1 + std::max(hNRepl, hRR);

  n->version = end_change(nodeOVL);

  int balN = hRL - hL;
  if (balN < -1 || balN > 1) {
    return n;
  }

  int balR = hRR - hNRepl;
  if (balR < -1 || balR > 1) {
    return nR;
  }

  return fix_height_nl(nParent);
}

template <typename Value_T, typename Key_T, int Threads>
typename Tree<Value_T, Key_T, Threads>::Node*   Tree<Value_T, Key_T, Threads>::rotate_right_over_left_nl(
  Node* nParent, Node* n, Node* nL, int hR, int hLL, Node* nLR, int hLRL)
{
  long nodeOVL = n->version;
  long leftOVL = nL->version;

  Node* nPL  = nParent->left;
  Node* nLRL = nLR->left;
  Node* nLRR = nLR->right;
  int   hLRR = height(nLRR);

  n->version  = begin_change(nodeOVL);
  nL->version = begin_change(leftOVL);

  n->left = nLRR;
  if (nLRR) {
    nLRR->parent = n;
  }

  nL->right = nLRL;
  if (nLRL) {
    nLRL->parent = nL;
  }

  nLR->left  = nL;
  nL->parent = nLR;
  nLR->right = n;
  n->parent  = nLR;

  if (nPL == n) {
    nParent->left = nLR;
  }
  else {
    nParent->right = nLR;
  }
  nLR->parent = nParent;

  int hNRepl = 1 + std::max(hLRR, hR);
  n->height  = hNRepl;

  int hLRepl = 1 + std::max(hLL, hLRL);
  nL->height = hLRepl;

  nLR->height = 1 + std::max(hLRepl, hNRepl);

  n->version  = end_change(nodeOVL);
  nL->version = end_change(leftOVL);

  int balN = hLRR - hR;
  if (balN < -1 || balN > 1) {
    return n;
  }

  int balLR = hLRepl - hNRepl;
  if (balLR < -1 || balLR > 1) {
    return nLR;
  }

  return fix_height_nl(nParent);
}

template <typename Value_T, typename Key_T, int Threads>
typename Tree<Value_T, Key_T, Threads>::Node*   Tree<Value_T, Key_T, Threads>::rotate_left_over_right_nl(
  Node* nParent, Node* n, int hL, Node* nR, Node* nRL, int hRR, int hRLR)
{
  long nodeOVL  = n->version;
  long rightOVL = nR->version;

  n->version  = begin_change(nodeOVL);
  nR->version = begin_change(rightOVL);

  Node* nPL  = nParent->left;
  Node* nRLL = nRL->left;
  Node* nRLR = nRL->right;
  int   hRLL = height(nRLL);

  n->right = nRLL;
  if (nRLL) {
    nRLL->parent = n;
  }

  nR->left = nRLR;
  if (nRLR) {
    nRLR->parent = nR;
  }

  nRL->right = nR;
  nR->parent = nRL;
  nRL->left  = n;
  n->parent  = nRL;

  if (nPL == n) {
    nParent->left = nRL;
  }
  else {
    nParent->right = nRL;
  }
  nRL->parent = nParent;

  int hNRepl  = 1 + std::max(hL, hRLL);
  n->height   = hNRepl;
  int hRRepl  = 1 + std::max(hRLR, hRR);
  nR->height  = hRRepl;
  nRL->height = 1 + std::max(hNRepl, hRRepl);

  n->version  = end_change(nodeOVL);
  nR->version = end_change(rightOVL);

  int balN = hRLL - hL;
  if (balN < -1 || balN > 1) {
    return n;
  }

  int balRL = hRRepl - hNRepl;
  if (balRL < -1 || balRL > 1) {
    return nRL;
  }

  return fix_height_nl(nParent);
}
}
}  // namespace Concurrent::AVL
#endif
