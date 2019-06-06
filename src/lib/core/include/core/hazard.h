/*
   Copyright [2017-2019] [IBM Corporation]
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

#ifndef HAZARD_MANAGER
#define HAZARD_MANAGER

#include <cassert>

//#define HAZARD_DEBUG //Indicates that the at() function is used for array
// accesses

#include <algorithm>
#include <array>
#include <iostream>
#include <list>

namespace Concurrent
{
// Thread local id
// Note: __thread is GCC specific
extern __thread unsigned int __hazard_thread_num;

/*!
 * A manager for Hazard Pointers manipulation.
 * \param Node The type of node to manage.
 * \param Threads The maximum number of threads.
 * \param Size The number of hazard pointers per thread.
 * \param Prefill The number of nodes to precreate in the queue.
 */
template <typename Node, unsigned int Threads, unsigned int Size = 2,
          unsigned int Prefill = 50>
class Hazard_manager {
 public:
  Hazard_manager();
  ~Hazard_manager();

  Hazard_manager(const Hazard_manager& rhs) = delete;
  Hazard_manager& operator=(const Hazard_manager& rhs) = delete;

  /*!
   * Release the node.
   */
  void release_node(Node* node);

  /*!
   * \brief Release the node by checking first if it is not already in the
   * queue. This method can be slow depending on the number of nodes already
   * released. \param node The node to release.
   */
  void safe_release_node(Node* node);

  /*!
   * Return a free node for the calling thread.
   * \return A free node
   */
  Node* get_free_node();

  /*!
   * Publish a reference to the given Node using ith Hazard Pointer.
   * \param node The node to be published
   * \param i The index of the pointer to use.
   */
  void publish(Node* node, unsigned int i);

  /*!
   * Release the ith reference of the calling thread.
   * \param i The reference index.
   */
  void release(unsigned int i);

  /*!
   * Release all the hazard points of the calling thread.
   */
  void releaseAll();

  /*!
   * Return a reference to the internal free queue of the given thread.
   * \return A reference to the free queue of the given thread.
   */
  std::list<Node*>& direct_free(unsigned int t);

  /*!
   * Return a reference to the internal local queue of the given thread.
   * \return A reference to the local queue of the given thread.
   */
  std::list<Node*>& direct_local(unsigned int t);

 private:
  std::array<std::array<Node*, Size>, Threads> Pointers;
  std::array<std::list<Node*>, Threads> LocalQueues;
  std::array<std::list<Node*>, Threads> FreeQueues;

  bool isReferenced(Node* node);

  /* Verify the template parameters */
  static_assert(Threads > 0, "The number of threads must be greater than 0");
  static_assert(Size > 0, "The number of hazard pointers must greater than 0");
};

template <typename Node, unsigned int Threads, unsigned int Size,
          unsigned int Prefill>
Hazard_manager<Node, Threads, Size, Prefill>::Hazard_manager() {
  for (unsigned int tid = 0; tid < Threads; ++tid) {
    for (unsigned int j = 0; j < Size; ++j) {
#ifdef HAZARD_DEBUG
      Pointers.at(tid).at(j) = nullptr;
#else
      Pointers[tid][j] = nullptr;
#endif
    }

    if (Prefill > 0) {
      for (unsigned int i = 0; i < Prefill; i++) {
#ifdef HAZARD_DEBUG
        FreeQueues.at(tid).push_back(new Node());
#else
        FreeQueues[tid].push_back(new Node());
#endif
      }
    }
  }
}

template <typename Node, unsigned int Threads, unsigned int Size,
          unsigned int Prefill>
Hazard_manager<Node, Threads, Size, Prefill>::~Hazard_manager() {
  for (unsigned int tid = 0; tid < Threads; ++tid) {
    // No need to delete Hazard Pointers because each thread need to release its
    // published references

#ifdef HAZARD_DEBUG
    while (!LocalQueues.at(tid).empty()) {
      delete LocalQueues.at(tid).front();
      LocalQueues.at(tid).pop_front();
    }

    while (!FreeQueues.at(tid).empty()) {
      delete FreeQueues.at(tid).front();
      FreeQueues.at(tid).pop_front();
    }
#else
    while (!LocalQueues[tid].empty()) {
      delete LocalQueues[tid].front();
      LocalQueues[tid].pop_front();
    }

    while (!FreeQueues[tid].empty()) {
      delete FreeQueues[tid].front();
      FreeQueues[tid].pop_front();
    }
#endif
  }
}

template <typename Node, unsigned int Threads, unsigned int Size,
          unsigned int Prefill>
std::list<Node*>& Hazard_manager<Node, Threads, Size, Prefill>::direct_free(
    unsigned int t) {
#ifdef HAZARD_DEBUG
  return FreeQueues.at(t);
#else
  return FreeQueues[t];
#endif
}

template <typename Node, unsigned int Threads, unsigned int Size,
          unsigned int Prefill>
std::list<Node*>& Hazard_manager<Node, Threads, Size, Prefill>::direct_local(
    unsigned int t) {
#ifdef HAZARD_DEBUG
  return LocalQueues.at(t);
#else
  return LocalQueues[t];
#endif
}

template <typename Node, unsigned int Threads, unsigned int Size,
          unsigned int Prefill>
void Hazard_manager<Node, Threads, Size, Prefill>::safe_release_node(
    Node* node) {
  // If the node is null, we have nothing to do
  if (node) {
    if (std::find(LocalQueues.at(__hazard_thread_num).begin(),
                  LocalQueues.at(__hazard_thread_num).end(),
                  node) != LocalQueues.at(__hazard_thread_num).end()) {
      return;
    }

    // Add the node to the localqueue
    LocalQueues.at(__hazard_thread_num).push_back(node);
  }
}

template <typename Node, unsigned int Threads, unsigned int Size,
          unsigned int Prefill>
void Hazard_manager<Node, Threads, Size, Prefill>::release_node(Node* node) {
  // If the node is null, we have nothing to do
  if (node) {
#ifdef HAZARD_DEBUG
    // Add the node to the localqueue
    LocalQueues.at(__hazard_thread_num).push_back(node);
#else
    // Add the node to the localqueue
    LocalQueues[__hazard_thread_num].push_back(node);
#endif
  }
}

template <typename Node, unsigned int Threads, unsigned int Size,
          unsigned int Prefill>
Node* Hazard_manager<Node, Threads, Size, Prefill>::get_free_node() {
  int tid = __hazard_thread_num;

#ifdef HAZARD_DEBUG
  // First, try to get a free node from the free queue
  if (!FreeQueues.at(tid).empty()) {
    Node* free = FreeQueues.at(tid).front();
    FreeQueues.at(tid).pop_front();

    return free;
  }

  // If there are enough local nodes, move then to the free queue
  if (LocalQueues.at(tid).size() > (Size + 1) * Threads) {
    typename std::list<Node*>::iterator it = LocalQueues.at(tid).begin();
    typename std::list<Node*>::iterator end = LocalQueues.at(tid).end();

    while (it != end) {
      if (!isReferenced(*it)) {
        FreeQueues.at(tid).push_back(*it);

        it = LocalQueues.at(tid).erase(it);
      }
      else {
        ++it;
      }
    }

    Node* free = FreeQueues.at(tid).front();
    FreeQueues.at(tid).pop_front();

    return free;
  }
#else
  // First, try to get a free node from the free queue
  if (!FreeQueues[tid].empty()) {
    Node* free = FreeQueues[tid].front();
    FreeQueues[tid].pop_front();

    return free;
  }

  // If there are enough local nodes, move then to the free queue
  if (LocalQueues[tid].size() > (Size + 1) * Threads) {
    typename std::list<Node*>::iterator it = LocalQueues[tid].begin();
    typename std::list<Node*>::iterator end = LocalQueues[tid].end();

    while (it != end) {
      if (!isReferenced(*it)) {
        FreeQueues[tid].push_back(*it);

        it = LocalQueues[tid].erase(it);
      }
      else {
        ++it;
      }
    }

    Node* free = FreeQueues[tid].front();
    FreeQueues[tid].pop_front();

    return free;
  }
#endif

  // There was no way to get a free node, allocate a new one
  return new Node();
}

template <typename Node, unsigned int Threads, unsigned int Size,
          unsigned int Prefill>
bool Hazard_manager<Node, Threads, Size, Prefill>::isReferenced(Node* node) {
#ifdef HAZARD_DEBUG
  for (unsigned int tid = 0; tid < Threads; ++tid) {
    for (unsigned int i = 0; i < Size; ++i) {
      if (Pointers.at(tid).at(i) == node) {
        return true;
      }
    }
  }
#else
  for (unsigned int tid = 0; tid < Threads; ++tid) {
    for (unsigned int i = 0; i < Size; ++i) {
      if (Pointers[tid][i] == node) {
        return true;
      }
    }
  }
#endif

  return false;
}

template <typename Node, unsigned int Threads, unsigned int Size,
          unsigned int Prefill>
void Hazard_manager<Node, Threads, Size, Prefill>::publish(Node* node,
                                                           unsigned int i) {
#ifdef HAZARD_DEBUG
  Pointers.at(__hazard_thread_num).at(i) = node;
#else
  Pointers[__hazard_thread_num][i] = node;
#endif
}

template <typename Node, unsigned int Threads, unsigned int Size,
          unsigned int Prefill>
void Hazard_manager<Node, Threads, Size, Prefill>::release(unsigned int i) {
#ifdef HAZARD_DEBUG
  Pointers.at(__hazard_thread_num).at(i) = nullptr;
#else
  Pointers[__hazard_thread_num][i] = nullptr;
#endif
}

template <typename Node, unsigned int Threads, unsigned int Size,
          unsigned int Prefill>
void Hazard_manager<Node, Threads, Size, Prefill>::releaseAll() {
#ifdef HAZARD_DEBUG
  for (unsigned int i = 0; i < Size; ++i) {
    Pointers.at(__hazard_thread_num).at(i) = nullptr;
  }
#else
  for (unsigned int i = 0; i < Size; ++i) {
    Pointers[__hazard_thread_num][i] = nullptr;
  }
#endif
}

}  // namespace Concurrent

#endif
