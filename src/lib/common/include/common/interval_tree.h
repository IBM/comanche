/*
   eXokernel Development Kit (XDK)

   Samsung Research America Copyright (C) 2013

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.
   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <http://www.gnu.org/licenses/>.

   As a special exception, if you link the code in this file with
   files compiled with a GNU compiler to produce an executable,
   that does not cause the resulting executable to be covered by
   the GNU Lesser General Public License.  This exception does not
   however invalidate any other reasons why the executable file
   might be covered by the GNU Lesser General Public License.
   This exception applies to code released by its copyright holders
   in files containing the exception.
*/

/*
 * Modifications by:
 *
 *   @author Daniel Waddington (daniel.waddington@ibm.com)
 */

#ifndef __INTERVAL_TREE_H__
#define __INTERVAL_TREE_H__

#include <algorithm>
#include <iostream>
#include <vector>

namespace Common
{
template <class T, typename K = std::size_t>
class Interval {
 public:
  K start;
  K stop;
  T value;
  Interval(K s, K e, const T &v) : start(s), stop(e), value(v) {}
  Interval() : start(0), stop(0) {}
};

template <class T, typename K = std::size_t>
bool operator<(const Interval<T, K> &left, const Interval<T, K> &right) {
  return left.start <= right.start;
}

template <class T, typename K>
K interval_start(const Interval<T, K> &i) {
  return i.start;
}

template <class T, typename K>
K interval_stop(const Interval<T, K> &i) {
  return i.stop;
}

template <class T, typename K>
std::ostream &operator<<(std::ostream &out, Interval<T, K> &i) {
  out << "Interval(" << i.start << ", " << i.stop << "): " << i.value;
  return out;
}

template <class T, typename K = std::size_t>
class IntervalStartSorter {
 public:
  bool operator()(const Interval<T, K> &a, const Interval<T, K> &b) {
    return a.start < b.start;
  }
};

template <class T, typename K = std::size_t>
class Interval_tree {
 public:
  typedef Interval<T, K> interval_t;
  typedef std::vector<interval_t> interval_vector_t;
  typedef Interval_tree<T, K> interval_tree_t;

  interval_vector_t intervals;
  interval_tree_t *left;
  interval_tree_t *right;
  K center;

  Interval_tree<T, K>(void) : left(NULL), right(NULL), center(0) {}

  Interval_tree<T, K>(const interval_tree_t &other) : left(NULL), right(NULL) {
    center = other.center;
    intervals = other.intervals;
    if (other.left) left = new interval_tree_t(*other.left);

    if (other.right) right = new interval_tree_t(*other.right);
  }

  Interval_tree<T, K> &operator=(const interval_tree_t &other) {
    center = other.center;
    intervals = other.intervals;
    if (other.left) {
      left = new interval_tree_t(*other.left);
    } else {
      if (left) delete left;
      left = NULL;
    }

    if (other.right) {
      right = new interval_tree_t(*other.right);
    } else {
      if (right) delete right;
      right = NULL;
    }
    return *this;
  }

  Interval_tree<T, K>(interval_vector_t &ivals, std::size_t depth = 16,
                      std::size_t minbucket = 64, K leftextent = 0,
                      K rightextent = 0, std::size_t maxbucket = 512)
      : left(NULL), right(NULL) {
    --depth;
    IntervalStartSorter<T, K> intervalStartSorter;
    if (depth == 0 || (ivals.size() < minbucket && ivals.size() < maxbucket)) {
      std::sort(ivals.begin(), ivals.end(), intervalStartSorter);
      intervals = ivals;
    } else {
      if (leftextent == 0 && rightextent == 0) {
        // sort intervals by start
        std::sort(ivals.begin(), ivals.end(), intervalStartSorter);
      }

      K leftp = 0;
      K rightp = 0;
      K centerp = 0;

      if (leftextent || rightextent) {
        leftp = leftextent;
        rightp = rightextent;
      } else {
        leftp = ivals.front().start;
        std::vector<K> stops;
        stops.resize(ivals.size());
        transform(ivals.begin(), ivals.end(), stops.begin(),
                  interval_stop<T, K>);
        rightp = *max_element(stops.begin(), stops.end());
      }

      // centerp = ( leftp + rightp ) / 2;
      centerp = ivals.at(ivals.size() / 2).start;
      center = centerp;

      interval_vector_t lefts, rights;

      for (auto interval : ivals) {
        if (interval.stop < center) {
          lefts.push_back(interval);
        } else if (interval.start > center) {
          rights.push_back(interval);
        } else {
          intervals.push_back(interval);
        }
      }

      if (!lefts.empty())
        left = new interval_tree_t(lefts, depth, minbucket, leftp, centerp);

      if (!rights.empty())
        right = new interval_tree_t(rights, depth, minbucket, centerp, rightp);
    }
  }

  void find_overlapping(K start, K stop, interval_vector_t &overlapping) const {
    if (!intervals.empty() && !(stop < intervals.front().start)) {
      for (auto interval : intervals) {
        if (interval.stop >= start && interval.start <= stop) {
          overlapping.push_back(interval);
        }
      }
    }

    if (left && start <= center)
      left->find_overlapping(start, stop, overlapping);

    if (right && stop >= center)
      right->find_overlapping(start, stop, overlapping);
  }

  bool find_point(K point, interval_t &ret_val) const {
    if (!intervals.empty() && !(point < intervals.front().start)) {
      for (auto interval : intervals) {
        if (point >= interval.start && point <= interval.stop) {
          ret_val = interval;
          return true;
        }
      }
    }

    if (left && point <= center) return left->find_point(point, ret_val);

    if (right && point >= center) return right->find_point(point, ret_val);

    return false;
  }

  K earliest_time() const { return intervals.front().start; }

  K latest_time() const { return intervals.back().stop; }

  void find_contained(K start, K stop, interval_vector_t &contained) const {
    if (!intervals.empty() && !(stop < intervals.front().start)) {
      for (auto interval : intervals) {
        if (interval.start >= start && interval.stop <= stop) {
          contained.push_back(interval);
        }
      }
    }

    if (left && start <= center) left->find_contained(start, stop, contained);

    if (right && stop >= center) right->find_contained(start, stop, contained);
  }

  ~Interval_tree(void) {
    // traverse the left and right
    // delete them all the way down
    if (left) delete left;

    if (right) delete right;
  }
};
}  // namespace Common
#endif
