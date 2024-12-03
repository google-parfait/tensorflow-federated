/*
 * Copyright 2022 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_AGG_VECTOR_ITERATOR_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_AGG_VECTOR_ITERATOR_H_

#include <cstddef>

#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_data.h"

namespace tensorflow_federated {
namespace aggregation {

// Iterator for AggVector which allows to iterate over sparse values
// as a collection of {index, value} pairs.
//
// This allows a simple iteration loops like the following:
// for (auto [index, value] : agg_vector) {
//    ... aggregate the value at the given dense index
// }
template <typename T>
struct AggVectorIterator {
  struct IndexValuePair {
    size_t index;
    T value;

    friend bool operator==(const IndexValuePair& a, const IndexValuePair& b) {
      return a.index == b.index && a.value == b.value;
    }

    friend bool operator!=(const IndexValuePair& a, const IndexValuePair& b) {
      return a.index != b.index || a.value != b.value;
    }
  };

  using value_type = IndexValuePair;
  using pointer = value_type*;
  using reference = value_type&;

  explicit AggVectorIterator(const TensorData* data)
      : AggVectorIterator(get_start_ptr(data), get_end_ptr(data), 0) {
    // If the TensorData buffer is non-null but empty then `ptr` will equal
    // `end_ptr`. We must in that case ensure that the iterator returned by this
    // constructor is equal to end(), otherwise using operator== to compare
    // against end() would indicate that there is an element to access, which is
    // incorrect and would lead to a buffer overrun.
    if (ptr == end_ptr) {
      *this = end();
    }
  }

  // Current dense index corresponding to the current value.
  size_t index() const { return dense_index; }
  // Current value.
  T value() const { return *ptr; }
  // The current interator {index, value} pair value. This is used by
  // for loop iterators.
  IndexValuePair operator*() const { return {dense_index, *ptr}; }

  AggVectorIterator& operator++() {
    TFF_CHECK(ptr != end_ptr);
    if (++ptr == end_ptr) {
      *this = end();
    } else {
      dense_index++;
    }
    return *this;
  }

  AggVectorIterator operator++(int) {
    AggVectorIterator tmp = *this;
    ++(*this);
    return tmp;
  }

  friend bool operator==(const AggVectorIterator& a,
                         const AggVectorIterator& b) {
    return a.ptr == b.ptr;
  }

  friend bool operator!=(const AggVectorIterator& a,
                         const AggVectorIterator& b) {
    return a.ptr != b.ptr;
  }

  static AggVectorIterator end() {
    return AggVectorIterator(nullptr, nullptr, 0);
  }

 private:
  AggVectorIterator(const T* ptr, const T* end_ptr, size_t dense_index)
      : ptr(ptr), end_ptr(end_ptr), dense_index(dense_index) {}

  static const T* get_start_ptr(const TensorData* data) {
    return static_cast<const T*>(data->data());
  }

  static const T* get_end_ptr(const TensorData* data) {
    return get_start_ptr(data) + data->byte_size() / sizeof(T);
  }

  const T* ptr;
  const T* end_ptr;
  size_t dense_index;
};

}  // namespace aggregation
}  // namespace tensorflow_federated

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_AGG_VECTOR_ITERATOR_H_
