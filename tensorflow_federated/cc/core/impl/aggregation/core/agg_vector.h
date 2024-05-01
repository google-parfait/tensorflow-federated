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

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_AGG_VECTOR_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_AGG_VECTOR_H_

#include <cstddef>

#include "tensorflow_federated/cc/core/impl/aggregation/core/agg_vector_iterator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_data.h"

namespace tensorflow_federated {
namespace aggregation {

// AggVector is flattened one-dimensional strongly typed view of tensor that
// provides immutable access to the values.
//
// AggVector hides the actual data organization of the tensor. The only
// way to access the tensor values is through the iterator that returns
// {index, value} pairs where each index is the dense index corresponding to
// the value.
//
// Example:
//
// template <typename T>
// void Iterate(const AggVector<T>& agg_vector) {
//   for (const auto& [index, value] : agg_vector) {
//     // Aggregate the `value` at the given `index`.
//   }
// }
//
template <typename T>
class AggVector final {
 public:
  using value_type = typename AggVectorIterator<T>::value_type;
  using const_iterator = AggVectorIterator<T>;

  // Iterator begin() function.
  const_iterator begin() const { return AggVectorIterator<T>(data_); }

  // Iterator end() function.
  const_iterator end() const { return AggVectorIterator<T>::end(); }

  // Entire AggVector length.
  size_t size() const { return size_; }

 private:
  // AggVector can be created only by Tensor::AsAggVector() method.
  friend class Tensor;
  explicit AggVector(const TensorData* data)
      : size_(data->byte_size() / sizeof(T)), data_(data) {}

  // The total length of the vector (in elements).
  size_t size_;
  // Tensor data, owned by the tensor object.
  const TensorData* data_;
};

}  // namespace aggregation
}  // namespace tensorflow_federated

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_AGG_VECTOR_H_
