/*
 * Copyright 2023 Google LLC
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

#include "tensorflow_federated/cc/core/impl/aggregation/core/partitioner.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/agg_vector.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/datatype.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_vector_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"

namespace tensorflow_federated {
namespace aggregation {

namespace {
// Combines two hashes together.
// Similar to boost's hash_combine().
size_t CombineHashes(size_t a, size_t b) {
  a ^= b + 0x9e3779b9 + (a << 6) + (a >> 2);
  return a;
}

template <typename T>
std::vector<std::vector<T>> PartitionInternal(
    absl::Span<const T> input, int num_partitions,
    const std::vector<size_t>& hashes) {
  std::vector<std::vector<T>> slices(num_partitions);

  for (int index = 0; index < input.size(); ++index) {
    auto hashed_index = hashes[index];
    slices[hashed_index].push_back(input[index]);
  }
  return slices;
}
}  // namespace

StatusOr<Partitioner> Partitioner::Create(
    const std::vector<Tensor>& key_tensors, int num_partitions) {
  if (key_tensors.empty()) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "Expected at least one output key tensor.";
  }
  if (key_tensors[0].shape().dim_sizes().size() != 1) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "Expected key tensor to be one-dimensional.";
  }
  int key_size = key_tensors[0].shape().dim_sizes()[0];
  for (const auto& key : key_tensors) {
    if (key.shape().dim_sizes().size() != 1 ||
        key.shape().dim_sizes()[0] != key_size) {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "All key tensors must have the same one-dimensional size.";
    }
  }
  std::vector<size_t> hashes(key_size, 0);
  for (const auto& key : key_tensors) {
    DTYPE_CASES(key.dtype(), T, {
      AggVector<T> data = key.AsAggVector<T>();
      for (const auto& [index, value] : data) {
        hashes[index] = CombineHashes(hashes[index], std::hash<T>()(value));
      }
    });
  }
  for (auto& hash : hashes) {
    hash = hash % num_partitions;
  }
  return Partitioner(std::move(hashes), num_partitions);
}

StatusOr<std::vector<Tensor>> Partitioner::PartitionKeys(
    const Tensor& key_tensor) {
  std::vector<Tensor> result_tensors;
  DTYPE_CASES(key_tensor.dtype(), T, {
    auto slices =
        PartitionInternal<T>(key_tensor.AsSpan<T>(), num_partitions_, hashes_);
    for (auto& slice : slices) {
      int dim_size = slice.size();
      TFF_ASSIGN_OR_RETURN(
          Tensor tensor,
          Tensor::Create(key_tensor.dtype(), {static_cast<int64_t>(dim_size)},
                         std::make_unique<MutableVectorData<T>>(slice.begin(),
                                                                slice.end())));
      result_tensors.push_back(std::move(tensor));
    }
  });
  return result_tensors;
}

}  // namespace aggregation
}  // namespace tensorflow_federated
