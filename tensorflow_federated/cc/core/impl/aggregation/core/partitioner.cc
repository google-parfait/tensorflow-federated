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
#include <utility>
#include <vector>

#include "absl/hash/hash.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/agg_vector.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/datatype.h"
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
        hashes[index] = CombineHashes(hashes[index], absl::HashOf(value));
      }
    });
  }
  for (auto& hash : hashes) {
    hash = hash % num_partitions;
  }
  return Partitioner(std::move(hashes), num_partitions);
}
}  // namespace aggregation
}  // namespace tensorflow_federated
