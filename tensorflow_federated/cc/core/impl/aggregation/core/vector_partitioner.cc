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

#include "tensorflow_federated/cc/core/impl/aggregation/core/vector_partitioner.h"

#include <cstddef>
#include <vector>

#include "absl/hash/hash.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/agg_vector.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/datatype.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"

namespace tensorflow_federated {
namespace aggregation {

namespace {
// Combine two hashes together.
// Similar to boost's hash_combine().
size_t CombineHashes(size_t a, size_t b) {
  a ^= b + 0x9e3779b9 + (a << 6) + (a >> 2);
  return a;
}

}  // namespace

VectorPartitioner::VectorPartitioner(const std::vector<Tensor>& key_tensors,
                                     int key_size, int num_partitions)
    : hashes_(key_size, 1) {
  TFF_CHECK(!key_tensors.empty())
      << "hash_keys: Expected at least one output key tensor.";
  int size = key_tensors[0].shape().dim_sizes()[0];
  TFF_CHECK(size == key_size) << "hash_keys: Expected key size to be "
                              << key_size << " but got " << size << ".";
  for (const auto& key : key_tensors) {
    DTYPE_CASES(key.dtype(), T, {
      AggVector<T> data = key.AsAggVector<T>();
      for (const auto& [index, value] : data) {
        hashes_[index] = CombineHashes(hashes_[index], absl::HashOf(value));
      }
    });
  }
  for (int i = 0; i < size; ++i) {
    hashes_[i] = hashes_[i] % num_partitions;
  }
}
}  // namespace aggregation
}  // namespace tensorflow_federated
