/*
 * Copyright 2025 Google LLC
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

#ifndef TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_PARTITIONER_H_
#define TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_PARTITIONER_H_

#include <cstddef>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"

namespace tensorflow_federated {
namespace aggregation {

// This class aims to partition the input data into multiple slices.
// Internally it generates a vector of hashes from the key tensors and then uses
// the generated hashes to partition the input data.
class Partitioner {
 public:
  // Calculates hashes from the key tensors and creates a Partitioner instance.
  static StatusOr<Partitioner> Create(const std::vector<Tensor>& key_tensors,
                                      int num_partitions);

  // Partitions the input key tensor into multiple slices.
  StatusOr<std::vector<Tensor>> PartitionKeys(const Tensor& key_tensor);

  // Partitions the input vector data into multiple slices.
  template <typename T>
  StatusOr<std::vector<std::vector<T>>> PartitionData(
      const std::vector<T>& data) const {
    if (data.size() != hashes_.size()) {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "The number of elements in the input data should be equal "
                "to the number of hashes.";
    }
    return PartitionInternal<T>(data);
  }

  // Returns the number of partitions.
  int GetNumPartitions() const { return num_partitions_; }

 private:
  Partitioner(std::vector<size_t> hashes, int num_partitions)
      : hashes_(hashes), num_partitions_(num_partitions) {}

  template <typename T>
  std::vector<std::vector<T>> PartitionInternal(
      absl::Span<const T> input) const {
    std::vector<std::vector<T>> slices(num_partitions_);

    for (int index = 0; index < input.size(); ++index) {
      auto hashed_index = hashes_[index];
      // TODO: b/437952802 - Calculate and reserve the size of each slice
      slices[hashed_index].push_back(input[index]);
    }
    return slices;
  }

  std::vector<size_t> hashes_;
  int num_partitions_;
};
}  // namespace aggregation
}  // namespace tensorflow_federated

#endif  // TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_PARTITIONER_H_
