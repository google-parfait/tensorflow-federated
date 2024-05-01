/*
 * Copyright 2024 Google LLC
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

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_MUTABLE_STRING_DATA_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_MUTABLE_STRING_DATA_H_

#include <algorithm>
#include <cstddef>
#include <list>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/fixed_array.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/datatype.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_data.h"

namespace tensorflow_federated {
namespace aggregation {

inline constexpr size_t kMinGrowthSize = 128;

// MutableStringData owns its string values and allows string values to be added
// one by one.
class MutableStringData : public TensorData {
 public:
  explicit MutableStringData(size_t expected_size) {
    growth_size_ = std::max(kMinGrowthSize, expected_size);
    strings_.emplace_back(absl::FixedArray<std::string>(growth_size_));
    string_views_.reserve(expected_size);
  }
  ~MutableStringData() override = default;

  // Implementation of TensorData methods.
  size_t byte_size() const override {
    return string_views_.size() * sizeof(string_view);
  }
  const void* data() const override { return string_views_.data(); }

  void Add(std::string&& string) {
    strings_.back()[fixed_array_index_] = std::move(string);
    string_views_.emplace_back(strings_.back()[fixed_array_index_]);
    fixed_array_index_++;
    if (fixed_array_index_ >= growth_size_) {
      fixed_array_index_ = 0;
      strings_.emplace_back(absl::FixedArray<std::string>(growth_size_));
    }
  }

 private:
  std::list<absl::FixedArray<std::string>> strings_;
  std::vector<string_view> string_views_;
  // Size of each FixedArray held by `strings_`. `strings_` is a linked list of
  // FixedArrays to provide pointer stability for `string_views`.
  size_t growth_size_;
  size_t fixed_array_index_ = 0;
};

}  // namespace aggregation
}  // namespace tensorflow_federated

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_MUTABLE_STRING_DATA_H_
