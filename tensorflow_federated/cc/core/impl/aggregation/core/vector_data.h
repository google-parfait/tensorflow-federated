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

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_VECTOR_DATA_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_VECTOR_DATA_H_

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_data.h"

namespace tensorflow_federated {
namespace aggregation {

// Immutable vector of values of type T.
template <typename T>
class VectorData : public TensorData {
 public:
  explicit VectorData(std::vector<T>&& values) : values_(std::move(values)) {}

  const void* data() const override { return values_.data(); }
  size_t byte_size() const override { return values_.size() * sizeof(T); }

 private:
  std::vector<T> values_;
};

template <>
class VectorData<absl::string_view> : public TensorData {
 public:
  explicit VectorData(std::vector<std::string>&& values)
      : strings_(std::move(values)) {
    string_views_.reserve(strings_.size());
    for (const std::string& s : strings_) string_views_.emplace_back(s);
  }

  // Implementation of TensorData methods.
  size_t byte_size() const override {
    return string_views_.size() * sizeof(absl::string_view);
  }
  const void* data() const override { return string_views_.data(); }

 private:
  std::vector<std::string> strings_;
  std::vector<absl::string_view> string_views_;
};

}  // namespace aggregation
}  // namespace tensorflow_federated

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_VECTOR_STRING_DATA_H_
