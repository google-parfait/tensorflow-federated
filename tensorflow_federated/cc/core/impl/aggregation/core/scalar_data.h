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

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_SCALAR_DATA_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_SCALAR_DATA_H_

#include <cstddef>
#include <string>
#include <utility>

#include "absl/strings/string_view.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_data.h"

namespace tensorflow_federated {
namespace aggregation {

template <typename T>
class ScalarData : public TensorData {
 public:
  explicit ScalarData(T value) : value_(value) {}

  const void* data() const override { return &value_; }
  size_t byte_size() const override { return sizeof(T); }

 private:
  T value_;
};

template <>
class ScalarData<absl::string_view> : public TensorData {
 public:
  explicit ScalarData(absl::string_view value)
      : value_(value), value_view_(value_) {}
  explicit ScalarData(std::string&& value)
      : value_(std::move(value)), value_view_(value_) {}

  const void* data() const override { return &value_view_; }
  size_t byte_size() const override { return sizeof(absl::string_view); }

 private:
  std::string value_;
  absl::string_view value_view_;
};

}  // namespace aggregation
}  // namespace tensorflow_federated

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_SCALAR_DATA_H_
