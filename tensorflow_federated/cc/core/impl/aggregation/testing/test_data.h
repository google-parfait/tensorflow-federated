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

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_TESTING_TEST_DATA_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_TESTING_TEST_DATA_H_

#include <initializer_list>
#include <memory>
#include <string>
#include <utility>

#include "absl/strings/string_view.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_string_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_unowned_string_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_vector_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_data.h"

namespace tensorflow_federated::aggregation {

// Creates test tensor data based on a vector<T> for all arithmetic types.
template <typename T>
std::unique_ptr<TensorData> CreateTestData(std::initializer_list<T> values) {
  return std::make_unique<MutableVectorData<T>>(values);
}

// Creates test tensor data based on a vector<absl::string_view>.
template <>
std::unique_ptr<TensorData> CreateTestData(
    std::initializer_list<absl::string_view> values) {
  return std::make_unique<MutableUnownedStringData>(values);
}

// Creates test tensor data based on a vector<std::string>.
template <>
std::unique_ptr<TensorData> CreateTestData(
    std::initializer_list<std::string> values) {
  auto data = std::make_unique<MutableStringData>(values.size());
  for (auto value : values) {
    data->Add(std::move(value));
  }
  return data;
}

}  // namespace tensorflow_federated::aggregation

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_TESTING_TEST_DATA_H_
