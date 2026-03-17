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

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_TESTING_TENSORFLOW_TESTING_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_TESTING_TENSORFLOW_TESTING_H_

#include <cstdint>
#include <initializer_list>
#include <string>

#include "googletest/include/gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/string_view.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/datatype.h"

namespace tensorflow_federated::aggregation {

template <typename T>
::tensorflow::Tensor CreateTfTensor(::tensorflow::DataType data_type,
                                    std::initializer_list<int64_t> dim_sizes,
                                    std::initializer_list<T> values) {
  ::tensorflow::TensorShape shape;
  EXPECT_TRUE(
      ::tensorflow::TensorShape::BuildTensorShape(dim_sizes, &shape).ok());
  ::tensorflow::Tensor tensor(data_type, shape);
  T* tensor_data_ptr = reinterpret_cast<T*>(tensor.data());
  for (auto value : values) {
    *tensor_data_ptr++ = value;
  }
  return tensor;
}

::tensorflow::Tensor CreateStringTfTensor(
    std::initializer_list<int64_t> dim_sizes,
    std::initializer_list<string_view> values);

// Wrapper around tf::ops::Save that sets up and runs the op.
absl::Status CreateTfCheckpoint(::tensorflow::Input filename,
                                ::tensorflow::Input tensor_names,
                                ::tensorflow::InputList tensors);

// Returns a summary of the checkpoint as a map of tensor names and values.
absl::StatusOr<absl::flat_hash_map<std::string, std::string>>
SummarizeCheckpoint(const absl::Cord& checkpoint);

}  // namespace tensorflow_federated::aggregation

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_TESTING_TENSORFLOW_TESTING_H_
