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
#include "tensorflow_federated/cc/core/impl/aggregation/core/partitioner.h"

#include <cstdint>
#include <vector>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/datatype.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/test_data.h"
#include "tensorflow_federated/cc/testing/status_matchers.h"

namespace tensorflow_federated {
namespace aggregation {
namespace {

TEST(PartitionerTest, CreatePartitioner_Succeeds) {
  std::vector<Tensor> key_tensors;
  key_tensors.push_back(Tensor::Create(DT_STRING, {3},
                                       CreateTestData<string_view>({
                                           "a",
                                           "b",
                                           "c",
                                       }))
                            .value());
  key_tensors.push_back(Tensor::Create(DT_INT64, {3},
                                       CreateTestData<int64_t>({
                                           1,
                                           2,
                                           3,
                                       }))
                            .value());
  key_tensors.push_back(
      Tensor::Create(DT_DOUBLE, {3}, CreateTestData<double>({1.0, 2.0, 3.0}))
          .value());

  EXPECT_THAT(Partitioner::Create(key_tensors, /*num_partitions=*/2), IsOk());
}

TEST(PartitionerTest, CreatePartitioner_EmptyKeyTensors) {
  std::vector<Tensor> key_tensors;
  EXPECT_THAT(
      Partitioner::Create(key_tensors,
                          /*num_partitions=*/2),
      StatusIs(INVALID_ARGUMENT,
               testing::HasSubstr("Expected at least one output key tensor.")));
}

TEST(PartitionerTest, CreatePartitioner_KeyTensorNotOneDimensional) {
  std::vector<Tensor> key_tensors;
  key_tensors.push_back(Tensor::Create(DT_STRING, {1, 2},
                                       CreateTestData<string_view>({
                                           "a",
                                           "b",
                                       }))
                            .value());
  EXPECT_THAT(Partitioner::Create(key_tensors,
                                  /*num_partitions=*/2),
              StatusIs(INVALID_ARGUMENT,
                       testing::HasSubstr(
                           "Expected key tensor to be one-dimensional.")));
}

TEST(PartitionerTest, CreatePartitioner_KeyTensorsWithDifferentSizes) {
  std::vector<Tensor> key_tensors;
  key_tensors.push_back(Tensor::Create(DT_STRING, {3},
                                       CreateTestData<string_view>({
                                           "a",
                                           "b",
                                           "c",
                                       }))
                            .value());
  key_tensors.push_back(Tensor::Create(DT_INT64, {2},
                                       CreateTestData<int64_t>({
                                           1,
                                           2,
                                       }))
                            .value());
  EXPECT_THAT(Partitioner::Create(key_tensors,
                                  /*num_partitions=*/2),
              StatusIs(INVALID_ARGUMENT,
                       testing::HasSubstr("All key tensors must have the "
                                          "same one-dimensional size.")));
}
}  // namespace
}  // namespace aggregation
}  // namespace tensorflow_federated
