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
#include "tensorflow_federated/cc/core/impl/aggregation/core/vector_partitioner.h"

#include <cstdint>
#include <vector>

#include "googletest/include/gtest/gtest.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/datatype.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/test_data.h"

namespace tensorflow_federated {
namespace aggregation {
namespace {

TEST(VectorPartitionerTest, CreateVectorPartitioner_InvalidKeySize) {
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

  EXPECT_DEATH(
      VectorPartitioner partitioner(key_tensors,
                                    /*key_size=*/4, /*num_partitions=*/2),
      "Expected key size to be 4 but got 3.");
}

TEST(VectorPartitionerTest, CreateVectorPartitioner_EmptyKeyTensors) {
  std::vector<Tensor> key_tensors;
  EXPECT_DEATH(VectorPartitioner partitioner(key_tensors,
                                             /*key_size=*/3,
                                             /*num_partitions=*/2),
               "Expected at least one output key tensor.");
}
}  // namespace
}  // namespace aggregation
}  // namespace tensorflow_federated
