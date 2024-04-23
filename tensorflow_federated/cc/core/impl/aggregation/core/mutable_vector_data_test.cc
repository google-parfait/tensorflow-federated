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
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_vector_data.h"

#include <cstdint>
#include <string>
#include <vector>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "tensorflow_federated/cc/core/impl/executors/status_matchers.h"

namespace tensorflow_federated {
namespace aggregation {
namespace {

TEST(MutableVectorDataTest, MutableVectorDataValid) {
  MutableVectorData<int64_t> vector_data;
  vector_data.push_back(1);
  vector_data.push_back(2);
  vector_data.push_back(3);
  EXPECT_THAT(vector_data.CheckValid<int64_t>(), IsOk());
}

TEST(MutableVectorDataTest, EncodeDecodeSucceeds) {
  MutableVectorData<int64_t> vector_data;
  vector_data.push_back(1);
  vector_data.push_back(2);
  vector_data.push_back(3);
  std::string encoded_vector_data = vector_data.EncodeContent();
  EXPECT_THAT(vector_data.CheckValid<int64_t>(), IsOk());
  auto decoded_vector_data =
      MutableVectorData<int64_t>::CreateFromEncodedContent(encoded_vector_data);
  EXPECT_THAT(decoded_vector_data->CheckValid<int64_t>(), IsOk());
  EXPECT_EQ(std::vector<int64_t>(*decoded_vector_data),
            std::vector<int64_t>({1, 2, 3}));
}

}  // namespace
}  // namespace aggregation
}  // namespace tensorflow_federated
