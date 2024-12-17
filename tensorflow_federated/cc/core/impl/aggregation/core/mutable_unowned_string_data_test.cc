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

#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_unowned_string_data.h"

#include <string>
#include <vector>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/strings/string_view.h"
#include "tensorflow_federated/cc/testing/status_matchers.h"

namespace tensorflow_federated {
namespace aggregation {
namespace {

TEST(MutableUnownedStringDataTest, MutableUnownedStringDataValid) {
  std::string string_1 = "foo";
  std::string string_2 = "bar";
  std::string string_3 = "baz";
  MutableUnownedStringData vector_data;
  vector_data.push_back(absl::string_view(string_1));
  vector_data.push_back(absl::string_view(string_2));
  vector_data.push_back(absl::string_view(string_3));
  EXPECT_THAT(vector_data.CheckValid<absl::string_view>(), IsOk());
}

TEST(MutableUnownedStringDataTest, EncodeDecodeSucceeds) {
  std::string string_1 = "foo";
  std::string string_2 = "bar";
  std::string string_3 = "baz";
  MutableUnownedStringData vector_data;
  vector_data.push_back(absl::string_view(string_1));
  vector_data.push_back(absl::string_view(string_2));
  vector_data.push_back(absl::string_view(string_3));
  std::string encoded_vector_data = vector_data.EncodeContent();
  EXPECT_THAT(vector_data.CheckValid<absl::string_view>(), IsOk());
  auto decoded_vector_data =
      MutableUnownedStringData::CreateFromEncodedContent(encoded_vector_data);
  EXPECT_THAT(decoded_vector_data->CheckValid<absl::string_view>(), IsOk());
  EXPECT_EQ((*decoded_vector_data)[0], string_1);
  EXPECT_EQ((*decoded_vector_data)[1], string_2);
  EXPECT_EQ((*decoded_vector_data)[2], string_3);
}

}  // namespace
}  // namespace aggregation
}  // namespace tensorflow_federated
