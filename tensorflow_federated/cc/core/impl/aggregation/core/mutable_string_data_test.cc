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
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_string_data.h"

#include <string>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/strings/string_view.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/datatype.h"
#include "tensorflow_federated/cc/core/impl/executors/status_matchers.h"

namespace tensorflow_federated {
namespace aggregation {
namespace {

TEST(MutableStringDataTest, MutableStringDataValid) {
  MutableStringData string_data(0);
  EXPECT_THAT(string_data.CheckValid<string_view>(), IsOk());
}

TEST(MutableStringDataTest, ValidAfterAddingValue) {
  MutableStringData string_data(1);
  string_data.Add("added-string");
  EXPECT_THAT(string_data.CheckValid<string_view>(), IsOk());
}

TEST(MutableStringDataTest, ValidAfterAddingValuePastExpectedSize) {
  MutableStringData string_data(1);
  string_data.Add("added-string");
  string_data.Add("more-than-expected-size");
  EXPECT_THAT(string_data.CheckValid<string_view>(), IsOk());
  const absl::string_view* first_element =
      static_cast<const absl::string_view*>(string_data.data());
  EXPECT_EQ(*first_element, "added-string");
}

}  // namespace
}  // namespace aggregation
}  // namespace tensorflow_federated
