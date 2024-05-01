/*
 * Copyright 2018 Google LLC
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

#include "tensorflow_federated/cc/core/impl/aggregation/base/move_to_lambda.h"

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/unique_value.h"

namespace tensorflow_federated {

using ::testing::Eq;

TEST(MoveToLambda, Basic) {
  auto capture = MoveToLambda(UniqueValue<int>{123});
  auto lambda = [capture]() {
    EXPECT_TRUE(capture->has_value()) << "Should have moved the original";
    return **capture;
  };

  int returned = lambda();
  EXPECT_FALSE(capture->has_value()) << "Should have moved the original";
  EXPECT_THAT(returned, Eq(123));

  int returned_again = lambda();
  EXPECT_THAT(returned_again, Eq(123)) << "Usage shouldn't be destructive";
}

TEST(MoveToLambda, Mutable) {
  auto capture = MoveToLambda(UniqueValue<int>{0});
  auto counter = [capture]() mutable {
    EXPECT_TRUE(capture->has_value()) << "Should have moved the original";
    return (**capture)++;
  };

  EXPECT_FALSE(capture->has_value()) << "Should have moved the original";

  EXPECT_THAT(counter(), Eq(0));
  EXPECT_THAT(counter(), Eq(1));
  EXPECT_THAT(counter(), Eq(2));
}

}  // namespace tensorflow_federated
