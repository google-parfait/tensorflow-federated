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

#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator_registry.h"

#include <memory>
#include <string>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/intrinsic.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator_factory.h"
#include "tensorflow_federated/cc/testing/status_matchers.h"

namespace tensorflow_federated {
namespace aggregation {
namespace {

class MockFactory : public TensorAggregatorFactory {
  MOCK_METHOD(StatusOr<std::unique_ptr<TensorAggregator>>, Create,
              (const Intrinsic&), (const, override));
  MOCK_METHOD(StatusOr<std::unique_ptr<TensorAggregator>>, Deserialize,
              (const Intrinsic&, std::string), (const, override));
};

REGISTER_AGGREGATOR_FACTORY("foobar", MockFactory);

TEST(TensorAggregatorRegistryTest, FactoryRegistrationSuccessful) {
  EXPECT_THAT(GetAggregatorFactory("foobar"), IsOk());
  EXPECT_THAT(GetAggregatorFactory("xyz"), StatusIs(NOT_FOUND));
}

TEST(TensorAggregatorRegistryTest, RepeatedRegistrationUnsuccessful) {
  MockFactory factory2;
  EXPECT_DEATH(RegisterAggregatorFactory("foobar", &factory2),
               "already registered");
}

}  // namespace
}  // namespace aggregation
}  // namespace tensorflow_federated
