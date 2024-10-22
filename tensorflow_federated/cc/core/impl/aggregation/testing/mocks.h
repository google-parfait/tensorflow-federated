/*
 * Copyright 2024 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_TESTING_MOCKS_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_TESTING_MOCKS_H_

#include <cstdint>
#include <memory>
#include <string>

#include "googlemock/include/gmock/gmock.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_builder.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_parser.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/resource_resolver.h"

namespace tensorflow_federated::aggregation {

class MockCheckpointParser : public CheckpointParser {
 public:
  MOCK_METHOD(absl::StatusOr<Tensor>, GetTensor, (const std::string& name),
              (override));
};

class MockCheckpointParserFactory : public CheckpointParserFactory {
 public:
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<CheckpointParser>>, Create,
              (const absl::Cord& serialized_checkpoint), (const, override));
};

class MockCheckpointBuilder : public CheckpointBuilder {
 public:
  MOCK_METHOD(absl::Status, Add,
              (const std::string& name, const Tensor& tensor), (override));
  MOCK_METHOD(absl::StatusOr<absl::Cord>, Build, (), (override));
};

class MockCheckpointBuilderFactory : public CheckpointBuilderFactory {
 public:
  MOCK_METHOD(std::unique_ptr<CheckpointBuilder>, Create, (),
              (const, override));
};

class MockResourceResolver : public ResourceResolver {
 public:
  MOCK_METHOD(absl::StatusOr<absl::Cord>, RetrieveResource,
              (int64_t client_id, const std::string& uri), (override));
};

}  // namespace tensorflow_federated::aggregation

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_TESTING_MOCKS_H_
