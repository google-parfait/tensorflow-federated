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

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_PROTOCOL_CHECKPOINT_BUILDER_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_PROTOCOL_CHECKPOINT_BUILDER_H_

#include <memory>
#include <string>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"

namespace tensorflow_federated::aggregation {

// Describes an abstract interface for building and formatting a checkpoint
// from a set of named tensors.
class CheckpointBuilder {
 public:
  virtual ~CheckpointBuilder() = default;

  // Adds a tensor to the checkpoint.
  virtual absl::Status Add(const std::string& name, const Tensor& tensor) = 0;

  // Builds and formats the checkpoint.
  virtual absl::StatusOr<absl::Cord> Build() = 0;
};

// Describes an abstract factory for creating instances of CheckpointBuilder.
class CheckpointBuilderFactory {
 public:
  virtual ~CheckpointBuilderFactory() = default;

  // Creates an instance of CheckpointBuilder.
  virtual std::unique_ptr<CheckpointBuilder> Create() const = 0;
};

}  // namespace tensorflow_federated::aggregation

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_PROTOCOL_CHECKPOINT_BUILDER_H_
