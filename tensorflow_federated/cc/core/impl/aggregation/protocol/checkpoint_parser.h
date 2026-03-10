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

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_PROTOCOL_CHECKPOINT_PARSER_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_PROTOCOL_CHECKPOINT_PARSER_H_

#include <memory>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"

namespace tensorflow_federated::aggregation {

// Describes an abstract interface for parsing a checkpoint from a blob
// and returning a set of named tensors.
class CheckpointParser {
 public:
  virtual ~CheckpointParser() = default;

  // Gets a tensor by name.
  // Note that depending on the implementation, this method may be destructive,
  // i.e. it may not be valid to call this method more than once for the same
  // tensor.
  virtual absl::StatusOr<Tensor> GetTensor(const std::string& name) = 0;

  // Loads all tensors from the checkpoint.
  // Note that depending on the implementation, this method may be destructive,
  // i.e. it may not be valid to call this method more than once.
  // For backward compatibility, by default, this is a no-op that returns
  // NotImplemented error.
  virtual absl::StatusOr<absl::flat_hash_map<std::string, Tensor>>
  LoadAllTensors() {
    return absl::UnimplementedError("LoadAllTensors not implemented.");
  }
};

// Describes an abstract factory for creating instances of CheckpointParser.
class CheckpointParserFactory {
 public:
  virtual ~CheckpointParserFactory() = default;

  // Creates an instance of CheckpointParser with the provided serialized
  // checkpoint content.
  virtual absl::StatusOr<std::unique_ptr<CheckpointParser>> Create(
      const absl::Cord& serialized_checkpoint) const = 0;
};

}  // namespace tensorflow_federated::aggregation

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_PROTOCOL_CHECKPOINT_PARSER_H_
