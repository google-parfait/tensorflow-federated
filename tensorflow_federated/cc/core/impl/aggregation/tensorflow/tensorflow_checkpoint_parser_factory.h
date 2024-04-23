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

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_TENSORFLOW_TENSORFLOW_CHECKPOINT_PARSER_FACTORY_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_TENSORFLOW_TENSORFLOW_CHECKPOINT_PARSER_FACTORY_H_

#include <memory>

#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_parser.h"

namespace tensorflow_federated::aggregation::tensorflow {

// A CheckpointParserFactory implementation that reads TensorFlow checkpoints.
class TensorflowCheckpointParserFactory
    : public tensorflow_federated::aggregation::CheckpointParserFactory {
 public:
  absl::StatusOr<
      std::unique_ptr<tensorflow_federated::aggregation::CheckpointParser>>
  Create(const absl::Cord& serialized_checkpoint) const override;
};

}  // namespace tensorflow_federated::aggregation::tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_TENSORFLOW_TENSORFLOW_CHECKPOINT_PARSER_FACTORY_H_
