// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_builder.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_builder.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_header.h"

namespace tensorflow_federated::aggregation {

namespace {
// Builds and formats a set of aggregation tensors using the new wire format for
// federated compute.
class FederatedComputeCheckpointBuilder final : public CheckpointBuilder {
 public:
  FederatedComputeCheckpointBuilder() {
    // Indicates that the checkpoint is using the new wire format.
    result_.Append(kFederatedComputeCheckpointHeader);
  }
  // Disallow copy and move constructors.
  FederatedComputeCheckpointBuilder(const FederatedComputeCheckpointBuilder&) =
      delete;
  FederatedComputeCheckpointBuilder& operator=(
      const FederatedComputeCheckpointBuilder&) = delete;

  absl::Status Add(const std::string& name, const Tensor& tensor) override {
    std::string metadata;
    google::protobuf::io::StringOutputStream out(&metadata);
    google::protobuf::io::CodedOutputStream coded_out(&out);
    coded_out.WriteVarint64(name.size());
    coded_out.WriteString(name);

    absl::Cord content(tensor.ToProto().SerializeAsString());
    if (content.empty()) {
      return absl::InternalError("Failed to add tensor for " + name);
    }
    coded_out.WriteVarint64(content.size());
    coded_out.Trim();
    result_.Append(metadata);
    result_.Append(content);
    return absl::OkStatus();
  }

  absl::StatusOr<absl::Cord> Build() override {
    uint32_t zero = 0;
    result_.Append(
        absl::string_view(reinterpret_cast<const char*>(&zero), sizeof(zero)));
    return result_;
  }

 private:
  absl::Cord result_;
};

}  // namespace

std::unique_ptr<CheckpointBuilder>
FederatedComputeCheckpointBuilderFactory::Create() const {
  return std::make_unique<FederatedComputeCheckpointBuilder>();
}

}  // namespace tensorflow_federated::aggregation
