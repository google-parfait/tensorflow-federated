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

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_TENSOR_AGGREGATOR_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_TENSOR_AGGREGATOR_H_

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/agg_core.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/input_tensor_list.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"

namespace tensorflow_federated {
namespace aggregation {

using OutputTensorList = std::vector<Tensor>;

// TensorAggregator is a base class for implementing Aggregation intrinsics
// with Tensor being an input and output type for the aggregation.
class TensorAggregator
    : public Aggregator<InputTensorList, OutputTensorList, TensorAggregator> {
 public:
  ~TensorAggregator() override = default;

  // Callers should only use this method if accumulate metadata is not needed.
  // Most non-test callers should use the Accumulate method with the metadata
  // parameter.
  Status Accumulate(InputTensorList tensors) {
    return Accumulate(std::move(tensors), /*metadata=*/std::nullopt);
  }

  // Implementation of the base Aggregator class methods.
  Status Accumulate(InputTensorList tensors,
                    std::optional<AccumulateMetadata> metadata) override;
  bool CanReport() const override;
  StatusOr<OutputTensorList> Report() && override;

  // Returns the number of aggregated inputs.
  virtual int GetNumInputs() const = 0;

  // Serialize the internal state of the TensorAggregator as a string.
  virtual StatusOr<std::string> Serialize() && = 0;

 protected:
  // Construct TensorAggregator
  explicit TensorAggregator() {}

  // A derived class should override either this method or the one below, but
  // not both. It should override the one with a metadata parameter if it wants
  // to use the metadata.
  virtual Status AggregateTensors(InputTensorList tensors,
                                  std::optional<AccumulateMetadata> metadata) {
    return AggregateTensors(std::move(tensors));
  }
  virtual Status AggregateTensors(InputTensorList tensors) {
    return absl::UnimplementedError(
        "No implementation of AggregateTensors provided.");
  }

  // Checks if the current TensorAggregator is valid e.g. the resulting output
  // hasn't been consumed.
  virtual Status CheckValid() const = 0;

  // Consumes the output of this TensorAggregator.
  virtual OutputTensorList TakeOutputs() && = 0;

 private:
  // Extracts the aggregated tensor and makes the current aggregator "consumed".
  OutputTensorList TakeTensors() &&;
};

}  // namespace aggregation
}  // namespace tensorflow_federated

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_TENSOR_AGGREGATOR_H_
