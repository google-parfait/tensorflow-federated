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

#include <string>
#include <vector>

#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
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

  // Check all required invariants on the input tensors, so that we can avoid
  // changing state via Accumulate if they are invalid.
  virtual Status ValidateInputs(const InputTensorList& tensors) const;

  // Implementation of the base Aggregator class methods.
  Status Accumulate(InputTensorList tensors) override;
  bool CanReport() const override;
  StatusOr<OutputTensorList> Report() && override;

  // Returns the number of aggregated inputs.
  virtual int GetNumInputs() const = 0;

  // Serializes the internal state of the TensorAggregator as a string.
  virtual StatusOr<std::string> Serialize() && = 0;

  // Partitions the internal state of the TensorAggregator and serializes them
  // as a vector of strings.
  virtual StatusOr<std::vector<std::string>> Partition(int num_partitions) &&;

 protected:
  // Construct TensorAggregator
  explicit TensorAggregator() {}

  // The actual implementation of the tensor aggregation to be provided by
  // a derived class.
  virtual Status AggregateTensors(InputTensorList tensors) = 0;

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
