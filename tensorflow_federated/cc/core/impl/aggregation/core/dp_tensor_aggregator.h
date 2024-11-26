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

#ifndef TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_DP_TENSOR_AGGREGATOR_H_
#define TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_DP_TENSOR_AGGREGATOR_H_

#include <utility>
#include <vector>

#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/input_tensor_list.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_spec.h"

namespace tensorflow_federated {
namespace aggregation {

// Virtual class for DP aggregations that do not involve grouping by sensitive
// attributes.
class DPTensorAggregator : public TensorAggregator {
 public:
  explicit DPTensorAggregator(const std::vector<TensorSpec>& input_specs)
      : input_specs_(input_specs) {}

  // This member function should be called in lieu of Report(). Given epsilon &
  // delta, it will perform the DP mechanism with those parameters and return
  // the result.
  virtual StatusOr<OutputTensorList> ReportWithEpsilonAndDelta(
      double epsilon, double delta) && = 0;

  // Verify that the input tensors match the member specifications.
  // Called within DPTensorAggregator::AggregateTensors(). Also called by
  // DPTensorAggregatorBundle::AggregateTensors(), to check all inputs before
  // passing them to the child aggregators.
  Status InputMatchesSpec(const InputTensorList& input) const;

 protected:
  // TakeOutputs() is deprecated given the use of ReportWithEpsilonAndDelta().
  OutputTensorList TakeOutputs() && override {
    TFF_CHECK(false) << "DPTensorAggregator::TakeOutputs: Not implemented.";
  }

  // Child-specific implementation of AggregateTensors().
  virtual Status AggregateTensorsInternal(InputTensorList tensors) = 0;

  // Verify that the input tensors match the expected specs and then call
  // AggregateTensorsInternal().
  Status AggregateTensors(InputTensorList tensors) override {
    TFF_RETURN_IF_ERROR(InputMatchesSpec(tensors));
    return AggregateTensorsInternal(std::move(tensors));
  }

 private:
  std::vector<TensorSpec> input_specs_;
};

}  // namespace aggregation
}  // namespace tensorflow_federated

#endif  // TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_DP_TENSOR_AGGREGATOR_H_
