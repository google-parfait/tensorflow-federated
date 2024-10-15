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

#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator.h"

namespace tensorflow_federated {
namespace aggregation {

// Virtual class for DP aggregations that do not involve grouping by sensitive
// attributes.
class DPTensorAggregator : public TensorAggregator {
 public:
  // This member function should be called in lieu of Report(). Given epsilon &
  // delta, it will perform the DP mechanism with those parameters and return
  // the result.
  virtual StatusOr<OutputTensorList> ReportWithEpsilonAndDelta(
      double epsilon, double delta) && = 0;

 protected:
  // TakeOutputs() is deprecated given the use of ReportWithEpsilonAndDelta().
  OutputTensorList TakeOutputs() && override {
    TFF_CHECK(false) << "DPTensorAggregator::TakeOutputs: Not implemented.";
  }
};

}  // namespace aggregation
}  // namespace tensorflow_federated

#endif  // TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_DP_TENSOR_AGGREGATOR_H_
