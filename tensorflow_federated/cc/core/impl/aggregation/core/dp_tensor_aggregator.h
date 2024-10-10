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

#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator.h"

namespace tensorflow_federated {
namespace aggregation {

// Virtual class for DP aggregations that do not involve grouping by sensitive
// attributes.
class DPTensorAggregator : public TensorAggregator {
 public:
  // Use -1 as placeholder for missing DP parameters.
  // If they are missing, they should be set before the DP algorithm is invoked.
  explicit DPTensorAggregator(double epsilon = -1, double delta = -1)
      : epsilon_(epsilon), delta_(delta) {}

  double GetEpsilon() const { return epsilon_; }
  double GetDelta() const { return delta_; }

 private:
  double epsilon_, delta_;
};

}  // namespace aggregation
}  // namespace tensorflow_federated

#endif  // TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_DP_TENSOR_AGGREGATOR_H_
