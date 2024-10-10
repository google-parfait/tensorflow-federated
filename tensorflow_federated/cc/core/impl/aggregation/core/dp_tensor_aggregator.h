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
  // Use -1 as placeholder for missing DP parameters.
  // If they are missing, they should be set before the DP algorithm is invoked.
  explicit DPTensorAggregator(double epsilon = -1, double delta = -1)
      : epsilon_(epsilon), delta_(delta) {}

  // Basic accessors for the DP parameters.
  double GetEpsilon() const { return epsilon_; }
  double GetDelta() const { return delta_; }

  // Boolean checks of validity of DP parameters.
  // Because different mechanisms have different requirements on delta, the
  // behavior of HaveValidDelta is delegated to that of ValidDelta, which is a
  // method that must be implemented by each child class.
  bool HaveValidEpsilon() const { return epsilon_ > 0.0; }
  bool HaveValidDelta() const { return ValidDelta(delta_); }
  bool HaveValidEpsilonAndDelta() const {
    return HaveValidEpsilon() && HaveValidDelta();
  }

  // Setters for DP parameters. If the provided value is invalid, return an
  // INVALID_ARGUMENT status.
  Status SetEpsilon(double epsilon) {
    if (epsilon <= 0.0) {
      return TFF_STATUS(INVALID_ARGUMENT) << "DPTensorAggregator::SetEpsilon: "
                                          << "Epsilon must be positive.";
    }
    epsilon_ = epsilon;
    return TFF_STATUS(OK);
  }

  Status SetDelta(double delta) {
    if (!ValidDelta(delta)) {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "DPTensorAggregator::SetDelta: "
             << "Provided delta is invalid: " << delta;
    }
    delta_ = delta;
    return TFF_STATUS(OK);
  }

 protected:
  // Internal method that checks the validity of the delta parameter.
  virtual bool ValidDelta(double delta) const = 0;

 private:
  double epsilon_, delta_;
};

}  // namespace aggregation
}  // namespace tensorflow_federated

#endif  // TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_DP_TENSOR_AGGREGATOR_H_
