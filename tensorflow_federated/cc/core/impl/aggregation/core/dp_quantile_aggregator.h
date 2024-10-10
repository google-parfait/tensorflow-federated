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

#ifndef TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_DP_QUANTILE_AGGREGATOR_H_
#define TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_DP_QUANTILE_AGGREGATOR_H_

#include <memory>
#include <string>
#include <vector>

#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/agg_core.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_tensor_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/input_tensor_list.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/intrinsic.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator_factory.h"

namespace tensorflow_federated {
namespace aggregation {

// DPQuantileAggregator is a DPTensorAggregator that computes a quantile of
// input scalars, in a differentially private manner.
// It stores the inputs in a buffer. When TakeOutputs() is called, it will sort
// the buffer and then employ the `PrivateQuantile` algorithm by Smith
// See https://cs-people.bu.edu/ads22/pubs/2011/stoc194-smith.pdf.
// The paper's analysis implicitly assumes access to an ideal functionality that
// selects a random interval according to a bespoke probability distribution.
// We implement this selection step by approximating Gumbel noise; the error of
// the approximation is captured by a nonzero delta parameter.
class DPQuantileAggregator final : public DPTensorAggregator {
 public:
  explicit DPQuantileAggregator(double target_quantile)
      : DPTensorAggregator(),
        target_quantile_(target_quantile),
        num_inputs_(0),
        buffer_() {
    TFF_CHECK(target_quantile > 0 && target_quantile < 1)
        << "Target quantile must be in (0, 1).";
  }

  int GetNumInputs() const override { return num_inputs_; }

  // To MergeWith another DPQuantileAggregator, we simply append the buffer of
  // the other aggregator to our buffer.
  Status MergeWith(TensorAggregator&& other) override;

  StatusOr<std::string> Serialize() && override;

 protected:
  // This DP mechanism will require a delta larger than 0 and less than 1.
  bool ValidDelta(double delta) const override;

  // This DP mechanism expects one scalar tensor in the input. It pushes the
  // scalar into the buffer if the buffer is smaller than kDPQuantileMaxInputs.
  // Otherwise, it will perform reservoir sampling
  Status AggregateTensors(InputTensorList tensors) override;

  // Checks if it is possible to run the DP quantile algorithm: is there data
  // and do we have valid DP parameters?
  Status CheckValid() const override;

  // Consumes the output of this DPQuantileAggregator. DP quantile algorithm
  // will be called here.
  OutputTensorList TakeOutputs() && override;

 private:
  double target_quantile_;
  int num_inputs_;
  std::vector<double> buffer_;
};

// This factory class expects only one parameter in the input intrinsic: the
// target quantile.
class DPQuantileAggregatorFactory final : public TensorAggregatorFactory {
 public:
  DPQuantileAggregatorFactory() = default;

  // DPQuantileAggregatorFactory isn't copyable or moveable.
  DPQuantileAggregatorFactory(const DPQuantileAggregatorFactory&) = delete;
  DPQuantileAggregatorFactory& operator=(const DPQuantileAggregatorFactory&) =
      delete;

  StatusOr<std::unique_ptr<TensorAggregator>> Deserialize(
      const Intrinsic& intrinsic, std::string serialized_state) const override;

  StatusOr<std::unique_ptr<TensorAggregator>> Create(
      const Intrinsic& intrinsic) const override;
};
}  // namespace aggregation
}  // namespace tensorflow_federated

#endif  // TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_DP_QUANTILE_AGGREGATOR_H_
