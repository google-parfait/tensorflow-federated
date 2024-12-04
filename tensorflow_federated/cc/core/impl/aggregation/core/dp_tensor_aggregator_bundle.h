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

#ifndef TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_DP_TENSOR_AGGREGATOR_BUNDLE_H_
#define TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_DP_TENSOR_AGGREGATOR_BUNDLE_H_

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

// Wrapper around multiple DPTensorAggregators. Used when there are no groups
// to keep private.
class DPTensorAggregatorBundle final : public TensorAggregator {
 public:
  explicit DPTensorAggregatorBundle(
      std::vector<std::unique_ptr<DPTensorAggregator>> aggregators,
      std::vector<int> num_tensors_per_agg, double epsilon_per_agg,
      double delta_per_agg, int num_inputs = 0);

  // Returns the number of aggregated inputs.
  int GetNumInputs() const override { return num_inputs_; };

  // Serialize the internal state of the TensorAggregator as a string.
  StatusOr<std::string> Serialize() && override;

  inline double GetEpsilonPerAgg() const { return epsilon_per_agg_; }
  inline double GetDeltaPerAgg() const { return delta_per_agg_; }

  Status IsCompatible(const TensorAggregator& other) const;

  Status MergeWith(TensorAggregator&& other) override;

 protected:
  Status AggregateTensors(InputTensorList tensors) override;

  // Checks if the current TensorAggregator is valid e.g. the resulting output
  // hasn't been consumed.
  Status CheckValid() const override {
    if (output_consumed_) {
      return TFF_STATUS(FAILED_PRECONDITION)
             << "DPTensorAggregatorBundle::CheckValid: Output has already been "
                "consumed.";
    }
    return TFF_STATUS(OK);
  }

  // Consumes the output of this TensorAggregator. Calls the
  // ReportWithEpsilonAndDelta() method of the underlying aggregators.
  OutputTensorList TakeOutputs() && override { return OutputTensorList(); }

 private:
  std::vector<std::unique_ptr<DPTensorAggregator>> aggregators_;

  // Each nested aggregator's Accumulate() method may expect a different
  // number of tensors in its given InputTensorList; this vector stores all
  // those numbers so that we can split the input accordingly.
  std::vector<int> num_tensors_per_agg_;

  // The sum over all num_tensors_per_agg_.
  int num_tensors_per_input_;

  // Budget for each nested aggregator.
  double epsilon_per_agg_, delta_per_agg_;

  // Number of times Accumulate() has been successfully called so far.
  int num_inputs_;

  bool output_consumed_ = false;
};

class DPTensorAggregatorBundleFactory : public TensorAggregatorFactory {
 public:
  DPTensorAggregatorBundleFactory() = default;

  // DPQuantileAggregatorFactory isn't copyable or moveable.
  DPTensorAggregatorBundleFactory(const DPTensorAggregatorBundleFactory&) =
      delete;
  DPTensorAggregatorBundleFactory& operator=(
      const DPTensorAggregatorBundleFactory&) = delete;

  StatusOr<std::unique_ptr<TensorAggregator>> Deserialize(
      const Intrinsic& intrinsic, std::string serialized_state) const override;

  StatusOr<std::unique_ptr<TensorAggregator>> Create(
      const Intrinsic& intrinsic) const override {
    return CreateInternal(intrinsic, nullptr);
  }

 private:
  StatusOr<std::unique_ptr<TensorAggregator>> CreateInternal(
      const Intrinsic& intrinsic,
      const DPTensorAggregatorBundleState* aggregator_state) const;
};

}  // namespace aggregation
}  // namespace tensorflow_federated
#endif  // TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_DP_TENSOR_AGGREGATOR_BUNDLE_H_
