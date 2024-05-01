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

#include <memory>
#include <string>
#include <utility>

#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/agg_core.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/agg_vector_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/datatype.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/intrinsic.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_vector_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator_factory.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator_registry.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_shape.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_spec.h"

namespace tensorflow_federated {
namespace aggregation {

constexpr char kFederatedSumUri[] = "federated_sum";

// Implementation of a generic sum aggregator.
template <typename T>
class FederatedSum final : public AggVectorAggregator<T> {
 public:
  using AggVectorAggregator<T>::AggVectorAggregator;
  using AggVectorAggregator<T>::data;

 private:
  void AggregateVector(const AggVector<T>& agg_vector) override {
    for (auto v : agg_vector) {
      data()[v.index] += v.value;
    }
  }
};

// Factory class for the FederatedSum.
class FederatedSumFactory final : public TensorAggregatorFactory {
 public:
  FederatedSumFactory() = default;

  // FederatedSumFactory isn't copyable or moveable.
  FederatedSumFactory(const FederatedSumFactory&) = delete;
  FederatedSumFactory& operator=(const FederatedSumFactory&) = delete;

  StatusOr<std::unique_ptr<TensorAggregator>> Create(
      const Intrinsic& intrinsic) const override {
    return CreateInternal(intrinsic, nullptr);
  }

  StatusOr<std::unique_ptr<TensorAggregator>> Deserialize(
      const Intrinsic& intrinsic, std::string serialized_state) const override {
    AggVectorAggregatorState aggregator_state;
    if (!aggregator_state.ParseFromString(serialized_state)) {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "FederatedSumFactory: Failed to deserialize the "
                "AggVectorAggregatorState.";
    }
    return CreateInternal(intrinsic, &aggregator_state);
  };

 private:
  StatusOr<std::unique_ptr<TensorAggregator>> CreateInternal(
      const Intrinsic& intrinsic,
      const AggVectorAggregatorState* aggregator_state) const {
    // Check that the configuration is valid for federated_sum.
    if (kFederatedSumUri != intrinsic.uri) {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "FederatedSumFactory: Expected intrinsic URI "
             << kFederatedSumUri << " but got uri " << intrinsic.uri;
    }
    if (intrinsic.inputs.size() != 1) {
      return TFF_STATUS(INVALID_ARGUMENT) << "FederatedSumFactory: Exactly one "
                                             "input is expected.";
    }
    if (intrinsic.outputs.size() != 1) {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "FederatedSumFactory: Exactly one output tensor is expected.";
    }
    if (!intrinsic.nested_intrinsics.empty()) {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "FederatedSumFactory: Expected no nested intrinsics.";
    }
    if (!intrinsic.parameters.empty()) {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "FederatedSumFactory: Expected no parameters.";
    }

    const TensorSpec& input_spec = intrinsic.inputs[0];
    const TensorSpec& output_spec = intrinsic.outputs[0];

    if (input_spec.dtype() != output_spec.dtype() ||
        input_spec.shape() != output_spec.shape()) {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "FederatedSumFactory: Input and output tensors have mismatched "
                "specs.";
    }
    std::unique_ptr<TensorAggregator> aggregator;
    if (aggregator_state == nullptr) {
      NUMERICAL_ONLY_DTYPE_CASES(
          input_spec.dtype(), T,
          aggregator = std::make_unique<FederatedSum<T>>(
              input_spec.dtype(), std::move(input_spec.shape())));
      return aggregator;
    }

    NUMERICAL_ONLY_DTYPE_CASES(
        input_spec.dtype(), T,
        aggregator = std::make_unique<FederatedSum<T>>(
            input_spec.dtype(), std::move(input_spec.shape()),
            MutableVectorData<T>::CreateFromEncodedContent(
                aggregator_state->vector_data()),
            aggregator_state->num_inputs()));
    return aggregator;
  }
};

REGISTER_AGGREGATOR_FACTORY(kFederatedSumUri, FederatedSumFactory);

}  // namespace aggregation
}  // namespace tensorflow_federated
