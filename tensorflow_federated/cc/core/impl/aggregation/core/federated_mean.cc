/*
 * Copyright 2023 Google LLC
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

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/agg_core.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/agg_vector.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/datatype.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/input_tensor_list.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/intrinsic.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_vector_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator_factory.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator_registry.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_shape.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_spec.h"

namespace tensorflow_federated {
namespace aggregation {

constexpr char kFederatedMeanUri[] = "federated_mean";
constexpr char kFederatedWeightedMeanUri[] = "federated_weighted_mean";

template <typename V, typename W>
class FederatedMean final : public TensorAggregator {
 public:
  explicit FederatedMean(
      DataType dtype, TensorShape shape,
      std::unique_ptr<MutableVectorData<V>> weighted_values_sum)
      : FederatedMean(dtype, shape, std::move(weighted_values_sum), 0, 0) {}

  FederatedMean(DataType dtype, TensorShape shape,
                std::unique_ptr<MutableVectorData<V>> weighted_values_sum,
                W weights_sum, int num_inputs)
      : dtype_(dtype),
        shape_(std::move(shape)),
        weighted_values_sum_(std::move(weighted_values_sum)),
        weights_sum_(weights_sum),
        num_inputs_(num_inputs) {}

  StatusOr<std::string> Serialize() && override {
    FederatedMeanAggregatorState aggregator_state;
    aggregator_state.set_num_inputs(num_inputs_);
    *(aggregator_state.mutable_weighted_values_sum()) =
        weighted_values_sum_->EncodeContent();
    *(aggregator_state.mutable_weights_sum()) = std::string(
        reinterpret_cast<char*>(&weights_sum_), sizeof(weights_sum_));
    return aggregator_state.SerializeAsString();
  }

 private:
  Status MergeWith(TensorAggregator&& other) override {
    TFF_RETURN_IF_ERROR(CheckValid());
    FederatedMean* other_ptr = dynamic_cast<FederatedMean*>(&other);
    if (other_ptr == nullptr) {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "FederatedMean::MergeWith: Can only merge with "
                "another FederatedMean.";
    }
    TFF_RETURN_IF_ERROR((*other_ptr).CheckValid());

    std::pair<std::unique_ptr<MutableVectorData<V>>, W> other_internal_state =
        other_ptr->GetInternalState();
    if (other_internal_state.first->size() != weighted_values_sum_->size()) {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "FederatedMean::MergeWith: Can only merge weighted value sum "
                "tensors of equal length.";
    }

    for (int i = 0; i < weighted_values_sum_->size(); ++i) {
      (*weighted_values_sum_)[i] += (*other_internal_state.first)[i];
    }
    weights_sum_ += other_internal_state.second;
    num_inputs_ += other_ptr->GetNumInputs();
    return TFF_STATUS(OK);
  }

  Status ValidateInputs(const InputTensorList& tensors) const override {
    for (const Tensor* tensor : tensors) {
      if (!tensor->is_dense()) {
        return TFF_STATUS(INVALID_ARGUMENT)
               << "FederatedMean::ValidateInputs: Only dense "
                  "tensors are supported.";
      }
    }

    // If the intrinsic is federated_weighted_mean, the second input tensor
    // should contain a positive scalar weight - check that it is the case.
    if (tensors.size() > 1) {
      if (tensors[1]->num_elements() != 1) {
        return TFF_STATUS(INVALID_ARGUMENT)
               << "FederatedMean::ValidateInputs: The weight must be a "
                  "scalar.";
      }
      AggVector<W> weights = tensors[1]->AsAggVector<W>();
      W weight = weights.begin().value();
      if (weight <= 0) {
        return TFF_STATUS(INVALID_ARGUMENT)
               << "FederatedMean::ValidateInputs: Only positive "
                  "weights are allowed.";
      }
    }
    return TFF_STATUS(OK);
  }

  Status AggregateTensors(InputTensorList tensors) override {
    AggVector<V> values = tensors[0]->AsAggVector<V>();
    if (tensors.size() > 1) {
      AggVector<W> weights = tensors[1]->AsAggVector<W>();
      W weight = weights.begin().value();
      for (auto value : values) {
        (*weighted_values_sum_)[value.index] += value.value * weight;
      }
      weights_sum_ += weight;
    } else {
      for (auto value : values) {
        (*weighted_values_sum_)[value.index] += value.value;
      }
    }
    num_inputs_++;
    return TFF_STATUS(OK);
  }

  Status CheckValid() const override {
    if (output_consumed_) {
      return TFF_STATUS(FAILED_PRECONDITION)
             << "FederatedMean::CheckValid: Output has already been consumed.";
    }
    return TFF_STATUS(OK);
  }

  OutputTensorList TakeOutputs() && override {
    output_consumed_ = true;
    // Produce the final weighted mean values by dividing the weighted values
    // sum by the weights sum (tracked by weights_sum_ in the weighted case and
    // num_inputs_ in the non-weighted case).
    for (int i = 0; i < weighted_values_sum_->size(); ++i) {
      (*weighted_values_sum_)[i] /=
          (weights_sum_ > 0 ? weights_sum_ : num_inputs_);
    }
    OutputTensorList outputs = std::vector<Tensor>();
    outputs.push_back(
        Tensor::Create(dtype_, shape_, std::move(weighted_values_sum_))
            .value());
    return outputs;
  }

  int GetNumInputs() const override { return num_inputs_; }

  std::pair<std::unique_ptr<MutableVectorData<V>>, W> GetInternalState() {
    output_consumed_ = true;
    return std::make_pair(std::move(weighted_values_sum_), weights_sum_);
  }

  bool output_consumed_ = false;
  DataType dtype_;
  TensorShape shape_;
  std::unique_ptr<MutableVectorData<V>> weighted_values_sum_;
  // In the weighted case, use the weights_sum_ variable to track the total
  // weight. Otherwise, just rely on the num_inputs_ variable.
  W weights_sum_;
  int num_inputs_;
};

// Factory class for the FederatedMean.
class FederatedMeanFactory final : public TensorAggregatorFactory {
 public:
  FederatedMeanFactory() = default;

  // FederatedMeanFactory isn't copyable or moveable.
  FederatedMeanFactory(const FederatedMeanFactory&) = delete;
  FederatedMeanFactory& operator=(const FederatedMeanFactory&) = delete;

  StatusOr<std::unique_ptr<TensorAggregator>> Create(
      const Intrinsic& intrinsic) const override {
    return CreateInternal(intrinsic, nullptr);
  }

  StatusOr<std::unique_ptr<TensorAggregator>> Deserialize(
      const Intrinsic& intrinsic, std::string serialized_state) const override {
    FederatedMeanAggregatorState aggregator_state;
    if (!aggregator_state.ParseFromString(serialized_state)) {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "FederatedMeanFactory::Deserialize: Failed to parse "
                "FederatedMeanAggregatorState.";
    }
    return CreateInternal(intrinsic, &aggregator_state);
  }

 private:
  StatusOr<std::unique_ptr<TensorAggregator>> CreateInternal(
      const Intrinsic& intrinsic,
      const FederatedMeanAggregatorState* aggregator_state) const {
    // Check that the configuration is valid.
    if (kFederatedMeanUri == intrinsic.uri) {
      if (intrinsic.inputs.size() != 1) {
        return TFF_STATUS(INVALID_ARGUMENT)
               << "FederatedMeanFactory: Exactly one input is expected for "
                  "federated_mean intrinsic.";
      }
    } else if (kFederatedWeightedMeanUri == intrinsic.uri) {
      if (intrinsic.inputs.size() != 2) {
        return TFF_STATUS(INVALID_ARGUMENT)
               << "FederatedMeanFactory: Exactly two inputs are expected for "
                  "federated_weighted_mean intrinsic.";
      }
    } else {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "FederatedMeanFactory: Expected intrinsic URI "
             << kFederatedMeanUri << " or " << kFederatedWeightedMeanUri
             << " but got uri " << intrinsic.uri;
    }
    if (intrinsic.outputs.size() != 1) {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "FederatedMeanFactory: Exactly one output tensor is expected.";
    }
    if (!intrinsic.nested_intrinsics.empty()) {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "FederatedMeanFactory: Expected no nested intrinsics.";
    }
    if (!intrinsic.parameters.empty()) {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "FederatedMeanFactory: Expected no parameters.";
    }

    const TensorSpec& input_value_spec = intrinsic.inputs[0];
    const TensorSpec& output_spec = intrinsic.outputs[0];

    if (input_value_spec.dtype() != output_spec.dtype() ||
        input_value_spec.shape() != output_spec.shape()) {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "FederatedMeanFactory: Input value tensor and output tensor "
                "have mismatched specs.";
    }
    if (input_value_spec.dtype() != DT_FLOAT &&
        input_value_spec.dtype() != DT_DOUBLE) {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "FederatedMeanFactory: Input value tensor type must be "
                "DT_FLOAT or DT_DOUBLE.";
    }
    StatusOr<size_t> value_num_elements =
        input_value_spec.shape().NumElements();
    if (!value_num_elements.ok()) {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "FederatedMeanFactory: All dimensions of value tensor shape "
                "must be known in advance.";
    }

    DataType input_value_type = input_value_spec.dtype();
    DataType input_weight_type;
    if (kFederatedWeightedMeanUri == intrinsic.uri) {
      input_weight_type = intrinsic.inputs[1].dtype();
      StatusOr<size_t> weight_num_elements =
          intrinsic.inputs[1].shape().NumElements();
      if (!weight_num_elements.ok() || weight_num_elements.value() != 1) {
        return TFF_STATUS(INVALID_ARGUMENT)
               << "FederatedMeanFactory: The weight must be a scalar.";
      }
    } else {
      input_weight_type = DT_INT32;
    }

    std::unique_ptr<TensorAggregator> aggregator;
    if (aggregator_state == nullptr) {
      FLOATING_ONLY_DTYPE_CASES(
          input_value_type, V,
          NUMERICAL_ONLY_DTYPE_CASES(
              input_weight_type, W,
              aggregator = (std::make_unique<FederatedMean<V, W>>(
                  input_value_type, input_value_spec.shape(),
                  std::make_unique<MutableVectorData<V>>(
                      value_num_elements.value())))));
      return aggregator;
    }

    FLOATING_ONLY_DTYPE_CASES(
        input_value_type, V,
        NUMERICAL_ONLY_DTYPE_CASES(
            input_weight_type, W,
            aggregator = (std::make_unique<FederatedMean<V, W>>(
                input_value_type, input_value_spec.shape(),
                MutableVectorData<V>::CreateFromEncodedContent(
                    aggregator_state->weighted_values_sum()),
                *(reinterpret_cast<const W*>(
                    aggregator_state->weights_sum().data())),
                aggregator_state->num_inputs()))));
    return aggregator;
  }
};

static auto unused = ::tensorflow_federated::aggregation::internal::Registrar<
    FederatedMeanFactory>(kFederatedMeanUri);
static auto unused_weighted =
    ::tensorflow_federated::aggregation::internal::Registrar<
        FederatedMeanFactory>(kFederatedWeightedMeanUri);

}  // namespace aggregation
}  // namespace tensorflow_federated
