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

#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_quantile_aggregator.h"

#include <memory>
#include <string>
#include <vector>

#include "absl/random/random.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/agg_core.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/datatype.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_fedsql_constants.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/input_tensor_list.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/intrinsic.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator_registry.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_spec.h"

namespace tensorflow_federated {
namespace aggregation {

template <typename T>
inline void DPQuantileAggregator<T>::InsertWithReservoirSampling(T value) {
  int index = absl::Uniform(bit_gen_, 0, num_inputs_);
  if (index < buffer_.size()) {
    buffer_[index] = value;
  }
}

// To merge, we insert up to capacity and then perform reservoir sampling.
template <typename T>
Status DPQuantileAggregator<T>::MergeWith(TensorAggregator&& other) {
  // Ensure that the other aggregator is of the same type.
  // Then use std::vector<T>::insert
  return TFF_STATUS(UNIMPLEMENTED) << "Will be implemented in a follow-up CL.";
}

template <typename T>
StatusOr<std::string> DPQuantileAggregator<T>::Serialize() && {
  return TFF_STATUS(UNIMPLEMENTED) << "Will be implemented in a follow-up CL.";
}

// Push back the input into the buffer or perform reservoir sampling.
template <typename T>
Status DPQuantileAggregator<T>::AggregateTensors(InputTensorList tensors) {
  TFF_RETURN_IF_ERROR(CheckValid());
  // Ensure that there is exactly one tensor.
  if (tensors.size() != 1) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "DPQuantileAggregator::AggregateTensors: Expected exactly one "
              "tensor, but got "
           << tensors.size();
  }
  // Ensure that the tensor only has one element.
  auto num_elements_in_tensor = tensors[0]->num_elements();
  if (num_elements_in_tensor != 1) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "DPQuantileAggregator::AggregateTensors: Expected a scalar "
              "tensor, but got a tensor with "
           << num_elements_in_tensor << " elements.";
  }

  // Ensure that the tensor is of the correct type.
  DataType dtype = tensors[0]->dtype();
  DataType expected_dtype = internal::TypeTraits<T>::kDataType;
  if (dtype != expected_dtype) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "DPQuantileAggregator::AggregateTensors: Expected a "
           << DataType_Name(expected_dtype) << " tensor, but got a "
           << DataType_Name(dtype) << " tensor.";
  }

  num_inputs_++;
  T value = tensors[0]->CastToScalar<T>();
  if (buffer_.size() < kDPQuantileMaxInputs) {
    buffer_.push_back(value);
  } else {
    InsertWithReservoirSampling(value);
  }

  return TFF_STATUS(OK);
}

// Checks if the output has not already been consumed.
template <typename T>
Status DPQuantileAggregator<T>::CheckValid() const {
  if (output_consumed_) {
    return TFF_STATUS(FAILED_PRECONDITION)
           << "DPQuantileAggregator::CheckValid: Output has already been "
              "consumed.";
  }
  return TFF_STATUS(OK);
}

// Trigger execution of the DP quantile algorithm.
template <typename T>
StatusOr<OutputTensorList> DPQuantileAggregator<T>::ReportWithEpsilonAndDelta(
    double epsilon, double delta) && {
  return TFF_STATUS(UNIMPLEMENTED) << "Will be implemented in a follow-up CL.";
}

StatusOr<std::unique_ptr<TensorAggregator>>
DPQuantileAggregatorFactory::Deserialize(const Intrinsic& intrinsic,
                                         std::string serialized_state) const {
  return TFF_STATUS(UNIMPLEMENTED) << "Will be implemented in a follow-up CL.";
}

// The Create method of the DPQuantileAggregatorFactory.
StatusOr<std::unique_ptr<TensorAggregator>> DPQuantileAggregatorFactory::Create(
    const Intrinsic& intrinsic) const {
  // First check that the parameter field has a valid target_quantile and
  // nothing else.
  if (intrinsic.parameters.size() != 1) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "DPQuantileAggregatorFactory::Create: Expected exactly one "
              "parameter, but got "
           << intrinsic.parameters.size();
  }

  auto& param = intrinsic.parameters[0];
  if (param.dtype() != DT_DOUBLE) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "DPQuantileAggregatorFactory::Create: Expected a double for the"
              " `target_quantile` parameter of DPQuantileAggregator, but got "
           << DataType_Name(param.dtype());
  }
  double target_quantile = param.CastToScalar<double>();
  if (target_quantile <= 0 || target_quantile >= 1) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "DPQuantileAggregatorFactory::Create: Target quantile must be "
              "in (0, 1).";
  }

  // Next, get the input and output types.
  const TensorSpec& input_spec = intrinsic.inputs[0];
  DataType input_type = input_spec.dtype();
  const TensorSpec& output_spec = intrinsic.outputs[0];
  DataType output_type = output_spec.dtype();

  // Quantile is only defined for numeric input types.
  if (internal::GetTypeKind(input_type) != internal::TypeKind::kNumeric) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "DPQuantileAggregatorFactory::Create: DPQuantileAggregator only "
              "supports numeric datatypes.";
  }

  // To adhere to existing specifications, the output must be a double.
  if (output_type != DT_DOUBLE) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "DPQuantileAggregatorFactory::Create: Output type must be "
              "double.";
  }
  DTYPE_CASES(
      input_type, T,
      return std::make_unique<DPQuantileAggregator<T>>(target_quantile));
}

REGISTER_AGGREGATOR_FACTORY(kDPQuantileUri, DPQuantileAggregatorFactory);

}  // namespace aggregation
}  // namespace tensorflow_federated
