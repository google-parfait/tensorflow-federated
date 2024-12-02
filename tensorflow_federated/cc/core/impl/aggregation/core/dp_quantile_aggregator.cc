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

#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/random/random.h"
#include "algorithms/numerical-mechanisms.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/agg_core.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/datatype.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_fedsql_constants.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/input_tensor_list.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/intrinsic.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_vector_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator_registry.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_spec.h"

namespace tensorflow_federated {
namespace aggregation {
// Wrapping around std::ceil to return an int.
int int_ceil(double val) { return static_cast<int>(std::ceil(val)); }

template <typename T>
inline void DPQuantileAggregator<T>::InsertWithReservoirSampling(T value) {
  int index = absl::Uniform(bit_gen_, 0, num_inputs_);
  if (index < buffer_.size()) {
    buffer_[index] = value;
  }
  reservoir_sampling_count_++;
}

template <typename T>
Status DPQuantileAggregator<T>::IsCompatible(
    const TensorAggregator& other) const {
  auto* other_ptr = dynamic_cast<const DPQuantileAggregator<T>*>(&other);
  if (other_ptr == nullptr) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "DPQuantileAggregator::IsCompatible: Can only merge with "
              "another DPQuantileAggregator of the same input type.";
  }

  // Ensure that the other aggregator has the same target quantile.
  if (target_quantile_ != other_ptr->target_quantile_) {
    return TFF_STATUS(INVALID_ARGUMENT) << "DPQuantileAggregator::IsCompatible:"
                                           " Target quantiles must match.";
  }

  return TFF_STATUS(OK);
}

// To merge, we insert up to capacity and then perform reservoir sampling.
template <typename T>
Status DPQuantileAggregator<T>::MergeWith(TensorAggregator&& other) {
  // Check validity and compatibility.
  TFF_RETURN_IF_ERROR(CheckValid());
  TFF_RETURN_IF_ERROR(IsCompatible(other));
  auto* other_ptr = dynamic_cast<const DPQuantileAggregator<T>*>(&other);
  TFF_CHECK(other_ptr != nullptr);
  TFF_RETURN_IF_ERROR(other_ptr->CheckValid());

  // Then use std::vector<T>::insert to copy as much as possible to our buffer.
  int remaining_capacity = kDPQuantileMaxInputs - buffer_.size();
  int other_buffer_size = other_ptr->GetBufferSize();
  int num_to_insert = remaining_capacity < other_buffer_size
                          ? remaining_capacity
                          : other_buffer_size;
  buffer_.insert(buffer_.end(), other_ptr->buffer_.begin(),
                 other_ptr->buffer_.begin() + num_to_insert);

  // For any remaining elements, call InsertWithReservoirSampling.
  auto itr = other_ptr->buffer_.begin() + num_to_insert;
  while (itr != other_ptr->buffer_.end()) {
    InsertWithReservoirSampling(*itr);
    itr++;
  }

  num_inputs_ += other_ptr->GetNumInputs();
  reservoir_sampling_count_ += other_ptr->GetReservoirSamplingCount();

  return TFF_STATUS(OK);
}

// Push back the input into the buffer or perform reservoir sampling.
template <typename T>
Status DPQuantileAggregator<T>::AggregateTensorsInternal(
    InputTensorList tensors) {
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
  if (buffer_.size() > kDPQuantileMaxInputs) {
    return TFF_STATUS(FAILED_PRECONDITION)
           << "DPQuantileAggregator::CheckValid: Buffer size is "
           << buffer_.size() << " which is greater than capacity "
           << kDPQuantileMaxInputs << ".";
  }
  if (output_consumed_) {
    return TFF_STATUS(FAILED_PRECONDITION)
           << "DPQuantileAggregator::CheckValid: Output has already been "
              "consumed.";
  }
  return TFF_STATUS(OK);
}

// Create an OutputTensorList containing a single scalar tensor.
StatusOr<OutputTensorList> SingleScalarTensor(double value) {
  auto data_container = std::make_unique<MutableVectorData<double>>();
  data_container->push_back(value);
  TFF_ASSIGN_OR_RETURN(
      auto tensor, Tensor::Create(DT_DOUBLE, {}, std::move(data_container)));
  OutputTensorList output;
  output.push_back(std::move(tensor));
  return output;
}

// Trigger execution of the DP quantile algorithm.
template <typename T>
StatusOr<OutputTensorList> DPQuantileAggregator<T>::ReportWithEpsilonAndDelta(
    double epsilon, double delta) && {
  TFF_RETURN_IF_ERROR(CheckValid());

  // When epsilon is above the threshold, noiselessly return the quantile.
  if (epsilon >= kEpsilonThreshold) {
    std::sort(buffer_.begin(), buffer_.end());
    int quantile_rank = static_cast<int>(GetTargetRank());
    double raw_quantile = static_cast<double>(buffer_[quantile_rank]);
    return SingleScalarTensor(raw_quantile);
  }

  // Make a histogram of buffer_'s values.
  absl::flat_hash_map<int, int> histogram;
  for (T& element : buffer_) {
    // Identify the bucket that the element belongs to.
    int bucket = GetBucket(element);

    if (histogram.contains(bucket)) {
      histogram[bucket]++;
    } else {
      histogram[bucket] = 1;
    }
  }

  // Calculate the rank of the target quantile in the buffer. It will serve as
  // the basis for a noisy threshold.
  auto target_rank = GetTargetRank();

  // Calculate the maximum bucket that we will consider.
  auto max_bucket = GetBucket(kDPQuantileMaxOutputMagnitude);

  // Get a bucket from PrefixSumAboveThreshold.
  TFF_ASSIGN_OR_RETURN(
      int quantile_bucket,
      PrefixSumAboveThreshold(epsilon, histogram, target_rank, max_bucket));

  // Get the quantile estimate from the bucket.
  auto quantile_estimate = BucketUpperBound(quantile_bucket);

  return SingleScalarTensor(quantile_estimate);
}

template <>
StatusOr<OutputTensorList>
DPQuantileAggregator<string_view>::ReportWithEpsilonAndDelta(double epsilon,
                                                             double delta) && {
  return TFF_STATUS(UNIMPLEMENTED)
         << "DPQuantileAggregator::ReportWithEpsilonAndDelta: string_view is"
            "not a supported type.";
}

// PrefixSumAboveThreshold iterates over histogram buckets and stops when a
// private prefix sum exceeds a noisy version of a given threshold.
template <typename T>
StatusOr<int> DPQuantileAggregator<T>::PrefixSumAboveThreshold(
    double epsilon, absl::flat_hash_map<int, int>& histogram, double threshold,
    int max_bucket) {
  // All estimates will come from the same DP mechanism, as we are answering
  // 1-sensitive counting queries that monotonically increase.
  differential_privacy::LaplaceMechanism::Builder builder;
  builder.SetL1Sensitivity(1.0);
  builder.SetEpsilon(epsilon / 2);
  TFF_ASSIGN_OR_RETURN(auto mechanism, builder.Build());

  double noisy_threshold = mechanism->AddNoise(threshold);

  int prefix_sum = 0;
  int bucket = 0;
  for (; bucket <= max_bucket; ++bucket) {
    // Update prefix_sum by consulting the histogram.
    if (histogram.contains(bucket)) {
      prefix_sum += histogram[bucket];
    }

    // If the noisy prefix_sum is above the noisy threshold, stop the loop.
    double noisy_prefix_sum = mechanism->AddNoise(prefix_sum);
    if (noisy_prefix_sum >= noisy_threshold) {
      break;
    }
  }

  return bucket;
}

// The Create method of the DPQuantileAggregatorFactory.
StatusOr<std::unique_ptr<TensorAggregator>>
DPQuantileAggregatorFactory::CreateInternal(
    const Intrinsic& intrinsic,
    const DPQuantileAggregatorState* aggregator_state) const {
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

  // Next, validate the input and output specs.
  // Ensure that input spec has exactly one tensor.
  if (intrinsic.inputs.size() != 1) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "DPQuantileAggregatorFactory::CreateInternal: Expected one input "
              "tensor, but got "
           << intrinsic.inputs.size();
  }
  const TensorSpec& input_spec_tensor = intrinsic.inputs[0];

  // Ensure that the input spec's tensor is a scalar.
  TFF_ASSIGN_OR_RETURN(auto num_elements_in_tensor,
                       input_spec_tensor.shape().NumElements());
  if (num_elements_in_tensor != 1) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "DPQuantileAggregatorFactory::CreateInternal: Expected a scalar "
              "input tensor, but got a tensor with "
           << num_elements_in_tensor << " elements.";
  }

  // Ensure that output spec has exactly one tensor.
  if (intrinsic.outputs.size() != 1) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "DPQuantileAggregatorFactory::CreateInternal: Expected one output"
              " tensor, but got "
           << intrinsic.outputs.size();
  }
  const TensorSpec& output_spec_tensor = intrinsic.outputs[0];

  // Ensure that the output spec's tensor is a scalar.
  TFF_ASSIGN_OR_RETURN(num_elements_in_tensor,
                       output_spec_tensor.shape().NumElements());
  if (num_elements_in_tensor != 1) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "DPQuantileAggregatorFactory::CreateInternal: Expected a scalar "
              "output tensor, but got a tensor with "
           << num_elements_in_tensor << " elements.";
  }

  DataType input_type = input_spec_tensor.dtype();
  DataType output_type = output_spec_tensor.dtype();

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

  if (aggregator_state == nullptr) {
    DTYPE_CASES(input_type, T,
                return std::make_unique<DPQuantileAggregator<T>>(
                    target_quantile, intrinsic.inputs));
  }
  auto num_inputs = aggregator_state->num_inputs();
  auto reservoir_sampling_count = aggregator_state->reservoir_sampling_count();
  DTYPE_CASES(input_type, T,
              return std::make_unique<DPQuantileAggregator<T>>(
                  target_quantile, num_inputs, reservoir_sampling_count,
                  MutableVectorData<T>::CreateFromEncodedContent(
                      aggregator_state->buffer()),
                  intrinsic.inputs));
}

REGISTER_AGGREGATOR_FACTORY(kDPQuantileUri, DPQuantileAggregatorFactory);

}  // namespace aggregation
}  // namespace tensorflow_federated
