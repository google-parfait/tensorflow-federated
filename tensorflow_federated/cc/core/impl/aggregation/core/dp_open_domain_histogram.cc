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

#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_open_domain_histogram.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/types/span.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/agg_core.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/composite_key_combiner.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/datatype.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_composite_key_combiner.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_fedsql_constants.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_noise_mechanisms.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/group_by_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/input_tensor_list.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/intrinsic.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_vector_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/one_dim_grouping_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_spec.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/vector_string_data.h"

namespace tensorflow_federated {
namespace aggregation {

using ::differential_privacy::sign;

namespace internal {

// Noise is added to each value stored in a column tensor. If the noised value
// falls below a given threshold, then the index of that value is removed from a
// set of survivors.
// NoiseAndThreshold will be called by DPOpenDomainHistogram::Report on multiple
// tensors. Upon completion, Report will copy a value at index i in an output
// tensor of NoiseAndThreshold if i is in the set of survivors.
// NB: It is possible to write NoiseAndThreshold to cull values that lie below
// a threshold, but the i-th item of column 1 might not correspond to the i-th
// item of column 2. So we defer the culling step until we know all indices of
// the survivors.
// References: The document Delta_For_Thresholding.pdf found in
// https://github.com/google/differential-privacy/blob/main/common_docs/ has a
// proof for the case where inputs are positive; our use of sign() generalizes
// the analysis to the non-positive case.
template <typename OutputType>
StatusOr<Tensor> NoiseAndThreshold(
    double epsilon, double delta, int64_t l0_bound, OutputType linfinity_bound,
    double l1_bound, double l2_bound, const Tensor& column_tensor,
    absl::flat_hash_set<size_t>& survivor_indices,
    std::vector<bool>& laplace_was_used) {
  TFF_ASSIGN_OR_RETURN(
      auto bundle,
      CreateDPHistogramBundle(epsilon, delta, l0_bound, linfinity_bound,
                              l1_bound, l2_bound, true));

  laplace_was_used.push_back(bundle.use_laplace);

  TFF_CHECK(bundle.threshold.has_value())
      << "NoiseAndThreshold: threshold was not set.";
  OutputType threshold = static_cast<OutputType>(bundle.threshold.value());

  auto column_span = column_tensor.AsSpan<OutputType>();
  auto noisy_values = std::make_unique<MutableVectorData<OutputType>>();
  noisy_values->reserve(column_span.size());

  // For every value in the column,
  for (size_t i = 0; i < column_span.size(); i++) {
    OutputType value = column_span[i];

    // Add noise and store noisy value
    OutputType noisy_value = (bundle.mechanism)->AddNoise(value);
    noisy_values->push_back(noisy_value);

    // If threshold is not crossed, index does not belong in survivor_indices
    OutputType sign_of_value = sign<OutputType>(value);
    if (sign_of_value * noisy_value < threshold) {
      survivor_indices.erase(i);
    }
  }
  return Tensor::Create(internal::TypeTraits<OutputType>::kDataType,
                        {static_cast<int64_t>(column_span.size())},
                        std::move(noisy_values));
}

// Given a column tensor, copy elements whose indices belong to a set of
// survivors to a new tensor.
template <typename OutputType>
StatusOr<Tensor> CopyOnlySurvivors(
    const Tensor& column_tensor,
    const absl::flat_hash_set<size_t>& survivor_indices) {
  auto column_span = column_tensor.AsSpan<OutputType>();
  auto dest = std::make_unique<MutableVectorData<OutputType>>();
  dest->reserve(survivor_indices.size());
  for (size_t i = 0; i < column_span.size(); i++) {
    if (!survivor_indices.contains(i)) {
      continue;
    }
    dest->push_back(column_span[i]);
  }
  return Tensor::Create(internal::TypeTraits<OutputType>::kDataType,
                        {static_cast<int64_t>(survivor_indices.size())},
                        std::move(dest));
}
// Specialization for string_view. First make a vector of surviving std::strings
// from the string_views and survivor_indices, then create a Tensor by moving
// the contents to an intermediate VectorStringData object.
template <>
StatusOr<Tensor> CopyOnlySurvivors<string_view>(
    const Tensor& column_tensor,
    const absl::flat_hash_set<size_t>& survivor_indices) {
  auto column_span = column_tensor.AsSpan<string_view>();
  std::vector<std::string> strings_for_output;
  strings_for_output.reserve(survivor_indices.size());
  for (size_t i = 0; i < column_span.size(); i++) {
    if (!survivor_indices.contains(i)) {
      continue;
    }
    strings_for_output.push_back(std::string(column_span[i]));
  }

  return Tensor::Create(
      DT_STRING, {static_cast<int64_t>(survivor_indices.size())},
      std::make_unique<VectorStringData>(std::move(strings_for_output)));
}
}  // namespace internal

DPOpenDomainHistogram::DPOpenDomainHistogram(
    const std::vector<TensorSpec>& input_key_specs,
    const std::vector<TensorSpec>* output_key_specs,
    const std::vector<Intrinsic>* intrinsics,
    std::unique_ptr<CompositeKeyCombiner> key_combiner,
    std::vector<std::unique_ptr<OneDimBaseGroupingAggregator>> aggregators,
    double epsilon_per_agg, double delta_per_agg, int64_t l0_bound,
    int num_inputs)
    : GroupByAggregator(input_key_specs, output_key_specs, intrinsics,
                        std::move(key_combiner), std::move(aggregators),
                        num_inputs),
      epsilon_per_agg_(epsilon_per_agg),
      delta_per_agg_(delta_per_agg),
      l0_bound_(l0_bound) {}

std::unique_ptr<DPCompositeKeyCombiner>
DPOpenDomainHistogram::CreateDPKeyCombiner(
    const std::vector<TensorSpec>& input_key_specs,
    const std::vector<TensorSpec>* output_key_specs, int64_t l0_bound) {
  // If there are no input keys, support a columnar aggregation that aggregates
  // all the values in each column and produces a single output value per
  // column. This would be equivalent to having identical key values for all
  // rows.
  if (input_key_specs.empty()) {
    return nullptr;
  }

  // Otherwise create a DP-ready key combiner
  return std::make_unique<DPCompositeKeyCombiner>(
      GroupByAggregator::CreateKeyTypes(input_key_specs.size(), input_key_specs,
                                        *output_key_specs),
      l0_bound);
}

StatusOr<Tensor> DPOpenDomainHistogram::CreateOrdinalsByGroupingKeysForMerge(
    const InputTensorList& inputs) {
  if (num_keys_per_input() > 0) {
    InputTensorList keys(num_keys_per_input());
    for (int i = 0; i < num_keys_per_input(); ++i) {
      keys[i] = inputs[i];
    }
    // Call the version of Accumulate that has no L0 bounding
    return key_combiner()->CompositeKeyCombiner::Accumulate(std::move(keys));
  }
  // If there are no keys, we should aggregate all elements in a column into one
  // element, as if there were an imaginary key column with identical values for
  // all rows.
  auto ordinals =
      std::make_unique<MutableVectorData<int64_t>>(inputs[0]->num_elements());
  return Tensor::Create(internal::TypeTraits<int64_t>::kDataType,
                        inputs[0]->shape(), std::move(ordinals));
}

StatusOr<OutputTensorList> DPOpenDomainHistogram::Report() && {
  TFF_RETURN_IF_ERROR(CheckValid());
  if (!CanReport()) {
    return TFF_STATUS(FAILED_PRECONDITION)
           << "DPOpenDomainHistogram::Report: the report goal isn't met";
  }
  // Compute the noiseless aggregate.
  OutputTensorList noiseless_aggregate = std::move(*this).TakeOutputs();

  // We skip noise addition if epsilon is too large to be meaningful
  if (epsilon_per_agg_ >= kEpsilonThreshold) {
    return noiseless_aggregate;
  }

  size_t num_aggregations = intrinsics().size();

  // TakeOutputs only includes a key if its name is in output_key_spec
  size_t num_output_keys = 0;
  for (int i = 0; i < output_key_specs().size(); ++i) {
    if (output_key_specs()[i].name().empty()) continue;
    num_output_keys++;
  }

  // Create a set of indices, one for each composite key in the histogram.
  absl::flat_hash_set<size_t> survivor_indices;
  for (size_t i = 0; i < noiseless_aggregate[0].num_elements(); i++) {
    survivor_indices.insert(i);
  }

  // For each aggregation, run NoiseAndThreshold to populate a list of Tensors
  OutputTensorList noisy_values;
  noisy_values.reserve(num_aggregations);
  for (int j = 0; j < num_aggregations; ++j) {
    const auto& inner_parameters = intrinsics()[j].parameters;
    const Tensor& linfinity_tensor = inner_parameters[kLinfinityIndex];
    double l1_bound = inner_parameters[kL1Index].CastToScalar<double>();
    double l2_bound = inner_parameters[kL2Index].CastToScalar<double>();
    size_t column = num_output_keys + j;
    StatusOr<Tensor> tensor;
    NUMERICAL_ONLY_DTYPE_CASES(
        noiseless_aggregate[column].dtype(), OutputType,
        TFF_ASSIGN_OR_RETURN(
            tensor, internal::NoiseAndThreshold<OutputType>(
                        epsilon_per_agg_, delta_per_agg_, l0_bound_,
                        linfinity_tensor.CastToScalar<OutputType>(), l1_bound,
                        l2_bound, noiseless_aggregate[column], survivor_indices,
                        laplace_was_used_)));
    noisy_values.push_back(std::move(tensor.value()));
  }

  // When there are no grouping keys, aggregation will be scalar. Hence, the
  // sole "group" does not need to be dropped for DP (because it exists whether
  // or not a given client contributed data)
  if (num_keys_per_input() == 0) {
    survivor_indices.insert(0);
  }

  // Produce a new list of tensors containing only the survivors of
  // thresholding
  OutputTensorList final_histogram;
  final_histogram.reserve(noiseless_aggregate.size());
  for (size_t j = 0; j < noiseless_aggregate.size(); j++) {
    // First batch of Tensors are for keys, second are for the values
    const Tensor& column_tensor = (j < num_output_keys)
                                      ? noiseless_aggregate[j]
                                      : noisy_values[j - num_output_keys];
    StatusOr<Tensor> tensor;
    DTYPE_CASES(
        column_tensor.dtype(), OutputType,
        TFF_ASSIGN_OR_RETURN(tensor, internal::CopyOnlySurvivors<OutputType>(
                                         column_tensor, survivor_indices)));
    final_histogram.push_back(std::move(tensor.value()));
  }
  return final_histogram;
}

}  // namespace aggregation
}  // namespace tensorflow_federated
