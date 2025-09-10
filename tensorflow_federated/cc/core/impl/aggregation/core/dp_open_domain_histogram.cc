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

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "algorithms/partition-selection.h"
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
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_slice_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_spec.h"

namespace tensorflow_federated {
namespace aggregation {

using ::differential_privacy::NearTruncatedGeometricPartitionSelection;
using ::differential_privacy::sign;

namespace internal {

// Noise is added to each value of a column. If the noised value
// falls below a given threshold, then the index of that value is removed from a
// set of survivors.
// NoiseAndThreshold will be called by DPOpenDomainHistogram::Report on multiple
// columns. Upon completion, Report will copy a value at index i in an output
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
Status NoiseAndThreshold(double epsilon, double delta, int64_t l0_bound,
                         OutputType linfinity_bound, double l1_bound,
                         double l2_bound, TensorSliceData& column,
                         absl::flat_hash_set<size_t>& survivor_indices) {
  TFF_ASSIGN_OR_RETURN(
      auto bundle,
      CreateDPHistogramBundle(epsilon, delta, l0_bound, linfinity_bound,
                              l1_bound, l2_bound, true));

  TFF_CHECK(bundle.threshold.has_value())
      << "NoiseAndThreshold: threshold was not set.";
  OutputType threshold = static_cast<OutputType>(bundle.threshold.value());

  TFF_ASSIGN_OR_RETURN(auto column_span, column.AsSpan<OutputType>());

  // For every value in the column,
  for (size_t i = 0; i < column_span.size(); ++i) {
    OutputType original_value = column_span[i];

    // Add noise and store noisy value
    OutputType noisy_value = (bundle.mechanism)->AddNoise(original_value);
    column_span[i] = noisy_value;

    // If threshold is not crossed, index does not belong in survivor_indices.
    OutputType sign_of_value = sign<OutputType>(original_value);
    if (sign_of_value * noisy_value < threshold) {
      survivor_indices.erase(i);
    }
  }
  return absl::OkStatus();
}

// Noise is added to each value of a column. This is used in
// DPOpenDomainHistogram::Report when k-thresholding is enabled and so we don't
// need to threshold based on the noisy value.
template <typename OutputType>
Status NoiseWithoutThresholding(double epsilon, double delta, int64_t l0_bound,
                                OutputType linfinity_bound, double l1_bound,
                                double l2_bound, TensorSliceData& column) {
  TFF_ASSIGN_OR_RETURN(
      DPHistogramBundle bundle,
      CreateDPHistogramBundle(epsilon, delta, l0_bound, linfinity_bound,
                              l1_bound, l2_bound,
                              /*threshold_by_value=*/false));

  TFF_ASSIGN_OR_RETURN(absl::Span<OutputType> column_span,
                       column.AsSpan<OutputType>());

  for (OutputType& column : column_span) {
    column = (bundle.mechanism)->AddNoise(column);
  }
  return absl::OkStatus();
}

}  // namespace internal

DPOpenDomainHistogram::DPOpenDomainHistogram(
    const std::vector<TensorSpec>& input_key_specs,
    const std::vector<TensorSpec>* output_key_specs,
    const std::vector<Intrinsic>* intrinsics,
    std::unique_ptr<CompositeKeyCombiner> key_combiner,
    std::vector<std::unique_ptr<OneDimBaseGroupingAggregator>> aggregators,
    double epsilon_per_agg, double delta_per_agg, int64_t l0_bound,
    int num_inputs, std::optional<int64_t> min_contributors_to_group,
    std::vector<int> contributors_to_groups)
    : GroupByAggregator(input_key_specs, output_key_specs, intrinsics,
                        std::move(key_combiner), std::move(aggregators),
                        num_inputs, min_contributors_to_group,
                        std::move(contributors_to_groups)),
      epsilon_per_agg_(epsilon_per_agg),
      delta_per_agg_(delta_per_agg),
      l0_bound_(l0_bound) {}

// Factory method to create DPOpenDomainHistogram
StatusOr<std::unique_ptr<DPOpenDomainHistogram>> DPOpenDomainHistogram::Create(
    const std::vector<TensorSpec>& input_key_specs,
    const std::vector<TensorSpec>* output_key_specs,
    const std::vector<Intrinsic>* intrinsics,
    std::unique_ptr<CompositeKeyCombiner> key_combiner,
    std::vector<std::unique_ptr<OneDimBaseGroupingAggregator>> aggregators,
    double epsilon, double delta, int64_t l0_bound, int num_inputs,
    std::optional<int64_t> min_contributors_to_group,
    std::vector<int> contributors_to_groups) {
  bool need_selector =
      min_contributors_to_group.has_value() && epsilon < kEpsilonThreshold;

  double epsilon_for_aggs = epsilon;
  double delta_for_aggs = delta;
  if (need_selector) {
    // We have to split the privacy budget between k-thresholding selection and
    // the noising of the aggregates. For now we split it evenly between
    // selections and aggregations.
    epsilon_for_aggs /= 2;
    delta_for_aggs /= 2;
  }

  auto num_nested_intrinsics = intrinsics->size();
  double epsilon_per_agg = (epsilon_for_aggs < kEpsilonThreshold
                                ? epsilon_for_aggs / num_nested_intrinsics
                                : kEpsilonThreshold);
  double delta_per_agg = delta_for_aggs / num_nested_intrinsics;
  auto dp_open_domain_histogram = absl::WrapUnique(new DPOpenDomainHistogram(
      input_key_specs, output_key_specs, intrinsics, std::move(key_combiner),
      std::move(aggregators), epsilon_per_agg, delta_per_agg, l0_bound,
      num_inputs, min_contributors_to_group, contributors_to_groups));

  if (need_selector) {
    double epsilon_for_selector = epsilon - epsilon_for_aggs;
    double delta_for_selector = delta - delta_for_aggs;
    TFF_ASSIGN_OR_RETURN(auto generic_selector,
                         NearTruncatedGeometricPartitionSelection::Builder()
                             .SetEpsilon(epsilon_for_selector)
                             .SetDelta(delta_for_selector)
                             .SetMaxPartitionsContributed(l0_bound)
                             .SetPreThreshold(*min_contributors_to_group)
                             .Build());
    // generic_selector is a unique_ptr to a base class, so we have to cast it
    // to the derived class in order to access the GetSecondCrossover and
    // ShouldKeep methods.
    auto* raw_ptr = dynamic_cast<NearTruncatedGeometricPartitionSelection*>(
        generic_selector.get());
    if (raw_ptr == nullptr) {
      return absl::InternalError(
          "Failed to cast to NearTruncatedGeometricPartitionSelection");
    }
    generic_selector.release();
    auto selector =
        std::unique_ptr<NearTruncatedGeometricPartitionSelection>(raw_ptr);

    // The selector publishes every group with strictly greater than second
    // crossover elements.
    dp_open_domain_histogram->set_max_contributors_to_group(
        ceil(selector->GetSecondCrossover()));
    dp_open_domain_histogram->selector_ = std::move(selector);
  }
  return dp_open_domain_histogram;
}

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
  if (!CanReport()) {
    return TFF_STATUS(FAILED_PRECONDITION)
           << "DPOpenDomainHistogram::Report: the report goal isn't met";
  }

  // If epsilon is too large to be meaningful, we perform the non-DP
  // aggregation.
  if (epsilon_per_agg_ >= kEpsilonThreshold) {
    return std::move(*this).GroupByAggregator::Report();
  }

  TFF_RETURN_IF_ERROR(CheckValid());
  // Compute the noiseless aggregate.
  OutputTensorList noiseless_aggregate = std::move(*this).TakeOutputs();

  // Log the histogram's Tensor types and migrate to TensorSliceData.
  TFF_ASSIGN_OR_RETURN(HistogramAsSliceData histogram_as_slice_data,
                       ConvertHistogramToSliceData(noiseless_aggregate));
  size_t num_rows = histogram_as_slice_data.num_rows;

  size_t num_aggregations = intrinsics().size();

  // TakeOutputs only includes a key if its name is in output_key_spec
  size_t num_output_keys = 0;
  for (int i = 0; i < output_key_specs().size(); ++i) {
    if (output_key_specs()[i].name().empty()) continue;
    num_output_keys++;
  }

  // Create a set of indices.
  absl::flat_hash_set<size_t> survivor_indices;
  if (!min_contributors_to_group().has_value()) {
    // If there is no k-thresholding, one for each composite key in the
    // histogram.
    for (size_t i = 0; i < num_rows; i++) {
      survivor_indices.insert(i);
    }

    std::vector<std::unique_ptr<TensorSliceData>>& column_data =
        histogram_as_slice_data.column_data;
    std::vector<DataType>& column_dtypes =
        histogram_as_slice_data.column_dtypes;

    // For each aggregation, run NoiseAndThreshold to noise the aggregates and
    // identify which of them that should survive.
    for (int j = 0; j < num_aggregations; ++j) {
      const auto& inner_parameters = intrinsics()[j].parameters;
      const Tensor& linfinity_tensor = inner_parameters[kLinfinityIndex];
      double l1_bound = inner_parameters[kL1Index].CastToScalar<double>();
      double l2_bound = inner_parameters[kL2Index].CastToScalar<double>();
      size_t column = num_output_keys + j;
      StatusOr<Tensor> tensor;
      NUMERICAL_ONLY_DTYPE_CASES(
          column_dtypes[column], OutputType,
          TFF_RETURN_IF_ERROR(internal::NoiseAndThreshold<OutputType>(
              epsilon_per_agg_, delta_per_agg_, l0_bound_,
              linfinity_tensor.CastToScalar<OutputType>(), l1_bound, l2_bound,
              *column_data[column], survivor_indices)));
    }

    // When there are no grouping keys, aggregation will be scalar. Hence, the
    // sole "group" does not need to be dropped for DP (because it exists
    // whether or not a given client contributed data)
    if (num_keys_per_input() == 0) {
      survivor_indices.insert(0);
    }

    // Produce a new list of tensors containing only the survivors of
    // thresholding, in uniformly random order.
    return ShrinkHistogramToSurvivors(std::move(histogram_as_slice_data),
                                      survivor_indices);
  }

  // We now perform k-thresholding as min_contributors_to_group_ is set.
  std::vector<int> contributor_counts = GetContributors();
  TFF_CHECK(contributor_counts.size() == num_rows)
      << "DPOpenDomainHistogram::Report: contributor_counts.size() != "
         "num_rows";
  for (size_t i = 0; i < num_rows; i++) {
    if (selector_->ShouldKeep(contributor_counts[i])) {
      survivor_indices.insert(i);
    }
  }

  TFF_ASSIGN_OR_RETURN(
      OutputTensorList shrunk_noiseless_aggregate,
      ShrinkHistogramToSurvivors(std::move(histogram_as_slice_data),
                                 survivor_indices));

  TFF_ASSIGN_OR_RETURN(HistogramAsSliceData shrunk_histogram_as_slice_data,
                       ConvertHistogramToSliceData(shrunk_noiseless_aggregate));

  std::vector<std::unique_ptr<TensorSliceData>>& column_data =
      shrunk_histogram_as_slice_data.column_data;
  std::vector<DataType>& column_dtypes =
      shrunk_histogram_as_slice_data.column_dtypes;

  // Noise all entries without thresholding by values.
  for (int j = 0; j < num_aggregations; ++j) {
    const std::vector<Tensor>& inner_parameters = intrinsics()[j].parameters;
    const Tensor& linfinity_tensor = inner_parameters[kLinfinityIndex];
    double l1_bound = inner_parameters[kL1Index].CastToScalar<double>();
    double l2_bound = inner_parameters[kL2Index].CastToScalar<double>();
    size_t column = num_output_keys + j;
    StatusOr<Tensor> tensor;
    NUMERICAL_ONLY_DTYPE_CASES(
        column_dtypes[column], OutputType,
        TFF_RETURN_IF_ERROR(internal::NoiseWithoutThresholding<OutputType>(
            epsilon_per_agg_, delta_per_agg_, l0_bound_,
            linfinity_tensor.CastToScalar<OutputType>(), l1_bound, l2_bound,
            *column_data[column])));
  }

  // Convert the sliced data back to a histogram i.e. OutputTensorList.
  OutputTensorList output_histogram(0);
  for (int i = 0; i < column_data.size(); ++i) {
    TFF_ASSIGN_OR_RETURN(
        output_histogram.emplace_back(),
        Tensor::Create(column_dtypes[i],
                       {static_cast<int64_t>(survivor_indices.size())},
                       std::move(column_data[i])));
  }

  return output_histogram;
}

}  // namespace aggregation
}  // namespace tensorflow_federated
