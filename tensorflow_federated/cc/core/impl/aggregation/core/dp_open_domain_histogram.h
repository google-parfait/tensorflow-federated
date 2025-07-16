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

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_DP_OPEN_DOMAIN_HISTOGRAM_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_DP_OPEN_DOMAIN_HISTOGRAM_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/agg_core.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/composite_key_combiner.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_composite_key_combiner.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/group_by_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/input_tensor_list.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/intrinsic.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/one_dim_grouping_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_slice_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_spec.h"

namespace tensorflow_federated {
namespace aggregation {

// DPOpenDomainHistogram is a child class of GroupByAggregator.
// ::AggregateTensorsInternal enforces a bound on the number of composite keys
// (ordinals) that any one aggregation can contribute to.
// ::Report adds noise to aggregates and removes composite keys that have value
// below a threshold.
// This class is not thread safe.
class DPOpenDomainHistogram : public GroupByAggregator {
 public:
  // Performs the same checks as TensorAggregator::Report but also checks
  // magnitude of DP budget. If too large, simply releases noiseless aggregate.
  // Otherwise, applies NoiseAndThreshold to the noiseless aggregate.
  StatusOr<OutputTensorList> Report() && override;

  // Accessor to vector that indicates, for each aggregation, whether Laplace
  // noise was used to ensure DP. This information is independent of user data
  // and only depends on the constructor's parameters.
  // If called before Report(), the vector will be empty.
  std::vector<bool> laplace_was_used() const { return laplace_was_used_; }

  // Given a column of data and a set of survivor indices, shrink the column to
  // only include the survivors.
  template <typename OutputType>
  static Status ShrinkToSurvivors(
      TensorSliceData& column,
      const absl::flat_hash_set<size_t>& survivor_indices) {
    TFF_ASSIGN_OR_RETURN(absl::Span<OutputType> column_span,
                         column.AsSpan<OutputType>());
    size_t num_elements = column_span.size();
    // Locate the smallest index of a non-survivor, then the first survivor
    // after it.
    int64_t destination;
    for (destination = 0;
         destination < num_elements && survivor_indices.contains(destination);
         destination++) {
    }
    int64_t source;
    for (source = destination + 1;
         source < num_elements && !survivor_indices.contains(source);
         source++) {
    }
    while (destination < num_elements && source < num_elements) {
      // Swap to lengthen the prefix of survivors, then advance the destination
      // and source indexes.
      std::swap(column_span[destination], column_span[source]);
      destination++;
      for (source++;
           source < num_elements && !survivor_indices.contains(source);
           source++) {
      }
    }

    // Now that the survivors are in the front, reduce the byte size of the
    // tensor to only include the survivors.
    TFF_RETURN_IF_ERROR(
        column.ReduceByteSize(survivor_indices.size() * sizeof(OutputType)));
    return absl::OkStatus();
  }

 protected:
  friend class DPGroupByFactory;

  // Constructs a DPOpenDomainHistogram.
  // This constructor is meant for use by the DPGroupByFactory; most callers
  // should instead create a DPOpenDomainHistogram from an intrinsic using the
  // factory, i.e.
  // `(*GetAggregatorFactory("fedsql_dp_group_by"))->Create(intrinsic)`
  //
  // Takes the same inputs as GroupByAggregator, in addition to:
  // * epsilon_per_agg: the privacy budget per nested intrinsic.
  // * delta_per_agg: the privacy failure parameter per nested intrinsic.
  // * l0_bound: the maximum number of composite keys one user can contribute to
  //   (assuming each DPOpenDomainHistogram::AggregateTensorsInternal call
  //    contains data from a unique user)
  DPOpenDomainHistogram(
      const std::vector<TensorSpec>& input_key_specs,
      const std::vector<TensorSpec>* output_key_specs,
      const std::vector<Intrinsic>* intrinsics,
      std::unique_ptr<CompositeKeyCombiner> key_combiner,
      std::vector<std::unique_ptr<OneDimBaseGroupingAggregator>> aggregators,
      double epsilon_per_agg, double delta_per_agg, int64_t l0_bound,
      int num_inputs);

 private:
  // Returns either nullptr or a unique_ptr to a CompositeKeyCombiner, depending
  // on the input specification
  static std::unique_ptr<DPCompositeKeyCombiner> CreateDPKeyCombiner(
      const std::vector<TensorSpec>& input_key_specs,
      const std::vector<TensorSpec>* output_key_specs, int64_t l0_bound);

  // When merging two DPOpenDomainHistograms, norm bounding the aggregates will
  // destroy accuracy and is not needed for privacy. Hence, this function calls
  // CompositeKeyCombiner::Accumulate, which has no L0 norm bounding.
  StatusOr<Tensor> CreateOrdinalsByGroupingKeysForMerge(
      const InputTensorList& inputs) override;

  double epsilon_per_agg_;
  double delta_per_agg_;
  int64_t l0_bound_;

  // At index i, the boolean in the below vector indicates if laplace noise was
  // used to ensure DP for the i-th aggregation. The vector is empty before
  // Report() is called.
  std::vector<bool> laplace_was_used_;
};

}  // namespace aggregation
}  // namespace tensorflow_federated

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_DP_OPEN_DOMAIN_HISTOGRAM_H_
