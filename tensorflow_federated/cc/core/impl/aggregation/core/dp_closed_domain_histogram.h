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

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_DP_CLOSED_DOMAIN_HISTOGRAM_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_DP_CLOSED_DOMAIN_HISTOGRAM_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/container/fixed_array.h"
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
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_spec.h"

namespace tensorflow_federated {
namespace aggregation {

// DPClosedDomainHistogram is a child class of GroupByAggregator.
// ::AggregateTensorsInternal enforces a bound on the number of composite keys
// (ordinals) that any one aggregation can contribute to.
// ::Report adds noise to aggregates
// This class is not thread safe.
class DPClosedDomainHistogram : public GroupByAggregator {
 public:
  // Performs the same checks as TensorAggregator::Report but also checks
  // magnitude of DP budget. If too large, simply releases noiseless aggregate.
  // Otherwise, adds noise scaled to either L1 or L2 sensitivity.
  StatusOr<OutputTensorList> Report() && override;

  // Accessor to the tensors that specify the domain of each key.
  const TensorSpan& domain_tensors() const { return domain_tensors_; }

 protected:
  friend class DPGroupByFactory;

  // Constructs a DPClosedDomainHistogram.
  // This constructor is meant for use by the DPGroupByFactory; most callers
  // should instead create a DPClosedDomainHistogram from an intrinsic using the
  // factory, i.e.
  // `(*GetAggregatorFactory("fedsql_dp_group_by"))->Create(intrinsic)`
  //
  // Takes the same inputs as GroupByAggregator, in addition to:
  // * epsilon_per_agg: the privacy budget per nested intrinsic.
  // * delta_per_agg: the privacy failure parameter per nested intrinsic.
  // * l0_bound: the maximum number of composite keys one user can contribute to
  //   (assuming each DPClosedDomainHistogram::AggregateTensorsInternal call
  //    contains data from a unique user). Can be negative, in which case no
  //    bound on the number of composite keys is enforced.
  // * domain_tensors: a Span of Tensors such that the i-th describes the domain
  //   of the i-th grouping key.
  DPClosedDomainHistogram(
      const std::vector<TensorSpec>& input_key_specs,
      const std::vector<TensorSpec>* output_key_specs,
      const std::vector<Intrinsic>* intrinsics,
      std::unique_ptr<CompositeKeyCombiner> key_combiner,
      std::vector<std::unique_ptr<OneDimBaseGroupingAggregator>> aggregators,
      double epsilon_per_agg, double delta_per_agg, int64_t l0_bound,
      TensorSpan domain_tensors, int num_inputs);

 private:
  // When merging two DPClosedDomainHistograms, bounding the aggregates will
  // destroy accuracy and is not needed for privacy. Hence, this function calls
  // CompositeKeyCombiner::Accumulate, which has no L0 norm bounding.
  StatusOr<Tensor> CreateOrdinalsByGroupingKeysForMerge(
      const InputTensorList& inputs) override;

  // Given indices that specify a combination of keys, increment the index
  // corresponding to a particular key. If the index is at the edge of the key's
  // domain, wrap around and recurse to the next key.
  // Return true if we can continue incrementing, false if we've reached the
  // end of the entire domain of composite keys.
  // Assumes that which_key is in the range [0, domain_tensors_.size()]
  // and that the i-th entry of domain_indices is in the range
  // [0, domain_tensors_[i].num_elements()]
  bool IncrementDomainIndices(absl::FixedArray<int64_t>& domain_indices,
                              int64_t which_key = 0);

  double epsilon_per_agg_;
  double delta_per_agg_;
  int64_t l0_bound_;

  TensorSpan domain_tensors_;
};

}  // namespace aggregation
}  // namespace tensorflow_federated

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_DP_CLOSED_DOMAIN_HISTOGRAM_H_
