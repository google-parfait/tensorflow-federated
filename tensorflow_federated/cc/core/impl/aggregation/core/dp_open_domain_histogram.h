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

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "algorithms/partition-selection.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/agg_core.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/composite_key_combiner.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_composite_key_combiner.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_fedsql_constants.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_group_by_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/input_tensor_list.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/intrinsic.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/one_dim_grouping_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_spec.h"

namespace tensorflow_federated {
namespace aggregation {

// DPOpenDomainHistogram is a child class of GroupByAggregator.
// ::AggregateTensorsInternal enforces a bound on the number of composite keys
// (ordinals) that any one aggregation can contribute to.
// ::Report adds noise to aggregates and removes composite keys that have value
// below a threshold.
// This class is not thread safe.
class DPOpenDomainHistogram : public DPGroupByAggregator {
 protected:
  friend class DPGroupByFactory;
  friend class DPOpenDomainHistogramPeer;

  // Constructs a DPOpenDomainHistogram.
  // This constructor is meant for use by the DPGroupByFactory; most callers
  // should instead create a DPOpenDomainHistogram from an intrinsic using the
  // factory, i.e.
  // `(*GetAggregatorFactory("fedsql_dp_group_by"))->Create(intrinsic)`
  static StatusOr<std::unique_ptr<DPOpenDomainHistogram>> Create(
      const std::vector<TensorSpec>& input_key_specs,
      const std::vector<TensorSpec>* output_key_specs,
      const std::vector<Intrinsic>* intrinsics,
      std::unique_ptr<CompositeKeyCombiner> key_combiner,
      std::vector<std::unique_ptr<OneDimBaseGroupingAggregator>> aggregators,
      int num_inputs, double epsilon, double delta,
      int64_t max_groups_contributed,
      std::optional<int> min_contributors_to_group = std::nullopt,
      std::vector<int> contributors_to_groups = {},
      int max_string_length = kDefaultMaxStringLength);

  // Returns either nullptr or a unique_ptr to a CompositeKeyCombiner, depending
  // on the input specification
  static std::unique_ptr<DPCompositeKeyCombiner> CreateDPKeyCombiner(
      const std::vector<TensorSpec>& input_key_specs,
      const std::vector<TensorSpec>* output_key_specs, int64_t l0_bound);

  // Applies NoiseAndThreshold to the noiseless aggregate.
  StatusOr<OutputTensorList> NoisyReport() override;

 private:
  // Constructs a DPOpenDomainHistogram. Only called by the Create() method
  // above.
  DPOpenDomainHistogram(
      const std::vector<TensorSpec>& input_key_specs,
      const std::vector<TensorSpec>* output_key_specs,
      const std::vector<Intrinsic>* intrinsics,
      std::unique_ptr<CompositeKeyCombiner> key_combiner,
      std::vector<std::unique_ptr<OneDimBaseGroupingAggregator>> aggregators,
      int num_inputs, double epsilon, double delta,
      int64_t max_groups_contributed,
      std::optional<int> min_contributors_to_group,
      std::vector<int> contributors_to_groups,
      int max_string_length = kDefaultMaxStringLength);

  // When merging two DPOpenDomainHistograms, norm bounding the aggregates will
  // destroy accuracy and is not needed for privacy. Hence, this function calls
  // CompositeKeyCombiner::Accumulate, which has no L0 norm bounding.
  StatusOr<Tensor> CreateOrdinalsByGroupingKeysForMerge(
      const InputTensorList& inputs) override;

  std::unique_ptr<
      differential_privacy::NearTruncatedGeometricPartitionSelection>
      selector_;
};

}  // namespace aggregation
}  // namespace tensorflow_federated

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_DP_OPEN_DOMAIN_HISTOGRAM_H_
