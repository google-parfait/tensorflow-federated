/*
 * Copyright 2025 Google LLC
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
#ifndef TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_DP_GROUP_BY_AGGREGATOR_H_
#define TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_DP_GROUP_BY_AGGREGATOR_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/composite_key_combiner.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_fedsql_constants.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/group_by_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/intrinsic.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/one_dim_grouping_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_spec.h"

namespace tensorflow_federated {
namespace aggregation {

// The DPGroupByAggregator is an abstract base class for GroupBy aggregations
// that enforce differential privacy (DP).
class DPGroupByAggregator : public GroupByAggregator {
 public:
  // Every DPGroupByAggregator's Report() has the same form: check preconditions
  // and then call NoisyReport().
  StatusOr<OutputTensorList> Report() && override;

 protected:
  // Constructs a DPGroupByAggregator. Only intended for use by child classes.
  //
  // Takes the same inputs as GroupByAggregator, in addition to:
  // * epsilon: the privacy budget for released aggregations.
  // * delta: the privacy failure parameter for released aggregations.
  // * max_groups_contributed: if all data from a unique privacy unit is in
  //   a single DPGroupByAggregator::AggregateTensorsInternal call, this is the
  //   max number of composite keys one privacy unit contributes to,
  // * max_string_length: the maximum length of any string datum; defaults to
  //   kDefaultMaxStringLength.
  //
  // The output_key_specs and intrinsics are passed as pointers  as opposed to
  // references because they are required to outlast the DPGroupByAggregator.
  DPGroupByAggregator(
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

  // Different DP algorithms will produce noisy reports in different ways.
  virtual StatusOr<OutputTensorList> NoisyReport() = 0;

  // Access the maximum number of groups that a privacy unit can contribute to.
  inline int64_t max_groups_contributed() const {
    return max_groups_contributed_;
  }

  // Access the epsilon budget allocated to each aggregation.
  inline double epsilon_per_agg() const { return epsilon_per_agg_; }

  // Access the delta budget allocated to each aggregation.
  inline double delta_per_agg() const { return delta_per_agg_; }

 private:
  double epsilon_;
  double delta_;
  int64_t max_groups_contributed_;
  int max_string_length_;
  double epsilon_per_agg_;
  double delta_per_agg_;
};

}  // namespace aggregation
}  // namespace tensorflow_federated

#endif  // TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_DP_GROUP_BY_AGGREGATOR_H_
