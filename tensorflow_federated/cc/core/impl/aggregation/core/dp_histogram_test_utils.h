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

#ifndef TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_DP_HISTOGRAM_TEST_UTILS_H_
#define TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_DP_HISTOGRAM_TEST_UTILS_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "googletest/include/gtest/gtest.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/agg_vector.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/agg_vector_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/datatype.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_exhaustive_report_histogram.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_fedsql_constants.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_thresholding_histogram.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/group_by_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/intrinsic.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_spec.h"

namespace tensorflow_federated {
namespace aggregation {

// Peer class for testing private methods of DPExhaustiveReportHistogram.
// This allows us to test, among other things, the noise description.
class DPExhaustiveReportHistogramPeer {
 public:
  explicit DPExhaustiveReportHistogramPeer(
      std::unique_ptr<TensorAggregator> aggregator) {
    auto* raw_ptr =
        dynamic_cast<DPExhaustiveReportHistogram*>(aggregator.get());
    TFF_CHECK(raw_ptr != nullptr)
        << "Aggregator must be a DPExhaustiveReportHistogram";
    dp_histogram_ = std::unique_ptr<DPExhaustiveReportHistogram>(
        dynamic_cast<DPExhaustiveReportHistogram*>(aggregator.release()));
  }
  StatusOr<std::string> GetNoiseDescription() const {
    return dp_histogram_->GetNoiseDescription();
  }

 private:
  std::unique_ptr<DPExhaustiveReportHistogram> dp_histogram_;
  friend class DPExhaustiveReportHistogram;
  friend class GroupByAggregator;
};

// Peer class for testing private methods of DPThresholdingHistogram.
// This allows us to test, among other things, the creation of the selector,
// which is not at time of writing exposed in the public API and won't be
// exposed except through complicated code that should be tested separately. We
// also have access to some other internals that are useful to test.
class DPThresholdingHistogramPeer {
 public:
  explicit DPThresholdingHistogramPeer(
      std::unique_ptr<TensorAggregator> aggregator) {
    auto* raw_ptr = dynamic_cast<DPThresholdingHistogram*>(aggregator.get());
    TFF_CHECK(raw_ptr != nullptr)
        << "Aggregator must be a DPThresholdingHistogram";
    dp_histogram_ = std::unique_ptr<DPThresholdingHistogram>(
        dynamic_cast<DPThresholdingHistogram*>(aggregator.release()));
  }

  bool HasSelector() const { return dp_histogram_->selector_ != nullptr; }

  int64_t GetMaxGroupsContributed() const {
    return dp_histogram_->max_groups_contributed();
  }

  std::optional<int64_t> GetMaxContributorsToGroup() const {
    return dp_histogram_->max_contributors_to_group();
  }

  const std::vector<int>& GetContributorsToGroups() const {
    return dp_histogram_->contributors_to_groups();
  }

  double GetSelectorEpsilon() const {
    return dp_histogram_->selector_->GetEpsilon();
  }

  double GetEpsilonPerAgg() const { return dp_histogram_->epsilon_per_agg(); }

  double GetSelectorDelta() const {
    return dp_histogram_->selector_->GetDelta();
  }

  double GetDeltaPerAgg() const { return dp_histogram_->delta_per_agg(); }

  StatusOr<std::string> GetNoiseDescription() const {
    return dp_histogram_->GetNoiseDescription();
  }

 private:
  std::unique_ptr<DPThresholdingHistogram> dp_histogram_;
  friend class DPThresholdingHistogram;
  friend class GroupByAggregator;
};

namespace dp_histogram_testing {

TensorSpec CreateTensorSpec(std::string name, DataType dtype);

// A simple Sum Aggregator
template <typename T>
class SumAggregator final : public AggVectorAggregator<T> {
 public:
  using AggVectorAggregator<T>::AggVectorAggregator;
  using AggVectorAggregator<T>::data;

 private:
  void AggregateVector(const AggVector<T>& agg_vector) override {
    // This aggregator is not expected to actually be used for aggregating in
    // the current tests.
    FAIL() << "SumAggregator should not be used for aggregation.";
  }
};

template <typename EpsilonType, typename DeltaType, typename L0_BoundType>
std::vector<Tensor> CreateTopLevelParameters(
    EpsilonType epsilon, DeltaType delta, L0_BoundType l0_bound,
    absl::Span<const DataType> key_types, bool add_spec = false) {
  std::vector<Tensor> parameters;

  parameters.push_back(Tensor(epsilon, "epsilon"));
  parameters.push_back(Tensor(delta, "delta"));
  parameters.push_back(Tensor(l0_bound, "max_groups_contributed"));

  if (key_types.empty() || !add_spec) {
    return parameters;
  }

  // First tensor contains the names of keys
  int64_t num_keys = key_types.size();
  std::vector<std::string> key_names(num_keys);
  for (int i = 0; i < num_keys; i++) {
    key_names[i] = absl::StrCat("key", i);
  }
  parameters.push_back(Tensor(std::move(key_names), "key_names"));

  // The ith tensor contains the domain of values that the ith key can take.
  int key_type_index = 0;
  for (DataType dtype : key_types) {
    std::string tensor_name = absl::StrCat("key", key_type_index);
    if (dtype == DT_STRING) {
      parameters.push_back(Tensor({"a", "b", "c"}, tensor_name));
    } else {
      NUMERICAL_ONLY_DTYPE_CASES(
          dtype, T,
          parameters.push_back(
              Tensor({static_cast<T>(0), static_cast<T>(1), static_cast<T>(2)},
                     tensor_name)));
    }
    key_type_index++;
  }

  return parameters;
}

template <typename EpsilonType, typename DeltaType, typename L0_BoundType>
std::vector<Tensor> CreateTopLevelParameters(EpsilonType epsilon,
                                             DeltaType delta,
                                             L0_BoundType l0_bound) {
  return CreateTopLevelParameters<EpsilonType, DeltaType, L0_BoundType>(
      epsilon, delta, l0_bound, {});
}

std::vector<Tensor> CreateTopLevelParameters(double epsilon, double delta,
                                             int64_t l0_bound);

template <typename InputType>
std::vector<Tensor> CreateNestedParameters(InputType linfinity_bound,
                                           double l1_bound, double l2_bound) {
  std::vector<Tensor> parameters;

  parameters.push_back(Tensor(linfinity_bound));
  parameters.push_back(Tensor(l1_bound));
  parameters.push_back(Tensor(l2_bound));

  return parameters;
}

template <typename InputType, typename OutputType>
Intrinsic CreateInnerIntrinsic(InputType linfinity_bound, double l1_bound,
                               double l2_bound) {
  return Intrinsic{.uri = kDPSumUri,
                   .inputs = {CreateTensorSpec(
                       "value", internal::TypeTraits<InputType>::kDataType)},
                   .outputs = {CreateTensorSpec(
                       "value", internal::TypeTraits<OutputType>::kDataType)},
                   .parameters = {CreateNestedParameters<InputType>(
                       linfinity_bound, l1_bound, l2_bound)},
                   .nested_intrinsics = {}};
}

template <typename InputType, typename OutputType>
Intrinsic CreateIntrinsic(double epsilon = kEpsilonThreshold,
                          double delta = 0.001, int64_t l0_bound = 100,
                          InputType linfinity_bound = 100, double l1_bound = -1,
                          double l2_bound = -1) {
  Intrinsic intrinsic = {
      .uri = kDPGroupByUri,
      .inputs = {CreateTensorSpec("key", DT_STRING)},
      .outputs = {CreateTensorSpec("key_out", DT_STRING)},
      .parameters = {CreateTopLevelParameters(epsilon, delta, l0_bound)},
      .nested_intrinsics = {}};
  intrinsic.nested_intrinsics.push_back(
      CreateInnerIntrinsic<InputType, OutputType>(linfinity_bound, l1_bound,
                                                  l2_bound));
  return intrinsic;
}

template <typename InputType, typename OutputType>
Intrinsic CreateIntrinsicWithKeyTypes(
    double epsilon = kEpsilonThreshold, double delta = 0.001,
    int64_t l0_bound = 100, InputType linfinity_bound = 100,
    double l1_bound = -1, double l2_bound = -1,
    std::vector<DataType> key_types = {DT_STRING}, bool add_spec = false) {
  std::vector<TensorSpec> input_key_specs;
  for (int i = 0; i < key_types.size(); i++) {
    input_key_specs.push_back(
        CreateTensorSpec(absl::StrCat("key", i), key_types[i]));
  }
  std::vector<TensorSpec> output_key_specs;
  for (int i = 0; i < key_types.size(); i++) {
    output_key_specs.push_back(CreateTensorSpec(
        absl::StrCat("key", std::to_string(i), "_out"), key_types[i]));
  }
  Intrinsic intrinsic = {.uri = kDPGroupByUri,
                         .inputs = input_key_specs,
                         .outputs = output_key_specs,
                         .parameters = {CreateTopLevelParameters(
                             epsilon, delta, l0_bound, key_types, add_spec)},
                         .nested_intrinsics = {}};
  intrinsic.nested_intrinsics.push_back(
      CreateInnerIntrinsic<InputType, OutputType>(linfinity_bound, l1_bound,
                                                  l2_bound));
  return intrinsic;
}

template <typename InputType, typename OutputType>
Intrinsic CreateIntrinsicWithKeyTypes_ExhaustiveReport(
    double epsilon = kEpsilonThreshold, double delta = 0.001,
    int64_t l0_bound = 100, InputType linfinity_bound = 100,
    double l1_bound = -1, double l2_bound = -1,
    std::vector<DataType> key_types = {DT_STRING}) {
  return CreateIntrinsicWithKeyTypes<InputType, OutputType>(
      epsilon, delta, l0_bound, linfinity_bound, l1_bound, l2_bound, key_types,
      /*add_spec=*/true);
}

// Creates an intrinsic for use in creating DPHistogram objects with a
// min_contributors_to_group parameter which is the k for k-thresholding.
// The key types and the other parameters can be set or the default values can
// be used for most purposes.
template <typename InputType, typename OutputType>
Intrinsic CreateIntrinsicWithMinContributors(
    int64_t min_contributors, double epsilon = kEpsilonThreshold,
    double delta = 0.001, int64_t l0_bound = 100,
    InputType linfinity_bound = 100, double l1_bound = -1, double l2_bound = -1,
    std::vector<DataType> key_types = {DT_STRING}) {
  Intrinsic intrinsic = CreateIntrinsicWithKeyTypes<InputType, OutputType>(
      epsilon, delta, l0_bound, linfinity_bound, l1_bound, l2_bound, key_types);
  intrinsic.parameters.push_back(
      Tensor(min_contributors, "min_contributors_to_group"));

  return intrinsic;
}

}  // namespace dp_histogram_testing
}  // namespace aggregation
}  // namespace tensorflow_federated

#endif  // TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_DP_HISTOGRAM_TEST_UTILS_H_
