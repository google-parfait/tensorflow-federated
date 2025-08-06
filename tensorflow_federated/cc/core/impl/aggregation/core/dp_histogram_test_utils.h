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
#include <string>
#include <utility>
#include <vector>

#include "googletest/include/gtest/gtest.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/agg_vector.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/agg_vector_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/datatype.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_fedsql_constants.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/intrinsic.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_string_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_vector_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_spec.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/test_data.h"

namespace tensorflow_federated {
namespace aggregation {
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
    absl::Span<const DataType> key_types) {
  std::vector<Tensor> parameters;

  std::unique_ptr<MutableVectorData<EpsilonType>> epsilon_tensor =
      CreateTestData<EpsilonType>({epsilon});
  parameters.push_back(
      Tensor::Create(internal::TypeTraits<EpsilonType>::kDataType, {},
                     std::move(epsilon_tensor), "epsilon")
          .value());

  std::unique_ptr<MutableVectorData<DeltaType>> delta_tensor =
      CreateTestData<DeltaType>({delta});
  parameters.push_back(
      Tensor::Create(internal::TypeTraits<DeltaType>::kDataType, {},
                     std::move(delta_tensor), "delta")
          .value());

  std::unique_ptr<MutableVectorData<L0_BoundType>> l0_bound_tensor =
      CreateTestData<L0_BoundType>({l0_bound});
  parameters.push_back(
      Tensor::Create(internal::TypeTraits<L0_BoundType>::kDataType, {},
                     std::move(l0_bound_tensor), "max_groups_contributed")
          .value());

  if (key_types.empty()) {
    return parameters;
  }

  // First tensor contains the names of keys
  int64_t num_keys = key_types.size();
  auto key_names = std::make_unique<MutableStringData>(num_keys);
  for (int i = 0; i < num_keys; i++) {
    key_names->Add(absl::StrCat("key", i));
  }
  parameters.push_back(
      Tensor::Create(DT_STRING, {num_keys}, std::move(key_names), "key_names")
          .value());

  // The ith tensor contains the domain of values that the ith key can take.
  int key_type_index = 0;
  for (DataType dtype : key_types) {
    const int kNumValues = 3;
    if (dtype == DT_STRING) {
      parameters.push_back(
          Tensor::Create(DT_STRING, {kNumValues},
                         CreateTestData<string_view>({"a", "b", "c"}),
                         absl::StrCat("key", key_type_index))
              .value());
    } else {
      NUMERICAL_ONLY_DTYPE_CASES(
          dtype, T,
          parameters.push_back(
              Tensor::Create(dtype, {kNumValues}, CreateTestData<T>({0, 1, 2}),
                             absl::StrCat("key", key_type_index))
                  .value()));
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

  parameters.push_back(
      Tensor::Create(internal::TypeTraits<InputType>::kDataType, {},
                     CreateTestData<InputType>({linfinity_bound}))
          .value());

  parameters.push_back(
      Tensor::Create(DT_DOUBLE, {}, CreateTestData<double>({l1_bound}))
          .value());

  parameters.push_back(
      Tensor::Create(DT_DOUBLE, {}, CreateTestData<double>({l2_bound}))
          .value());

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
    std::vector<DataType> key_types = {DT_STRING}) {
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
                             epsilon, delta, l0_bound, key_types)},
                         .nested_intrinsics = {}};
  intrinsic.nested_intrinsics.push_back(
      CreateInnerIntrinsic<InputType, OutputType>(linfinity_bound, l1_bound,
                                                  l2_bound));
  return intrinsic;
}

template <typename InputType, typename OutputType>
Intrinsic CreateIntrinsicWithMinContributors(
    int64_t min_contributors, std::vector<DataType> key_types = {}) {
  Intrinsic intrinsic = CreateIntrinsicWithKeyTypes<InputType, OutputType>(
      kEpsilonThreshold, 0.001, 100, 100, -1, -1, key_types);
  std::unique_ptr<MutableVectorData<int64_t>> min_contributors_tensor =
      CreateTestData<int64_t>({min_contributors});
  intrinsic.parameters.push_back(
      Tensor::Create(internal::TypeTraits<int64_t>::kDataType, {},
                     std::move(min_contributors_tensor),
                     "min_contributors_to_group")
          .value());

  return intrinsic;
}

}  // namespace dp_histogram_testing
}  // namespace aggregation
}  // namespace tensorflow_federated

#endif  // TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_DP_HISTOGRAM_TEST_UTILS_H_
