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


#include <cstdint>
#include <initializer_list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/agg_vector.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/agg_vector_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/datatype.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_fedsql_constants.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/intrinsic.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_vector_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator_registry.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_shape.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_spec.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/test_data.h"
#include "tensorflow_federated/cc/testing/status_matchers.h"

namespace tensorflow_federated {
namespace aggregation {
namespace {

using ::testing::HasSubstr;
using testing::TestWithParam;

using DPClosedDomainHistogramTest = TestWithParam<bool>;

constexpr int kNumValues = 3;

TensorSpec CreateTensorSpec(std::string name, DataType dtype) {
  return TensorSpec(name, dtype, {-1});
}

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
    ASSERT_TRUE(false);
  }
};

template <typename EpsilonType, typename DeltaType, typename L0_BoundType>
std::vector<Tensor> CreateTopLevelParameters(EpsilonType epsilon,
                                             DeltaType delta,
                                             L0_BoundType l0_bound,
                                             std::vector<DataType> key_types) {
  std::vector<Tensor> parameters;

  std::unique_ptr<MutableVectorData<EpsilonType>> epsilon_tensor =
      CreateTestData<EpsilonType>({epsilon});
  parameters.push_back(
      Tensor::Create(internal::TypeTraits<EpsilonType>::kDataType, {},
                     std::move(epsilon_tensor))
          .value());

  std::unique_ptr<MutableVectorData<DeltaType>> delta_tensor =
      CreateTestData<DeltaType>({delta});
  parameters.push_back(
      Tensor::Create(internal::TypeTraits<DeltaType>::kDataType, {},
                     std::move(delta_tensor))
          .value());

  std::unique_ptr<MutableVectorData<L0_BoundType>> l0_bound_tensor =
      CreateTestData<L0_BoundType>({l0_bound});
  parameters.push_back(
      Tensor::Create(internal::TypeTraits<L0_BoundType>::kDataType, {},
                     std::move(l0_bound_tensor))
          .value());

  // First tensor contains the names of keys
  int64_t num_keys = key_types.size();
  MutableVectorData<string_view> key_names(num_keys);
  for (int i = 0; i < num_keys; i++) {
    key_names[i] = "key" + std::to_string(i);
  }
  parameters.push_back(
      Tensor::Create(DT_STRING, {num_keys},
                     std::make_unique<MutableVectorData<string_view>>(
                         std::move(key_names)))
          .value());

  // The ith tensor contains the domain of values that the ith key can take.
  for (auto& dtype : key_types) {
    if (dtype == DT_STRING) {
      parameters.push_back(
          Tensor::Create(DT_STRING, {kNumValues},
                         CreateTestData<string_view>({"a", "b", "c"}))
              .value());
    } else {
      NUMERICAL_ONLY_DTYPE_CASES(
          dtype, T,
          parameters.push_back(
              Tensor::Create(dtype, {kNumValues}, CreateTestData<T>({0, 1, 2}))
                  .value()));
    }
  }

  return parameters;
}

std::vector<Tensor> CreateTopLevelParameters(
    double epsilon = 1000.0, double delta = 0.001, int64_t l0_bound = 100,
    std::vector<DataType> key_types = {DT_STRING}) {
  return CreateTopLevelParameters<double, double, int64_t>(epsilon, delta,
                                                           l0_bound, key_types);
}

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
  return Intrinsic{
      kDPSumUri,
      {CreateTensorSpec("value", internal::TypeTraits<InputType>::kDataType)},
      {CreateTensorSpec("value", internal::TypeTraits<OutputType>::kDataType)},
      {CreateNestedParameters<InputType>(linfinity_bound, l1_bound, l2_bound)},
      {}};
}

template <typename InputType, typename OutputType>
Intrinsic CreateIntrinsic(double epsilon = 1000.0, double delta = 0.001,
                          int64_t l0_bound = 100,
                          InputType linfinity_bound = 100, double l1_bound = -1,
                          double l2_bound = -1,
                          std::vector<DataType> key_types = {DT_STRING}) {
  Intrinsic intrinsic{
      kDPGroupByUri,
      {CreateTensorSpec("key", DT_STRING)},
      {CreateTensorSpec("key_out", DT_STRING)},
      {CreateTopLevelParameters(epsilon, delta, l0_bound, key_types)},
      {}};
  intrinsic.nested_intrinsics.push_back(
      CreateInnerIntrinsic<InputType, OutputType>(linfinity_bound, l1_bound,
                                                  l2_bound));
  return intrinsic;
}

// First batch of tests validate the intrinsic(s)
TEST(DPClosedDomainHistogramTest, CatchWrongNumberOfParameters) {
  // Provide domain spec for one key but there are two keys
  std::vector<DataType> key_types = {DT_STRING};
  Intrinsic too_few{kDPGroupByUri,
                    {CreateTensorSpec("key0", DT_STRING),
                     CreateTensorSpec("key1", DT_STRING)},
                    {CreateTensorSpec("key0_out", DT_STRING),
                     CreateTensorSpec("key1_out", DT_STRING)},
                    {CreateTopLevelParameters(1.0, 0.01, 10, key_types)},
                    {}};
  auto too_few_status = CreateTensorAggregator(too_few).status();
  EXPECT_THAT(too_few_status, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(too_few_status.message(),
              HasSubstr("Expected 3 domain tensors but got 2 of them"));
  // Provide domain spec for three keys but there are two keys
  key_types = {DT_STRING, DT_STRING, DT_STRING};
  Intrinsic too_many{kDPGroupByUri,
                     {CreateTensorSpec("key0", DT_STRING),
                      CreateTensorSpec("key1", DT_STRING)},
                     {CreateTensorSpec("key0_out", DT_STRING),
                      CreateTensorSpec("key1_out", DT_STRING)},
                     {CreateTopLevelParameters(1.0, 0.01, 10, key_types)},
                     {}};
  auto too_many_status = CreateTensorAggregator(too_many).status();
  EXPECT_THAT(too_many_status, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(too_many_status.message(),
              HasSubstr("Expected 3 domain tensors but got 4 of them"));
}

TEST(DPClosedDomainHistogramTest, CatchWrongKeyNameType) {
  Intrinsic intrinsic = CreateIntrinsic<int64_t, int64_t>();
  intrinsic.parameters[3] =
      Tensor::Create(DT_INT64, {1}, CreateTestData<int64_t>({1})).value();
  auto wrong_key_name_type_status = CreateTensorAggregator(intrinsic).status();
  EXPECT_THAT(wrong_key_name_type_status, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(wrong_key_name_type_status.message(),
              HasSubstr("First domain tensor should have string type"));
}

TEST(DPClosedDomainHistogramTest, CatchWrongKeyTypes) {
  std::vector<DataType> key_types = {DT_DOUBLE};

  Intrinsic intrinsic{kDPGroupByUri,
                      {CreateTensorSpec("key0", DT_STRING)},
                      {CreateTensorSpec("key0_out", DT_STRING)},
                      {CreateTopLevelParameters(1, 0.001, 10, key_types)},
                      {}};
  auto wrong_key_types_status = CreateTensorAggregator(intrinsic).status();
  EXPECT_THAT(wrong_key_types_status, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(
      wrong_key_types_status.message(),
      HasSubstr("tensor for key 0 should have type " + DataType_Name(DT_STRING)
                + " but got " + DataType_Name(DT_DOUBLE) + " instead"));
}

TEST(DPOpenDomainHistogramTest, CatchInvalidParameterValues) {
  // Negative epsilon
  Intrinsic intrinsic0 = CreateIntrinsic<int64_t, int64_t>(-1, 0.001, 10);
  auto bad_epsilon = CreateTensorAggregator(intrinsic0).status();
  EXPECT_THAT(bad_epsilon, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(bad_epsilon.message(), HasSubstr("Epsilon must be positive"));

  // Delta too large
  Intrinsic intrinsic2 =
      CreateIntrinsic<int64_t, int64_t>(1, 2, 10, 10, -1, -1);
  auto bad_delta1 = CreateTensorAggregator(intrinsic2).status();
  EXPECT_THAT(bad_delta1, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(bad_delta1.message(), HasSubstr("delta must be less than 1"));

  // Missing norm bounds
  Intrinsic intrinsic1 =
      CreateIntrinsic<int64_t, int64_t>(1, 0.001, 3, -1, -1, -1);
  auto bad_bounds = CreateTensorAggregator(intrinsic1).status();
  EXPECT_THAT(bad_bounds, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(bad_bounds.message(),
              HasSubstr("either an L1 bound, an L2 bound,"
                        " or both Linfinity and L0 bounds"));

  // Delta not provided (implied 0) and L1 bound not provided
  Intrinsic intrinsic3 =
      CreateIntrinsic<int64_t, int64_t>(1, -1, 10, 10, -1, -1);
  auto bad_delta2 = CreateTensorAggregator(intrinsic3).status();
  EXPECT_THAT(bad_delta2, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(bad_delta2.message(),
              HasSubstr("either a positive delta or an L1 bound"));
}


}  // namespace
}  // namespace aggregation
}  // namespace tensorflow_federated
