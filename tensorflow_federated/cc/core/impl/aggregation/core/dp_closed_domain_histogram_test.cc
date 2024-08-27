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
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_closed_domain_histogram.h"

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/strings/str_cat.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/agg_vector.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/agg_vector_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/datatype.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_composite_key_combiner.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_fedsql_constants.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/intrinsic.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_vector_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator_registry.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_shape.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_spec.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/test_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/testing.h"
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
  MutableVectorData<string_view> key_names;
  for (int i = 0; i < num_keys; i++) {
    key_names.push_back(absl::StrCat("key", i));
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
Intrinsic CreateIntrinsic(double epsilon = kEpsilonThreshold,
                          double delta = 0.001, int64_t l0_bound = 100,
                          InputType linfinity_bound = 100, double l1_bound = -1,
                          double l2_bound = -1,
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
  Intrinsic intrinsic{
      kDPGroupByUri,
      input_key_specs,
      output_key_specs,
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
  EXPECT_THAT(wrong_key_types_status.message(),
              HasSubstr(absl::StrCat("tensor for key 0 should have type ",
                                     DataType_Name(DT_STRING), " but got ",
                                     DataType_Name(DT_DOUBLE), " instead")));
}

std::vector<Tensor> CreateNParameters(int n) {
  std::vector<Tensor> parameters;
  for (int i = 0; i < n; i++) {
    parameters.push_back(
        Tensor::Create(DT_DOUBLE, {}, CreateTestData<double>({10})).value());
  }
  return parameters;
}

TEST(DPClosedDomainHistogramTest, CatchInnerParameters_WrongNumber) {
  // Too few parameters (only linfinity bound)
  Intrinsic intrinsic1{kDPGroupByUri,
                       {CreateTensorSpec("key", DT_STRING)},
                       {CreateTensorSpec("key_out", DT_STRING)},
                       {CreateTopLevelParameters()},
                       {}};
  intrinsic1.nested_intrinsics.push_back(
      Intrinsic{kDPSumUri,
                {CreateTensorSpec("value", DT_DOUBLE)},
                {CreateTensorSpec("value", DT_DOUBLE)},
                {CreateNParameters(1)},
                {}});

  auto aggregator_status = CreateTensorAggregator(intrinsic1);
  EXPECT_THAT(aggregator_status, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(aggregator_status.status().message(),
              HasSubstr("Linfinity, L1, and L2 bounds are expected"));

  // Too many parameters (linfinity, l1, l2, ??)
  Intrinsic intrinsic2{kDPGroupByUri,
                       {CreateTensorSpec("key", DT_STRING)},
                       {CreateTensorSpec("key_out", DT_STRING)},
                       {CreateTopLevelParameters()},
                       {}};
  intrinsic2.nested_intrinsics.push_back(
      Intrinsic{kDPSumUri,
                {CreateTensorSpec("value", DT_DOUBLE)},
                {CreateTensorSpec("value", DT_DOUBLE)},
                {CreateNParameters(4)},
                {}});
  auto aggregator_status2 = CreateTensorAggregator(intrinsic2);
  EXPECT_THAT(aggregator_status2, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(aggregator_status2.status().message(),
              HasSubstr("Linfinity, L1, and L2 bounds are expected"));
}

template <typename LinfType, typename L1Type, typename L2Type>
std::vector<Tensor> CreateGenericDPGFSParameters(LinfType linfinity_bound,
                                                 L1Type l1_bound,
                                                 L2Type l2_bound) {
  std::vector<Tensor> parameters;

  parameters.push_back(
      Tensor::Create(internal::TypeTraits<LinfType>::kDataType, {},
                     CreateTestData<LinfType>({linfinity_bound}))
          .value());

  parameters.push_back(Tensor::Create(internal::TypeTraits<L1Type>::kDataType,
                                      {}, CreateTestData<L1Type>({l1_bound}))
                           .value());

  parameters.push_back(Tensor::Create(internal::TypeTraits<L2Type>::kDataType,
                                      {}, CreateTestData<L2Type>({l2_bound}))
                           .value());

  return parameters;
}
TEST(DPClosedDomainHistogramTest, CatchInnerParameters_WrongTypes) {
  Intrinsic intrinsic1{kDPGroupByUri,
                       {CreateTensorSpec("key", DT_STRING)},
                       {CreateTensorSpec("key_out", DT_STRING)},
                       {CreateTopLevelParameters()},
                       {}};
  intrinsic1.nested_intrinsics.push_back(
      Intrinsic{kDPSumUri,
                {CreateTensorSpec("value", DT_INT64)},
                {CreateTensorSpec("value", DT_INT64)},
                {CreateGenericDPGFSParameters<string_view, double, double>(
                    /*linfinity_bound=*/"x",
                    /*l1_bound=*/-1,
                    /*l2_bound=*/-1)},
                {}});
  auto aggregator_status1 = CreateTensorAggregator(intrinsic1);
  EXPECT_THAT(aggregator_status1, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(aggregator_status1.status().message(),
              HasSubstr("numerical Tensors"));

  Intrinsic intrinsic2{kDPGroupByUri,
                       {CreateTensorSpec("key", DT_STRING)},
                       {CreateTensorSpec("key_out", DT_STRING)},
                       {CreateTopLevelParameters()},
                       {}};
  intrinsic2.nested_intrinsics.push_back(Intrinsic{
      kDPSumUri,
      {CreateTensorSpec("value", DT_INT64)},
      {CreateTensorSpec("value", DT_INT64)},
      {CreateGenericDPGFSParameters<int64_t, string_view, double>(10, "x", -1)},
      {}});

  auto aggregator_status2 = CreateTensorAggregator(intrinsic2);
  EXPECT_THAT(aggregator_status2, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(aggregator_status2.status().message(),
              HasSubstr("numerical Tensors"));

  Intrinsic intrinsic3{kDPGroupByUri,
                       {CreateTensorSpec("key", DT_STRING)},
                       {CreateTensorSpec("key_out", DT_STRING)},
                       {CreateTopLevelParameters()},
                       {}};
  intrinsic3.nested_intrinsics.push_back(Intrinsic{
      kDPSumUri,
      {CreateTensorSpec("value", DT_INT64)},
      {CreateTensorSpec("value", DT_INT64)},
      {CreateGenericDPGFSParameters<int64_t, double, string_view>(10, -1, "x")},
      {}});
  auto aggregator_status3 = CreateTensorAggregator(intrinsic3);
  EXPECT_THAT(aggregator_status3, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(aggregator_status3.status().message(),
              HasSubstr("numerical Tensors"));
}

TEST(DPClosedDomainHistogramTest, CatchInvalidParameterValues) {
  // Negative epsilon
  Intrinsic intrinsic0 = CreateIntrinsic<int64_t, int64_t>(/*epsilon=*/-1,
                                                           /*delta=*/0.001,
                                                           /*l0_bound=*/10);
  auto bad_epsilon = CreateTensorAggregator(intrinsic0).status();
  EXPECT_THAT(bad_epsilon, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(bad_epsilon.message(), HasSubstr("Epsilon must be positive"));

  // Delta too large
  Intrinsic intrinsic2 =
      CreateIntrinsic<int64_t, int64_t>(1, /*delta=*/2, 10, 10, -1, -1);
  auto bad_delta1 = CreateTensorAggregator(intrinsic2).status();
  EXPECT_THAT(bad_delta1, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(bad_delta1.message(), HasSubstr("delta must be less than 1"));

  // Missing norm bounds
  Intrinsic intrinsic1 =
      CreateIntrinsic<int64_t, int64_t>(1, 0.001, 3, /*linfinity_bound=*/-1,
                                        /*l1_bound=*/-1,
                                        /*l2_bound=*/-1);
  auto bad_bounds = CreateTensorAggregator(intrinsic1).status();
  EXPECT_THAT(bad_bounds, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(bad_bounds.message(),
              HasSubstr("either an L1 bound, an L2 bound,"
                        " or both Linfinity and L0 bounds"));

  // Delta not provided and only an L2 bound was provided
  Intrinsic intrinsic3 =
      CreateIntrinsic<int64_t, int64_t>(1, /*delta=*/-1, -1, -1, -1,
                                        /*l2_bound=*/3);
  auto bad_delta2 = CreateTensorAggregator(intrinsic3).status();
  EXPECT_THAT(bad_delta2, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(
      bad_delta2.message(),
      HasSubstr("either a positive delta or one of the following: "
                "(a) an L1 bound (b) an Linfinity bound and an L0 bound"));
}

// Second batch of tests validate the aggregator itself, without DP noise.

// Make sure we can successfully create a DPClosedDomainHistogram object.
TEST(DPClosedDomainHistogramTest, CreateAggregator_Success) {
  Intrinsic intrinsic = CreateIntrinsic<int64_t, int64_t>();
  auto status = CreateTensorAggregator(intrinsic);
  TFF_EXPECT_OK(status);

  // Validate the domain tensor: default intrinsic has one key that takes values
  // in the set {"a", "b", "c"}
  auto& agg = status.value();
  auto& dpcdh = dynamic_cast<DPClosedDomainHistogram&>(*agg);
  TensorSpan domain_tensors = dpcdh.domain_tensors();

  EXPECT_EQ(domain_tensors.size(), 1);
  EXPECT_EQ(domain_tensors[0].shape(), TensorShape({3}));
  EXPECT_EQ(domain_tensors[0].dtype(), DT_STRING);
  EXPECT_EQ(domain_tensors[0].AsSpan<string_view>()[0], "a");
  EXPECT_EQ(domain_tensors[0].AsSpan<string_view>()[1], "b");
  EXPECT_EQ(domain_tensors[0].AsSpan<string_view>()[2], "c");
}

// Make sure the Report without DP noise contains all composite keys and their
// aggregations.
// One key taking values in the set {"a", "b", "c"}
TEST(DPClosedDomainHistogramTest, NoiselessReport_OneKey) {
  // Create intrinsic with one string key ({"a", "b", "c"} is default domain)
  Intrinsic intrinsic = CreateIntrinsic<int64_t, int64_t>(
      kEpsilonThreshold, 0.001, 10, 10, -1, -1, {DT_STRING});
  // Create a DPClosedDomainHistogram object
  auto status = CreateTensorAggregator(intrinsic);
  TFF_EXPECT_OK(status);
  auto& agg = status.value();

  // Accumulate twice
  Tensor key1 =
      Tensor::Create(DT_STRING, {2}, CreateTestData<string_view>({"c", "a"}))
          .value();
  Tensor value1 =
      Tensor::Create(DT_INT64, {2}, CreateTestData<int64_t>({1, 2})).value();
  auto acc_status = agg->Accumulate({&key1, &value1});
  TFF_EXPECT_OK(acc_status);
  Tensor key2 =
      Tensor::Create(DT_STRING, {1}, CreateTestData<string_view>({"a"}))
          .value();
  Tensor value2 =
      Tensor::Create(DT_INT64, {1}, CreateTestData<int64_t>({3})).value();
  acc_status = agg->Accumulate({&key2, &value2});
  TFF_EXPECT_OK(acc_status);

  // Report should look like {a: 5, b: 0, c: 1}
  EXPECT_TRUE(agg->CanReport());
  auto report_status = std::move(*agg).Report();
  TFF_EXPECT_OK(report_status);
  auto& report = report_status.value();
  ASSERT_EQ(report.size(), 2);
  EXPECT_THAT(report[0], IsTensor<string_view>({3}, {"a", "b", "c"}));
  EXPECT_THAT(report[1], IsTensor<int64_t>({3}, {5, 0, 1}));
}

// Two keys taking values in the sets {"a", "b", "c"} and {0, 1, 2}
// Number of possible composite keys is 9 = 3 * 3.
TEST(DPClosedDomainHistogramTest, NoiselessReport_TwoKeys) {
  Intrinsic intrinsic = CreateIntrinsic<int64_t, int64_t>(
      kEpsilonThreshold, 0.001, 10, 10, -1, -1, {DT_STRING, DT_INT64});
  // Create a DPClosedDomainHistogram object
  auto status = CreateTensorAggregator(intrinsic);
  TFF_EXPECT_OK(status);
  auto& agg = status.value();

  // Accumulate twice
  Tensor key1a =
      Tensor::Create(DT_STRING, {2}, CreateTestData<string_view>({"c", "a"}))
          .value();
  Tensor key1b =
      Tensor::Create(DT_INT64, {2}, CreateTestData<int64_t>({1, 2})).value();
  Tensor value1 =
      Tensor::Create(DT_INT64, {2}, CreateTestData<int64_t>({1, 2})).value();
  auto acc_status = agg->Accumulate({&key1a, &key1b, &value1});
  TFF_EXPECT_OK(acc_status);
  Tensor key2a =
      Tensor::Create(DT_STRING, {2}, CreateTestData<string_view>({"a", "a"}))
          .value();
  Tensor key2b =
      Tensor::Create(DT_INT64, {2}, CreateTestData<int64_t>({0, 2})).value();
  Tensor value2 =
      Tensor::Create(DT_INT64, {2}, CreateTestData<int64_t>({3, 3})).value();
  acc_status = agg->Accumulate({&key2a, &key2b, &value2});
  TFF_EXPECT_OK(acc_status);

  EXPECT_TRUE(agg->CanReport());
  auto report_status = std::move(*agg).Report();
  TFF_EXPECT_OK(report_status);
  auto& report = report_status.value();
  // three tensors (columns): first key, second key, aggregation
  ASSERT_EQ(report.size(), 3);

  // first key: letters cycle as a, b, c, a, b, c, a, b, c
  EXPECT_THAT(report[0], IsTensor<string_view>({9}, {"a", "b", "c", "a", "b",
                                                     "c", "a", "b", "c"}));

  // second key: numbers cycle as 0, 0, 0, 1, 1, 1, 2, 2, 2
  EXPECT_THAT(report[1], IsTensor<int64_t>({9}, {0, 0, 0, 1, 1, 1, 2, 2, 2}));

  // Report should map a0 to 3, c1 to 1, a2 to 5 = 2+3, and all else to 0.
  // (a0 is the composite key at index 0, c1 is at index 5, a2 is at index 6)
  EXPECT_THAT(report[2], IsTensor<int64_t>({9}, {3, 0, 0, 0, 0, 1, 5, 0, 0}));
}

// Same as above except we do not output the key that takes numerical values.
TEST(DPClosedDomainHistogramTest, NoiselessReport_TwoKeys_DropSecondKey) {
  Intrinsic intrinsic = CreateIntrinsic<int64_t, int64_t>(
      /*epsilon=*/kEpsilonThreshold, /*delta=*/0.001, /*l0_bound=*/10,
      /*linfinity_bound=*/10, /*l1_bound=*/-1, /*l2_bound=*/-1,
      /*key_types=*/{DT_STRING, DT_INT64});
  intrinsic.outputs[1] = CreateTensorSpec("", DT_INT64);

  // Create a DPClosedDomainHistogram object
  auto status = CreateTensorAggregator(intrinsic);
  TFF_EXPECT_OK(status);
  auto& agg = status.value();

  // Accumulate twice
  Tensor key1a =
      Tensor::Create(DT_STRING, {2}, CreateTestData<string_view>({"c", "a"}))
          .value();
  Tensor key1b =
      Tensor::Create(DT_INT64, {2}, CreateTestData<int64_t>({1, 2})).value();
  Tensor value1 =
      Tensor::Create(DT_INT64, {2}, CreateTestData<int64_t>({1, 2})).value();
  auto acc_status = agg->Accumulate({&key1a, &key1b, &value1});
  TFF_EXPECT_OK(acc_status);
  Tensor key2a =
      Tensor::Create(DT_STRING, {2}, CreateTestData<string_view>({"a", "a"}))
          .value();
  Tensor key2b =
      Tensor::Create(DT_INT64, {2}, CreateTestData<int64_t>({0, 2})).value();
  Tensor value2 =
      Tensor::Create(DT_INT64, {2}, CreateTestData<int64_t>({3, 3})).value();
  acc_status = agg->Accumulate({&key2a, &key2b, &value2});
  TFF_EXPECT_OK(acc_status);

  EXPECT_TRUE(agg->CanReport());
  auto report_status = std::move(*agg).Report();
  TFF_EXPECT_OK(report_status);
  auto& report = report_status.value();
  // two tensors (columns): first key of letters, then aggregation
  ASSERT_EQ(report.size(), 2);

  // first key: letters cycle as a, b, c, a, b, c, a, b, c
  EXPECT_THAT(report[0], IsTensor<string_view>({9}, {"a", "b", "c", "a", "b",
                                                     "c", "a", "b", "c"}));

  // Report should map a0 to 3, c1 to 1, a2 to 5 = 2+3, and all else to 0.
  // (a0 is the composite key at index 0, c1 is at index 5, a2 is at index 6)
  EXPECT_THAT(report[1], IsTensor<int64_t>({9}, {3, 0, 0, 0, 0, 1, 5, 0, 0}));
}

// Third: Check that noise is added. the noised sum should not be the same as
// the unnoised sum. The chance of a false negative shrinks with epsilon.
TEST(DPClosedDomainHistogramTest, NoiseAddedForSmallEpsilons) {
  Intrinsic intrinsic =
      CreateIntrinsic<int32_t, int64_t>(/*epsilon=*/0.05,
                                        /*delta=*/1e-8,
                                        /*l0_bound=*/2,
                                        /*linfinity_bound=*/1);
  auto aggregator = CreateTensorAggregator(intrinsic).value();
  int num_inputs = 4000;
  for (int i = 0; i < num_inputs; i++) {
    Tensor keys =
        Tensor::Create(DT_STRING, {2}, CreateTestData<string_view>({"a", "b"}))
            .value();
    Tensor values =
        Tensor::Create(DT_INT32, {2}, CreateTestData<int32_t>({1, 1})).value();
    auto acc_status = aggregator->Accumulate({&keys, &values});
    EXPECT_THAT(acc_status, IsOk());
  }
  EXPECT_EQ(aggregator->GetNumInputs(), num_inputs);
  EXPECT_TRUE(aggregator->CanReport());

  auto report = std::move(*aggregator).Report();
  EXPECT_THAT(report, IsOk());

  // There must be 2 columns, one for keys and one for aggregated values.
  ASSERT_EQ(report->size(), 2);

  const auto& values = report.value()[1].AsSpan<int64_t>();

  // There must be 3 rows, one per key (a, b, c)
  ASSERT_EQ(values.size(), 3);

  // We expect that there is some perturbation in the output.
  // The values for a and b should be num_inputs +/- noise, while the value for
  // c should be 0 +/- noise.
  EXPECT_TRUE(values[0] != num_inputs && values[1] != num_inputs &&
              values[2] != 0);
}

}  // namespace
}  // namespace aggregation
}  // namespace tensorflow_federated
