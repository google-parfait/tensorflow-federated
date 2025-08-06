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

#include <cmath>
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
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_histogram_test_utils.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/intrinsic.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_string_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_vector_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator_registry.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_shape.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_spec.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/test_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/testing.h"
#include "tensorflow_federated/cc/testing/status_matchers.h"

namespace tensorflow_federated {
namespace aggregation {
namespace {

using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::Ne;
using ::testing::Not;
using ::testing::TestWithParam;

using ::tensorflow_federated::aggregation::dp_histogram_testing::
    CreateIntrinsicWithKeyTypes;
using ::tensorflow_federated::aggregation::dp_histogram_testing::
    CreateTensorSpec;
using ::tensorflow_federated::aggregation::dp_histogram_testing::
    CreateTopLevelParameters;

using DPClosedDomainHistogramTest = TestWithParam<bool>;

constexpr int kNumValues = 3;

std::vector<Tensor> CreateTopLevelParameters() {
  return CreateTopLevelParameters<double, double, int64_t>(1000.0, 0.001, 100,
                                                           {DT_STRING});
}

// First batch of tests validate the intrinsic(s)
TEST(DPClosedDomainHistogramTest, CatchWrongNumberOfKeyNames) {
  // Provide domain spec for one key but there are two keys
  std::vector<DataType> key_types = {DT_STRING};
  Intrinsic too_few = {
      .uri = kDPGroupByUri,
      .inputs = {CreateTensorSpec("key0", DT_STRING),
                 CreateTensorSpec("key1", DT_STRING)},
      .outputs = {CreateTensorSpec("key0_out", DT_STRING),
                  CreateTensorSpec("key1_out", DT_STRING)},
      .parameters = {CreateTopLevelParameters(1.0, 0.01, 10, key_types)},
      .nested_intrinsics = {}};
  // Provide domain spec for three keys but there are two keys
  key_types = {DT_STRING, DT_STRING, DT_STRING};
  Intrinsic too_many{
      .uri = kDPGroupByUri,
      .inputs = {CreateTensorSpec("key0", DT_STRING),
                 CreateTensorSpec("key1", DT_STRING)},
      .outputs = {CreateTensorSpec("key0_out", DT_STRING),
                  CreateTensorSpec("key1_out", DT_STRING)},
      .parameters = {CreateTopLevelParameters(1.0, 0.01, 10, key_types)},
      .nested_intrinsics = {}};

  EXPECT_THAT(
      CreateTensorAggregator(too_few).status(),
      StatusIs(INVALID_ARGUMENT,
               HasSubstr("The number of key names provided (1) does not match "
                         "the number of input tensors provided (2)")));
  EXPECT_THAT(
      CreateTensorAggregator(too_many).status(),
      StatusIs(INVALID_ARGUMENT,
               HasSubstr("The number of key names provided (3) does not match "
                         "the number of input tensors provided (2)")));
}

TEST(DPClosedDomainHistogramTest, CatchWrongKeyNamesType) {
  Intrinsic wrong_key_names_type =
      CreateIntrinsicWithKeyTypes<int64_t, int64_t>();
  wrong_key_names_type.parameters[3] =
      Tensor::Create(DT_INT64, {1}, CreateTestData<int64_t>({1}), "key_names")
          .value();

  EXPECT_THAT(CreateTensorAggregator(wrong_key_names_type).status(),
              StatusIs(INVALID_ARGUMENT,
                       HasSubstr("Key names should be of type string")));
}

TEST(DPClosedDomainHistogramTest, CatchWrongKeyTypes) {
  std::vector<DataType> key_types = {DT_DOUBLE};
  Intrinsic wrong_key_types = {
      .uri = kDPGroupByUri,
      .inputs = {CreateTensorSpec("key0", DT_STRING)},
      .outputs = {CreateTensorSpec("key0_out", DT_STRING)},
      .parameters = {CreateTopLevelParameters(1, 0.001, 10, key_types)},
      .nested_intrinsics = {}};

  EXPECT_THAT(
      CreateTensorAggregator(wrong_key_types).status(),
      StatusIs(INVALID_ARGUMENT,
               HasSubstr(absl::StrCat(
                   "for key key0 should have type ", DataType_Name(DT_STRING),
                   " but has type ", DataType_Name(DT_DOUBLE), " instead"))));
}

std::vector<Tensor> CreateNParameters(int n) {
  std::vector<Tensor> parameters;
  for (int i = 0; i < n; i++) {
    parameters.push_back(
        Tensor::Create(DT_DOUBLE, {}, CreateTestData<double>({10})).value());
  }
  return parameters;
}

TEST(DPClosedDomainHistogramTest, CatchMisnamedParameters) {
  std::vector<DataType> key_types = {DT_STRING, DT_STRING};
  std::vector<Tensor> parameters =
      CreateTopLevelParameters(1.0, 0.01, 10, key_types);
  parameters[5] = Tensor::Create(DT_STRING, {kNumValues},
                                 CreateTestData<string_view>({"a", "b", "c"}),
                                 "wrong key name")
                      .value();
  Intrinsic invalid_key_name = {
      .uri = kDPGroupByUri,
      .inputs = {CreateTensorSpec("key0", DT_STRING),
                 CreateTensorSpec("key1", DT_STRING)},
      .outputs = {CreateTensorSpec("key0_out", DT_STRING),
                  CreateTensorSpec("key1_out", DT_STRING)},
      .parameters = {std::move(parameters)},
      .nested_intrinsics = {}};

  EXPECT_THAT(
      CreateTensorAggregator(invalid_key_name),
      StatusIs(INVALID_ARGUMENT,
               HasSubstr(
                   "All keys must be listed in order amongst the parameters")));
}

TEST(DPClosedDomainHistogramTest, CatchUnnamedParameter) {
  std::vector<DataType> key_types = {DT_STRING, DT_STRING};
  std::vector<Tensor> parameters =
      CreateTopLevelParameters(1.0, 0.01, 10, key_types);
  parameters[1] = Tensor::Create(internal::TypeTraits<double>::kDataType, {},
                                 CreateTestData<double>({0.01}), "")
                      .value();
  Intrinsic unnamed_delta = {
      .uri = kDPGroupByUri,
      .inputs = {CreateTensorSpec("key0", DT_STRING),
                 CreateTensorSpec("key1", DT_STRING)},
      .outputs = {CreateTensorSpec("key0_out", DT_STRING),
                  CreateTensorSpec("key1_out", DT_STRING)},
      .parameters = {std::move(parameters)},
      .nested_intrinsics = {}};

  EXPECT_THAT(CreateTensorAggregator(unnamed_delta),
              StatusIs(INVALID_ARGUMENT,
                       HasSubstr("For all DP histograms, epsilon, delta, and "
                                 "max_groups_contributed must be provided")));
}

TEST(DPClosedDomainHistogramTest, CatchInnerParameters_WrongNumber) {
  // Too few parameters (only linfinity bound)
  Intrinsic too_few_parameters = {
      .uri = kDPGroupByUri,
      .inputs = {CreateTensorSpec("key", DT_STRING)},
      .outputs = {CreateTensorSpec("key_out", DT_STRING)},
      .parameters = {CreateTopLevelParameters()},
      .nested_intrinsics = {}};
  too_few_parameters.nested_intrinsics.push_back(
      Intrinsic{.uri = kDPSumUri,
                .inputs = {CreateTensorSpec("value", DT_DOUBLE)},
                .outputs = {CreateTensorSpec("value", DT_DOUBLE)},
                .parameters = {CreateNParameters(1)},
                .nested_intrinsics = {}});
  // Too many parameters (linfinity, l1, l2, ??)
  Intrinsic too_many_parameters = {
      .uri = kDPGroupByUri,
      .inputs = {CreateTensorSpec("key", DT_STRING)},
      .outputs = {CreateTensorSpec("key_out", DT_STRING)},
      .parameters = {CreateTopLevelParameters()},
      .nested_intrinsics = {}};
  too_many_parameters.nested_intrinsics.push_back(
      Intrinsic{.uri = kDPSumUri,
                .inputs = {CreateTensorSpec("value", DT_DOUBLE)},
                .outputs = {CreateTensorSpec("value", DT_DOUBLE)},
                .parameters = {CreateNParameters(4)},
                .nested_intrinsics = {}});

  EXPECT_THAT(CreateTensorAggregator(too_few_parameters),
              StatusIs(INVALID_ARGUMENT,
                       HasSubstr("Linfinity, L1, and L2 bounds are expected")));
  EXPECT_THAT(CreateTensorAggregator(too_many_parameters),
              StatusIs(INVALID_ARGUMENT,
                       HasSubstr("Linfinity, L1, and L2 bounds are expected")));
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
  Intrinsic first_inner_parameter_wrong_type = {
      .uri = kDPGroupByUri,
      .inputs = {CreateTensorSpec("key", DT_STRING)},
      .outputs = {CreateTensorSpec("key_out", DT_STRING)},
      .parameters = {CreateTopLevelParameters()},
      .nested_intrinsics = {}};
  first_inner_parameter_wrong_type.nested_intrinsics.push_back(Intrinsic{
      .uri = kDPSumUri,
      .inputs = {CreateTensorSpec("value", DT_INT64)},
      .outputs = {CreateTensorSpec("value", DT_INT64)},
      .parameters = {CreateGenericDPGFSParameters<string_view, double, double>(
          /*linfinity_bound=*/"x",
          /*l1_bound=*/-1,
          /*l2_bound=*/-1)},
      .nested_intrinsics = {}});
  Intrinsic second_inner_parameter_wrong_type = {
      .uri = kDPGroupByUri,
      .inputs = {CreateTensorSpec("key", DT_STRING)},
      .outputs = {CreateTensorSpec("key_out", DT_STRING)},
      .parameters = {CreateTopLevelParameters()},
      .nested_intrinsics = {}};
  second_inner_parameter_wrong_type.nested_intrinsics.push_back(Intrinsic{
      .uri = kDPSumUri,
      .inputs = {CreateTensorSpec("value", DT_INT64)},
      .outputs = {CreateTensorSpec("value", DT_INT64)},
      .parameters = {CreateGenericDPGFSParameters<int64_t, string_view, double>(
          10, "x", -1)},
      .nested_intrinsics = {}});
  Intrinsic third_inner_parameter_wrong_type{
      .uri = kDPGroupByUri,
      .inputs = {CreateTensorSpec("key", DT_STRING)},
      .outputs = {CreateTensorSpec("key_out", DT_STRING)},
      .parameters = {CreateTopLevelParameters()},
      .nested_intrinsics = {}};
  third_inner_parameter_wrong_type.nested_intrinsics.push_back(Intrinsic{
      .uri = kDPSumUri,
      .inputs = {CreateTensorSpec("value", DT_INT64)},
      .outputs = {CreateTensorSpec("value", DT_INT64)},
      .parameters = {CreateGenericDPGFSParameters<int64_t, double, string_view>(
          10, -1, "x")},
      .nested_intrinsics = {}});

  EXPECT_THAT(CreateTensorAggregator(first_inner_parameter_wrong_type),
              StatusIs(INVALID_ARGUMENT, HasSubstr("numerical Tensors")));
  EXPECT_THAT(CreateTensorAggregator(second_inner_parameter_wrong_type),
              StatusIs(INVALID_ARGUMENT, HasSubstr("numerical Tensors")));
  EXPECT_THAT(CreateTensorAggregator(third_inner_parameter_wrong_type),
              StatusIs(INVALID_ARGUMENT, HasSubstr("numerical Tensors")));
}

TEST(DPClosedDomainHistogramTest, CatchInvalidParameterValues) {
  Intrinsic negative_epsilon =
      CreateIntrinsicWithKeyTypes<int64_t, int64_t>(/*epsilon=*/-1,
                                                    /*delta=*/0.001,
                                                    /*l0_bound=*/10);
  Intrinsic delta_too_large = CreateIntrinsicWithKeyTypes<int64_t, int64_t>(
      1, /*delta=*/2, 10, 10, -1, -1);
  Intrinsic missing_norm_bounds = CreateIntrinsicWithKeyTypes<int64_t, int64_t>(
      1, 0.001, 3, /*linfinity_bound=*/-1,
      /*l1_bound=*/-1,
      /*l2_bound=*/-1);
  Intrinsic no_delta_only_l2_bound =
      CreateIntrinsicWithKeyTypes<int64_t, int64_t>(1, /*delta=*/-1, -1, -1, -1,
                                                    /*l2_bound=*/3);

  EXPECT_THAT(
      CreateTensorAggregator(negative_epsilon),
      StatusIs(INVALID_ARGUMENT, HasSubstr("Epsilon must be positive")));
  EXPECT_THAT(
      CreateTensorAggregator(delta_too_large),
      StatusIs(INVALID_ARGUMENT, HasSubstr("delta must be less than 1")));
  EXPECT_THAT(CreateTensorAggregator(missing_norm_bounds),
              StatusIs(INVALID_ARGUMENT,
                       HasSubstr("either an L1 bound, an L2 bound,"
                                 " or both Linfinity and L0 bounds")));
  EXPECT_THAT(
      CreateTensorAggregator(no_delta_only_l2_bound),
      StatusIs(
          INVALID_ARGUMENT,
          HasSubstr("either a positive delta or one of the following: "
                    "(a) an L1 bound (b) an Linfinity bound and an L0 bound")));
}

TEST(DPClosedDomainHistogramTest, CatchDuplicateParameterNames) {
  Intrinsic duplicate_parameter_names =
      CreateIntrinsicWithKeyTypes<int64_t, int64_t>();
  duplicate_parameter_names.parameters.push_back(
      Tensor::Create(DT_INT32, {}, CreateTestData({1}), "epsilon").value());

  EXPECT_THAT(CreateTensorAggregator(duplicate_parameter_names),
              StatusIs(INVALID_ARGUMENT,
                       HasSubstr("Duplicate parameter name: epsilon")));
}

// Second batch of tests validate the aggregator itself, without DP noise.

// Make sure we can successfully create a DPClosedDomainHistogram object.
TEST(DPClosedDomainHistogramTest, CreateAggregator_Success) {
  Intrinsic intrinsic = CreateIntrinsicWithKeyTypes<int64_t, int64_t>();
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
  Intrinsic intrinsic = CreateIntrinsicWithKeyTypes<int64_t, int64_t>(
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
  Intrinsic intrinsic = CreateIntrinsicWithKeyTypes<int64_t, int64_t>(
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
  Intrinsic intrinsic = CreateIntrinsicWithKeyTypes<int64_t, int64_t>(
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

// Third: Check that noise is added. A noised sum should not be the same as
// the unnoised sum; as epsilon decreases, the scale of noise will increase.
TEST(DPClosedDomainHistogramTest, NoiseAddedForSmallEpsilons) {
  Intrinsic intrinsic =
      CreateIntrinsicWithKeyTypes<int32_t, int64_t>(/*epsilon=*/0.01,
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

  // The report should encode the following noisy histogram
  // {a: num_inputs + noise, b: num_inputs + noise, c: 0 + noise}

  // There must be 2 columns, one for keys and one for aggregated values.
  ASSERT_EQ(report->size(), 2);

  const auto& values = report.value()[1].AsSpan<int64_t>();

  // There must be 3 rows, one per key (a, b, c)
  ASSERT_EQ(values.size(), 3);

  // We expect that there is some perturbation in at least one output.
  // Specifically, (num_inputs + noise, num_inputs + noise, noise) should not
  // match (num_inputs, num_inputs, 0)
  EXPECT_THAT(values, Not(ElementsAre(num_inputs, num_inputs, 0)));
}

// Ensure that we have floating point output when we request it.
TEST(DPClosedDomainHistogramTest, FloatTest) {
  Intrinsic intrinsic =
      CreateIntrinsicWithKeyTypes<float, float>(/*epsilon=*/0.01,
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
        Tensor::Create(DT_FLOAT, {2}, CreateTestData<float>({1, 0})).value();
    auto acc_status = aggregator->Accumulate({&keys, &values});
    EXPECT_THAT(acc_status, IsOk());
  }
  EXPECT_EQ(aggregator->GetNumInputs(), num_inputs);
  EXPECT_TRUE(aggregator->CanReport());

  auto report = std::move(*aggregator).Report();
  EXPECT_THAT(report, IsOk());

  // There must be 2 columns, one for keys and one for aggregated values.
  ASSERT_EQ(report->size(), 2);

  // The type of the noisy values should be float.
  ASSERT_EQ(report.value()[1].dtype(), DT_FLOAT);

  // Because the output spec calls for floats and our noise-generating code
  // should sample according to the output spec, we expect that the fractional
  // part of each noisy value is non-zero.
  auto noisy_values = report.value()[1].AsSpan<float>();
  for (float noisy_value : noisy_values) {
    noisy_value = std::abs(noisy_value);
    EXPECT_THAT(noisy_value - std::floor(noisy_value), Ne(0.0));
  }
}

}  // namespace
}  // namespace aggregation
}  // namespace tensorflow_federated
