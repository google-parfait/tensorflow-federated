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
#include <utility>
#include <vector>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/strings/str_cat.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/datatype.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_closed_domain_histogram.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_fedsql_constants.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_histogram_test_utils.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_open_domain_histogram.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/intrinsic.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator_registry.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/test_data.h"
#include "tensorflow_federated/cc/testing/status_matchers.h"

namespace tensorflow_federated {
namespace aggregation {
namespace {

using ::testing::HasSubstr;

using ::tensorflow_federated::aggregation::dp_histogram_testing::
    CreateIntrinsic;
using ::tensorflow_federated::aggregation::dp_histogram_testing::
    CreateIntrinsicWithKeyTypes;
using ::tensorflow_federated::aggregation::dp_histogram_testing::
    CreateIntrinsicWithMinContributors;
using ::tensorflow_federated::aggregation::dp_histogram_testing::
    CreateNestedParameters;
using ::tensorflow_federated::aggregation::dp_histogram_testing::
    CreateTensorSpec;
using ::tensorflow_federated::aggregation::dp_histogram_testing::
    CreateTopLevelParameters;
using ::testing::HasSubstr;
using ::testing::TestWithParam;

using DPClosedDomainHistogramTest = TestWithParam<bool>;
using DPOpenDomainHistogramTest = TestWithParam<bool>;

// Phase 1:Test that the factory catches problems in Intrinsics that are set up
// for the closed-domain case.
std::vector<Tensor> CreateTopLevelParameters_ClosedDomain() {
  return CreateTopLevelParameters<double, double, int64_t>(1000.0, 0.001, 100,
                                                           {DT_STRING});
}

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
  parameters[5] = Tensor::Create(DT_STRING, {3},
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
      .parameters = {CreateTopLevelParameters_ClosedDomain()},
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
      .parameters = {CreateTopLevelParameters_ClosedDomain()},
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
      .parameters = {CreateTopLevelParameters_ClosedDomain()},
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
      .parameters = {CreateTopLevelParameters_ClosedDomain()},
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
      .parameters = {CreateTopLevelParameters_ClosedDomain()},
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

// Phase 2: Test that the factory catches problems in Intrinsics that are set up
// for the open-domain case.
std::vector<Tensor> CreateTopLevelParameters_OpenDomain() {
  return CreateTopLevelParameters(1000.0, 0.001, 100);
}

TEST(DPOpenDomainHistogramTest, CatchTooFewParameters) {
  std::vector<Tensor> parameters;
  auto epsilon_tensor = CreateTestData({1.0});
  parameters.push_back(
      Tensor::Create(DT_DOUBLE, {}, std::move(epsilon_tensor), "epsilon")
          .value());

  Intrinsic too_few = {.uri = kDPGroupByUri,
                       .inputs = {CreateTensorSpec("key", DT_STRING)},
                       .outputs = {CreateTensorSpec("key_out", DT_STRING)},
                       .parameters = {std::move(parameters)},
                       .nested_intrinsics = {}};
  EXPECT_THAT(CreateTensorAggregator(too_few),
              StatusIs(INVALID_ARGUMENT,
                       HasSubstr("For all DP histograms, epsilon, delta, and "
                                 "max_groups_contributed must be provided")));
}

TEST(DPOpenDomainHistogramTest, CatchInvalidParameterTypes) {
  Intrinsic intrinsic0{
      .uri = kDPGroupByUri,
      .inputs = {CreateTensorSpec("key", DT_STRING)},
      .outputs = {CreateTensorSpec("key_out", DT_STRING)},
      .parameters = {CreateTopLevelParameters<string_view, double, int64_t>(
          "x", 0.1, 10)},
      .nested_intrinsics = {}};
  auto bad_epsilon = CreateTensorAggregator(intrinsic0).status();
  EXPECT_THAT(bad_epsilon, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(bad_epsilon.message(), HasSubstr("must be numerical"));

  Intrinsic intrinsic1{
      .uri = kDPGroupByUri,
      .inputs = {CreateTensorSpec("key", DT_STRING)},
      .outputs = {CreateTensorSpec("key_out", DT_STRING)},
      .parameters = {CreateTopLevelParameters<double, string_view, int64_t>(
          1.0, "x", 10)},
      .nested_intrinsics = {}};
  auto bad_delta = CreateTensorAggregator(intrinsic1).status();
  EXPECT_THAT(bad_delta, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(bad_delta.message(), HasSubstr("must be numerical"));

  Intrinsic intrinsic2{
      .uri = kDPGroupByUri,
      .inputs = {CreateTensorSpec("key", DT_STRING)},
      .outputs = {CreateTensorSpec("key_out", DT_STRING)},
      .parameters = {CreateTopLevelParameters<double, double, string_view>(
          1.0, 0.1, "x")},
      .nested_intrinsics = {}};
  auto bad_l0_bound = CreateTensorAggregator(intrinsic2).status();
  EXPECT_THAT(bad_l0_bound, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(bad_l0_bound.message(), HasSubstr("must be numerical"));
}

TEST(DPOpenDomainHistogramTest, CatchInvalidParameterValues) {
  Intrinsic intrinsic0 = CreateIntrinsic<int64_t, int64_t>(-1, 0.001, 10);
  auto bad_epsilon = CreateTensorAggregator(intrinsic0).status();
  EXPECT_THAT(bad_epsilon, StatusIs(INVALID_ARGUMENT,
                                    HasSubstr("Epsilon must be positive")));

  Intrinsic intrinsic1 = CreateIntrinsic<int64_t, int64_t>(1.0, -1, 10);
  auto bad_delta = CreateTensorAggregator(intrinsic1).status();
  EXPECT_THAT(bad_delta, StatusIs(INVALID_ARGUMENT,
                                  HasSubstr("delta must lie between 0 and 1")));

  Intrinsic intrinsic2 = CreateIntrinsic<int64_t, int64_t>(1.0, 0.001, -1);
  auto bad_l0_bound = CreateTensorAggregator(intrinsic2).status();
  EXPECT_THAT(bad_l0_bound,
              StatusIs(INVALID_ARGUMENT,
                       HasSubstr("max_groups_contributed must be positive")));
}

TEST(DPOpenDomainHistogramTest, CatchInvalidLinfinityBound) {
  Intrinsic intrinsic =
      CreateIntrinsic<int64_t, int64_t>(1.0, 0.001, 10, -1, 2, 3);
  auto aggregator_status = CreateTensorAggregator(intrinsic).status();
  EXPECT_THAT(aggregator_status,
              StatusIs(INVALID_ARGUMENT,
                       HasSubstr("must provide a positive Linfinity bound.")));
}

TEST(DPOpenDomainHistogramTest, CatchUnsupportedNestedIntrinsic) {
  Intrinsic intrinsic = {.uri = kDPGroupByUri,
                         .inputs = {CreateTensorSpec("key", DT_STRING)},
                         .outputs = {CreateTensorSpec("key_out", DT_STRING)},
                         .parameters = {CreateTopLevelParameters_OpenDomain()},
                         .nested_intrinsics = {}};
  intrinsic.nested_intrinsics.push_back(Intrinsic{
      "GoogleSQL:$not_differential_privacy_sum",
      {CreateTensorSpec("value", internal::TypeTraits<int32_t>::kDataType)},
      {CreateTensorSpec("value", internal::TypeTraits<int32_t>::kDataType)},
      {CreateNestedParameters<int32_t>(1000, -1, -1)},
      {}});
  auto aggregator_status = CreateTensorAggregator(intrinsic).status();
  EXPECT_THAT(aggregator_status, StatusIs(UNIMPLEMENTED));
  EXPECT_THAT(aggregator_status.message(), HasSubstr("Currently, only nested "
                                                     "DP sums are supported"));
}

// Phase 3: Test that the factory can create DPClosedDomainHistogram objects.
TEST(DPClosedDomainHistogramTest, CreateWithPositiveDelta) {
  // Only L2 bound
  std::vector<DataType> key_types = {DT_STRING, DT_STRING};
  Intrinsic intrinsic1 = CreateIntrinsicWithKeyTypes<int32_t, int64_t>(
      /*epsilon=*/1, /*delta=*/0.001, /*l0_bound=*/-1,
      /*linfinity_bound=*/-1, /*l1_bound=*/-1, /*l2_bound=*/10, key_types);

  // Both Linf and L0 bounds
  TFF_EXPECT_OK(CreateTensorAggregator(intrinsic1).status());
  Intrinsic intrinsic2 = CreateIntrinsicWithKeyTypes<int32_t, int64_t>(
      /*epsilon=*/1, /*delta=*/0.001, /*l0_bound=*/1,
      /*linfinity_bound=*/5, /*l1_bound=*/-1, /*l2_bound=*/-1, key_types);
  TFF_EXPECT_OK(CreateTensorAggregator(intrinsic2).status());
}
TEST(DPClosedDomainHistogramTest, CreateWithZeroDelta) {
  // Only L1 bound
  std::vector<DataType> key_types = {DT_STRING, DT_STRING};
  Intrinsic intrinsic1 = CreateIntrinsicWithKeyTypes<int32_t, int64_t>(
      /*epsilon=*/1, /*delta=*/0, /*l0_bound=*/-1,
      /*linfinity_bound=*/-1, /*l1_bound=*/10, /*l2_bound=*/-1, key_types);
  TFF_EXPECT_OK(CreateTensorAggregator(intrinsic1).status());

  // Both Linf and L0 bounds
  Intrinsic intrinsic2 = CreateIntrinsicWithKeyTypes<int32_t, int64_t>(
      /*epsilon=*/1, /*delta=*/0, /*l0_bound=*/1,
      /*linfinity_bound=*/5, /*l1_bound=*/-1, /*l2_bound=*/-1, key_types);
  TFF_EXPECT_OK(CreateTensorAggregator(intrinsic2).status());
}

// Phase 4: Test that the factory can create DPOpenDomainHistogram objects.
TEST(DPOpenDomainHistogramTest, CreateWithGroupingKeys) {
  Intrinsic intrinsic = CreateIntrinsic<int32_t, int64_t>(
      /*epsilon=*/1, /*delta=*/0.001, /*l0_bound=*/1,
      /*linfinity_bound=*/5);
  TFF_EXPECT_OK(CreateTensorAggregator(intrinsic).status());
}
TEST(DPOpenDomainHistogramTest, CreateWithNoGroupingKeys) {
  Intrinsic intrinsic = {
      .uri = kDPGroupByUri,
      .inputs = {},
      .outputs = {},
      .parameters = {CreateTopLevelParameters(
          /*epsilon=*/1.0,
          /*delta=*/0.001, /*l0_bound=max_groups_contributed=*/-1)},
      .nested_intrinsics = {}};
  intrinsic.nested_intrinsics.push_back(
      Intrinsic{kDPSumUri,
                {CreateTensorSpec("value", DT_INT32)},
                {CreateTensorSpec("value", DT_INT64)},
                {CreateNestedParameters<int32_t>(1000, -1, -1)},
                {}});
  StatusOr<std::unique_ptr<TensorAggregator>> aggregator =
      CreateTensorAggregator(intrinsic);
  TFF_EXPECT_OK(aggregator.status());
  // The max_groups_contributed parameter should be changed from -1 to 1 because
  // 1 simulated group will be made to aggregate all elements.
  DPOpenDomainHistogramPeer peer(std::move(aggregator).value());
  EXPECT_EQ(peer.GetMaxGroupsContributed(), 1);
}

// Phase 5: Test that the factory handles min_contributors_to_group correctly.
TEST(DPGroupByFactoryTest, CreateAggregatorWithMinContributorsNoKeys) {
  Intrinsic intrinsic = CreateIntrinsicWithMinContributors<int64_t, int64_t>(
      /*min_contributors=*/10, /*key_types=*/{});
  TFF_ASSERT_OK_AND_ASSIGN(auto aggregator, CreateTensorAggregator(intrinsic));
  // Check that the returned aggregator is a DPOpenDomainHistogram.
  auto* dp_open_domain_histogram =
      dynamic_cast<DPOpenDomainHistogram*>(aggregator.get());
  ASSERT_NE(dp_open_domain_histogram, nullptr);
}

TEST(DPGroupByFactoryTest, CreateAggregatorWithMinContributorsWithKeys) {
  Intrinsic intrinsic = CreateIntrinsicWithMinContributors<int64_t, int64_t>(
      /*min_contributors=*/10, /*key_types=*/{DT_STRING});
  TFF_ASSERT_OK_AND_ASSIGN(auto aggregator, CreateTensorAggregator(intrinsic));
  // Check that the returned aggregator is a DPOpenDomainHistogram.
  auto* dp_open_domain_histogram =
      dynamic_cast<DPOpenDomainHistogram*>(aggregator.get());
  ASSERT_NE(dp_open_domain_histogram, nullptr);
}

TEST(DPGroupByFactoryTest, CreateAggregatorWithZeroMinContributors_Fails) {
  Intrinsic intrinsic = CreateIntrinsicWithMinContributors<int64_t, int64_t>(
      /*min_contributors=*/0);
  EXPECT_THAT(
      CreateTensorAggregator(intrinsic),
      StatusIs(INVALID_ARGUMENT,
               HasSubstr("min_contributors_to_group must be positive")));
}

TEST(DPGroupByFactoryTest, CreateAggregatorWithNegativeMinContributors_Fails) {
  Intrinsic intrinsic = CreateIntrinsicWithMinContributors<int64_t, int64_t>(
      /*min_contributors=*/-1);
  EXPECT_THAT(
      CreateTensorAggregator(intrinsic),
      StatusIs(INVALID_ARGUMENT,
               HasSubstr("min_contributors_to_group must be positive")));
}

// Make sure we can successfully create a DPOpenDomainHistogram object with no
// keys.
TEST(DPGroupByFactoryTest, CreateAggregatorNoKeys_Success) {
  // Create intrinsic with default parameters except no key_types.
  Intrinsic intrinsic = CreateIntrinsicWithKeyTypes<int64_t, int64_t>(
      kEpsilonThreshold, 0.001, 100, 100, -1, -1, /*key_types=*/{});
  TFF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<TensorAggregator> aggregator,
                           CreateTensorAggregator(intrinsic));

  // Validate that domain_tensors is a valid empty span.
  auto* dp_open_domain_histogram =
      dynamic_cast<DPOpenDomainHistogram*>(aggregator.get());
  ASSERT_NE(dp_open_domain_histogram, nullptr);
}

TEST(DPGroupByFactoryTest, CreateAggregatorWithKeysNoMinContributors_Success) {
  // Create intrinsic with default parameters and key_types, and no min
  // contributors.
  Intrinsic intrinsic = CreateIntrinsicWithKeyTypes<int64_t, int64_t>(
      kEpsilonThreshold, 0.001, 100, 100, -1, -1, /*key_types=*/{DT_STRING});
  TFF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<TensorAggregator> aggregator,
                           CreateTensorAggregator(intrinsic));

  // Validate that the aggregator is a DPClosedDomainHistogram.
  auto* dp_closed_domain_histogram =
      dynamic_cast<DPClosedDomainHistogram*>(aggregator.get());
  ASSERT_NE(dp_closed_domain_histogram, nullptr);
}

// Phase 6: Test that the factory can create aggregators with custom choice of
// max_string_length.
TEST(DPGroupByFactoryTest, CreateAggregatorWithMaxStringLength) {
  Intrinsic intrinsic = CreateIntrinsicWithKeyTypes<int64_t, int64_t>(
      /*epsilon=*/1, /*delta=*/1e-6, /*l0_bound=*/1, /*linfinity_bound=*/1,
      /*l1_bound=*/-1, /*l2_bound=*/-1, /*key_types=*/{DT_STRING});
  intrinsic.parameters.push_back(Tensor::Create(DT_INT32, {},
                                                CreateTestData({2}),
                                                /*name=*/"max_string_length")
                                     .value());
  TFF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<TensorAggregator> aggregator,
                           CreateTensorAggregator(intrinsic));

  TFF_ASSERT_OK_AND_ASSIGN(
      Tensor short_keys,
      Tensor::Create(DT_STRING, {2}, CreateTestData<string_view>({"ab", "b"})));
  TFF_ASSERT_OK_AND_ASSIGN(
      Tensor short_values,
      Tensor::Create(DT_INT64, {2}, CreateTestData<int64_t>({1, 2})));
  TFF_EXPECT_OK(aggregator->ValidateInputs({&short_keys, &short_values}));

  TFF_ASSERT_OK_AND_ASSIGN(
      Tensor long_keys,
      Tensor::Create(DT_STRING, {2},
                     CreateTestData<string_view>({"abc", "bcde"})));
  TFF_ASSERT_OK_AND_ASSIGN(
      Tensor long_values,
      Tensor::Create(DT_INT64, {2}, CreateTestData<int64_t>({1, 2})));
  EXPECT_THAT(aggregator->ValidateInputs({&long_keys, &long_values}),
              StatusIs(INVALID_ARGUMENT, HasSubstr("got a string exceeding that"
                                                   " length in tensor 0")));
}

}  // namespace
}  // namespace aggregation
}  // namespace tensorflow_federated
