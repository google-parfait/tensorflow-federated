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
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/agg_vector.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/datatype.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_composite_key_combiner.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_fedsql_constants.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_histogram_test_utils.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/intrinsic.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator_registry.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_shape.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/test_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/testing.h"
#include "tensorflow_federated/cc/testing/status_matchers.h"

namespace tensorflow_federated {
namespace aggregation {
namespace {

using ::testing::ElementsAre;
using ::testing::Ne;
using ::testing::Not;
using ::testing::TestWithParam;

using ::tensorflow_federated::aggregation::dp_histogram_testing::
    CreateIntrinsicWithKeyTypes_ClosedDomain;
using ::tensorflow_federated::aggregation::dp_histogram_testing::
    CreateTensorSpec;

using DPClosedDomainHistogramTest = TestWithParam<bool>;

// First batch of tests validate the aggregator itself, without DP noise.

// Make sure we can successfully create a DPClosedDomainHistogram object.
TEST(DPClosedDomainHistogramTest, CreateAggregator_Success) {
  Intrinsic intrinsic =
      CreateIntrinsicWithKeyTypes_ClosedDomain<int64_t, int64_t>();
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
  Intrinsic intrinsic =
      CreateIntrinsicWithKeyTypes_ClosedDomain<int64_t, int64_t>(
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
  Intrinsic intrinsic =
      CreateIntrinsicWithKeyTypes_ClosedDomain<int64_t, int64_t>(
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
  Intrinsic intrinsic =
      CreateIntrinsicWithKeyTypes_ClosedDomain<int64_t, int64_t>(
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

// Second batch of tests: check that noise is added. A noised sum should not be
// the same as the unnoised sum; as epsilon decreases, the scale of noise will
// increase.
TEST(DPClosedDomainHistogramTest, NoiseAddedForSmallEpsilons) {
  Intrinsic intrinsic =
      CreateIntrinsicWithKeyTypes_ClosedDomain<int32_t, int64_t>(
          /*epsilon=*/0.01,
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
  Intrinsic intrinsic = CreateIntrinsicWithKeyTypes_ClosedDomain<float, float>(
      /*epsilon=*/0.01,
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
