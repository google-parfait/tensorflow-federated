// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow_federated/cc/core/impl/aggregation/protocol/simple_aggregation/latency_aggregator.h"

#include <cmath>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/time/time.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/executors/status_matchers.h"

namespace tensorflow_federated::aggregation {
namespace {

using ::testing::AllOf;
using ::testing::Ge;
using ::testing::Le;

auto DurationApproxEq(absl::Duration d) {
  return AllOf(Ge(d - absl::Nanoseconds(1)), Le(d + absl::Nanoseconds(1)));
}

TEST(LatencyAggregatorTest, TestZeroSamples) {
  LatencyAggregator aggregator;
  EXPECT_EQ(aggregator.GetCount(), 0);
  EXPECT_EQ(aggregator.GetMean(), absl::ZeroDuration());
  EXPECT_THAT(aggregator.GetStandardDeviation(), StatusIs(FAILED_PRECONDITION));
}

TEST(LatencyAggregatorTest, TestOneSample) {
  LatencyAggregator aggregator;
  aggregator.Add(absl::Seconds(1));
  EXPECT_EQ(aggregator.GetCount(), 1);
  EXPECT_EQ(aggregator.GetMean(), absl::Seconds(1));
  EXPECT_THAT(aggregator.GetStandardDeviation(), StatusIs(FAILED_PRECONDITION));
}

TEST(LatencyAggregatorTest, TestCount) {
  LatencyAggregator aggregator;
  aggregator.Add(absl::Seconds(1));
  aggregator.Add(absl::Seconds(2));
  EXPECT_EQ(aggregator.GetCount(), 2);
  aggregator.Add(absl::Seconds(3));
  EXPECT_EQ(aggregator.GetCount(), 3);
}

TEST(LatencyAggregatorTest, TestMean) {
  LatencyAggregator aggregator;
  aggregator.Add(absl::Seconds(1));
  aggregator.Add(absl::Seconds(3));
  EXPECT_EQ(aggregator.GetMean(), absl::Seconds(2));
  aggregator.Add(absl::Seconds(5));
  EXPECT_EQ(aggregator.GetMean(), absl::Seconds(3));
}

TEST(LatencyAggregatorTest, TestStandardDeviation) {
  LatencyAggregator aggregator;
  aggregator.Add(absl::Seconds(1));
  aggregator.Add(absl::Seconds(1));
  auto standard_deviation = aggregator.GetStandardDeviation();
  EXPECT_THAT(standard_deviation, IsOk());
  EXPECT_EQ(standard_deviation.value(), absl::ZeroDuration());
}

TEST(LatencyAggregatorTest, TestSequence) {
  // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Two-pass_algorithm
  //
  // mean = Sum(i)/n
  // sample_var = Sum((i - mean)^2)/(n-1)
  // sample_std_dev = Sqrt(sample_var)

  LatencyAggregator aggregator;
  double latencies_sec[] = {3,  3,  4,  4,   7,   8,   8,   10,  12,  12,
                            12, 15, 79, 131, 241, 299, 453, 484, 503, 1001};
  double sum_sec = 0;
  int count = sizeof(latencies_sec) / sizeof(double);

  for (double latency_sec : latencies_sec) {
    sum_sec += latency_sec;
    aggregator.Add(absl::Seconds(latency_sec));
  }

  double mean = sum_sec / count;
  double sum_of_squares = 0.0;
  for (double latency_sec : latencies_sec) {
    double delta_from_mean = latency_sec - mean;
    sum_of_squares += delta_from_mean * delta_from_mean;
  }

  absl::Duration expected_mean = absl::Seconds(mean);
  absl::Duration expected_std_dev =
      absl::Seconds(std::sqrt(sum_of_squares / (count - 1)));

  EXPECT_EQ(aggregator.GetCount(), count);
  EXPECT_THAT(aggregator.GetMean(), DurationApproxEq(expected_mean));
  auto standard_deviation = aggregator.GetStandardDeviation();
  EXPECT_THAT(standard_deviation, IsOk());
  EXPECT_THAT(standard_deviation.value(), DurationApproxEq(expected_std_dev));
}

}  // namespace
}  // namespace tensorflow_federated::aggregation
