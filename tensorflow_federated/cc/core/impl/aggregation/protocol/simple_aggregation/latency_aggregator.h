/*
 * Copyright 2024 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_PROTOCOL_SIMPLE_AGGREGATION_LATENCY_AGGREGATOR_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_PROTOCOL_SIMPLE_AGGREGATION_LATENCY_AGGREGATOR_H_

#include <cmath>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"

namespace tensorflow_federated::aggregation {

// Utility class used to calculate mean and standard deviation of client
// participation in the Aggregation Protocol.
//
// This class implements Welford's online algorithm:
// https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
//
// This class isn't thread-safe. The caller must ensure that the
// calls are strictly sequential.
class LatencyAggregator final {
 public:
  void Add(absl::Duration latency) {
    double latency_sec = absl::ToDoubleSeconds(latency);
    count_++;
    double delta = latency_sec - mean_;
    mean_ += delta / count_;
    double delta2 = latency_sec - mean_;
    sum_of_squares_ += delta * delta2;
  }

  size_t GetCount() const { return count_; }

  absl::Duration GetMean() const { return absl::Seconds(mean_); }

  absl::StatusOr<absl::Duration> GetStandardDeviation() const {
    if (count_ < 2) {
      return absl::FailedPreconditionError(
          "At least 2 latency samples required");
    }
    return absl::Seconds(std::sqrt(sum_of_squares_ / (count_ - 1)));
  }

 private:
  size_t count_ = 0;
  double mean_ = 0.0;
  double sum_of_squares_ = 0.0;
};

}  // namespace tensorflow_federated::aggregation

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_PROTOCOL_SIMPLE_AGGREGATION_LATENCY_AGGREGATOR_H_
